"""
trainer.py — one training loop for both pretraining and SFT, single- or multi-GPU.

The only difference between the two modes is the loss mask: pretrain batches are
(x, y); SFT batches are (x, y, mask). masked_cross_entropy with mask=None is
mathematically identical to plain mean cross-entropy, so a single loop covers both.

Multi-GPU is DDP: when launched under torchrun (WORLD_SIZE>1) the model is wrapped
in DistributedDataParallel, gradients all-reduce at the accumulation boundary,
validation loss is averaged across ranks, and only rank 0 logs / writes checkpoints.
When run single-process everything below degrades to the original behavior.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from ..utils.distributed import (
    is_distributed,
    is_main,
    get_local_rank,
    all_reduce_mean,
)
from ..utils.logging import TrainingLogger
from .checkpoint import save_best, save_last, save_checkpoint
from .optimizer import build_optimizer
from .scheduler import cosine_lr


def masked_cross_entropy(logits, target, mask=None):
    """
    logits : (B, T, V)
    target : (B, T)
    mask   : (B, T) float, 1 = include token in loss. None -> plain mean CE.
    """
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    target_flat = target.reshape(B * T)

    if mask is None:
        return F.cross_entropy(logits_flat, target_flat)

    loss_per_token = F.cross_entropy(logits_flat, target_flat, reduction="none")
    mask_flat = mask.reshape(B * T)
    return (loss_per_token * mask_flat).sum() / mask_flat.sum().clamp(min=1e-8)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,                 # may be None
        training_cfg,
        ckpt_dir,
        *,
        method: str | None = None,
        meta: dict | None = None,
        run_name: str | None = None,
        run_id: int | None = None,
        seq_len: int = 512,
        mode: str = "train",
    ):
        self.device = training_cfg.device
        self.raw_model = model.to(self.device)

        # wrap for DDP when distributed; raw_model stays the unwrapped reference
        # used for optimizer construction, grad clipping, and checkpoint saving.
        if is_distributed():
            ddp_kwargs = {}
            if str(self.device).startswith("cuda"):
                ddp_kwargs = {"device_ids": [get_local_rank()],
                              "output_device": get_local_rank()}
            self.model = DDP(self.raw_model, **ddp_kwargs)
        else:
            self.model = self.raw_model

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = training_cfg
        self.ckpt_dir = Path(ckpt_dir)
        self.method = method
        self.meta = meta or {}

        self.optimizer = build_optimizer(self.raw_model, training_cfg)
        self.step = 0
        self.best_val_loss = float("inf")

        self.logger = TrainingLogger(
            seq_len=seq_len,
            batch_size=training_cfg.batch_size,
            run_name=run_name,
            run_id=run_id,
            mode=mode,
        )
        self.progress = tqdm(
            total=training_cfg.total_steps,
            desc="training",
            dynamic_ncols=True,
            leave=True,
        ) if is_main() else None

    # ── helpers ──────────────────────────────────────────────
    def _split_batch(self, batch):
        if len(batch) == 3:
            x, y, mask = batch
            mask = mask.to(self.device, non_blocking=True)
        else:
            x, y = batch
            mask = None
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        return x, y, mask

    def _apply_lr(self):
        lr_mult = cosine_lr(self.step, self.config)
        for g in self.optimizer.param_groups:
            g["lr"] = self.config.lr * lr_mult

    def _set_epoch(self, epoch: int):
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def grad_norm(self) -> float:
        total = 0.0
        for p in self.raw_model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    # ── training loop ────────────────────────────────────────
    def train(self):
        config = self.config
        model = self.model
        optimizer = self.optimizer
        scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision)

        model.train()
        grad_norm = 0.0
        last_val = float("inf")
        epoch = 0

        self._apply_lr()

        while self.step < config.total_steps:
            self._set_epoch(epoch)
            for batch in self.train_loader:
                x, y, mask = self._split_batch(batch)

                accum_boundary = (self.step + 1) % config.grad_accum == 0
                # skip the inter-GPU all-reduce on non-boundary micro-steps
                sync_ctx = (model.no_sync()
                            if (is_distributed() and not accum_boundary)
                            else nullcontext())

                with sync_ctx:
                    with torch.amp.autocast("cuda", enabled=config.mixed_precision):
                        logits = model(x)
                        loss = masked_cross_entropy(logits, y, mask) / config.grad_accum
                    scaler.scale(loss).backward()

                if accum_boundary:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(), 1.0
                    ).item()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    self._apply_lr()

                if self.step % 100 == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    self.logger.train(
                        self.step, loss.item() * config.grad_accum, lr, grad_norm
                    )

                # ── evaluation + best checkpoint ──────────────
                if (self.val_loader is not None
                        and self.step % config.eval_interval == 0
                        and self.step != 0):
                    last_val = self.evaluate()
                    self.logger.eval(self.step, last_val)
                    if last_val < self.best_val_loss:
                        self.best_val_loss = last_val
                        if is_main():
                            save_best(
                                self.raw_model, optimizer, self.step, self.ckpt_dir,
                                method=self.method, val_loss=last_val, meta=self.meta,
                            )
                            self.logger.checkpoint(self.step, last_val)

                # ── periodic checkpoint (rank 0 only) ─────────
                if self.step % config.save_interval == 0 and self.step != 0 and is_main():
                    save_checkpoint(
                        self.raw_model, optimizer, self.step,
                        self.ckpt_dir / f"step_{self.step:06d}.pt",
                        method=self.method,
                        val_loss=last_val if last_val != float("inf") else None,
                        meta=self.meta,
                    )

                self.step += 1
                if self.progress is not None:
                    self.progress.update(1)
                if self.step >= config.total_steps:
                    break
            epoch += 1

        if self.progress is not None:
            self.progress.close()

        # always leave a final "last" checkpoint (rank 0)
        if is_main():
            save_last(
                self.raw_model, optimizer, self.step, self.ckpt_dir,
                method=self.method,
                val_loss=last_val if last_val != float("inf") else None,
                meta=self.meta,
            )

    # ── evaluation ───────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses = []
        for batch in self.val_loader:
            x, y, mask = self._split_batch(batch)
            logits = self.model(x)
            losses.append(masked_cross_entropy(logits, y, mask).item())
        self.model.train()
        local_mean = sum(losses) / max(len(losses), 1)
        # average across ranks so every rank agrees on the best-checkpoint decision
        return all_reduce_mean(local_mean, self.device)
