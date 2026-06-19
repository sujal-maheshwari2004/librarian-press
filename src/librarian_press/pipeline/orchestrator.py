"""
orchestrator.py — run pretrain / sft / both with manifest-based resume.

Each data stage is gated by its manifest (skipped when already complete).
train/eval stages always run and manage their own checkpoints. In mode "both",
the SFT base checkpoint and tokenizer are auto-wired from the pretrain outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config.schema import RunConfig
from ..config.paths import RunPaths
from ..data.dataset import PackedDataset, FinetuneDataset
from ..data.pack import run_pack
from ..data.prepare_sft import run_prepare_sft
from ..data.shard import ingest_pretrain, ingest_sft
from ..data.tokenize import run_tokenize
from ..evaluation.evaluator import evaluate_pretrain, evaluate_sft
from ..model.build import build_model, load_for_sft
from ..tokenizer.train import train_tokenizer
from ..tokenizer.load import load_tokenizer
from ..training.checkpoint import load_checkpoint, save_checkpoint
from ..training.trainer import Trainer
from ..utils.device import resolve_device, adjust_amp
from ..utils.distributed import is_distributed, is_main, barrier, maybe_local_device
from ..utils.logging import StageLogger
from .cleanup import safe_delete_stage
from .manifest import StageManifest, file_checksum
from .stages import PRETRAIN_STAGES, SFT_STAGES, DATA_STAGES, MAIN_ONLY_STAGES


def _make_loaders(train_ds, val_ds, batch_size, shuffle=True):
    """Build train/val DataLoaders, using DistributedSampler under DDP."""
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_ds, shuffle=shuffle)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=train_sampler, drop_last=True)
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(val_ds, batch_size=batch_size,
                                    sampler=DistributedSampler(val_ds, shuffle=False))
        return train_loader, val_loader

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds is not None else None
    return train_loader, val_loader


# ── helpers ──────────────────────────────────────────────────────────
def _complete(manifest_path: Path) -> bool:
    if not Path(manifest_path).exists():
        return False
    return StageManifest(manifest_path).is_complete()


def _resolve_ckpt(ckpt_dir: Path) -> Path | None:
    best = ckpt_dir / "best.pt"
    last = ckpt_dir / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    return None


def _tokenizer_meta(tokenizer_path: str) -> dict:
    p = Path(tokenizer_path)
    if not p.exists():
        return {}
    try:
        vocab = load_tokenizer(p).get_vocab_size()
    except Exception:
        vocab = None
    return {"tokenizer_sha": file_checksum(p), "tokenizer_vocab": vocab}


def _save_eval(paths: RunPaths, name: str, results: dict, extra: dict):
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    out = paths.log_dir / f"{name}.json"
    with open(out, "w") as f:
        json.dump({**extra, "results": results}, f, indent=2)
    print(f"\n=== {name} results ===")
    for k, v in results.items():
        print(f"  {k:<16} {v}")
    print(f"Saved to: {out}")


# ── pretrain train / eval ────────────────────────────────────────────
def _train_pretrain(cfg: RunConfig, paths: RunPaths, resume, run_id, stage_log):
    sec = cfg.pretrain
    device = maybe_local_device(resolve_device(sec.training.device))
    adjust_amp(sec.training, device)
    sec.training.device = device

    model = build_model(cfg.model)
    if is_main():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    seq_len = sec.data.seq_len
    train_ds = PackedDataset(paths.pretrain_data_dir / "train_packed.bin", seq_len)
    val_path = paths.pretrain_data_dir / "validation_packed.bin"
    val_ds = PackedDataset(val_path, seq_len) if val_path.exists() else None

    train_loader, val_loader = _make_loaders(train_ds, val_ds, sec.training.batch_size)
    if is_main():
        print(f"Train sequences: {len(train_ds):,}"
              + (f" | Val sequences: {len(val_ds):,}" if val_ds else " | no validation split"))

    ckpt_dir = paths.checkpoint_dir("pretrain")
    meta = {**_tokenizer_meta(cfg.tokenizer.path), "vocab_size": cfg.model.vocab_size}

    trainer = Trainer(
        model, train_loader, val_loader, sec.training, ckpt_dir,
        method=None, meta=meta, run_name=cfg.name, run_id=run_id, seq_len=seq_len,
    )
    if resume:
        trainer.step = load_checkpoint(model, trainer.optimizer, resume)
    trainer.train()


def _eval_pretrain(cfg: RunConfig, paths: RunPaths, checkpoint, stage_log):
    sec = cfg.pretrain
    device = maybe_local_device(resolve_device(sec.training.device))
    ckpt = checkpoint or _resolve_ckpt(paths.checkpoint_dir("pretrain"))
    if ckpt is None:
        print("[eval_pre] no checkpoint found — skipping")
        return
    model = build_model(cfg.model)
    load_checkpoint(model, None, ckpt)
    model.to(device)

    seq_len = sec.data.seq_len
    eval_path = paths.pretrain_data_dir / "validation_packed.bin"
    if not eval_path.exists():
        eval_path = paths.pretrain_data_dir / "test_packed.bin"
    if not eval_path.exists():
        print("[eval_pre] no validation/test split — skipping")
        return
    loader = DataLoader(PackedDataset(eval_path, seq_len), batch_size=sec.training.batch_size)
    results = evaluate_pretrain(model, loader, device)
    _save_eval(paths, "eval_pretrain", results, {"run_name": cfg.name, "checkpoint": str(ckpt)})


# ── sft train / eval ─────────────────────────────────────────────────
def _train_sft(cfg: RunConfig, paths: RunPaths, resume, run_id, stage_log):
    sec = cfg.sft
    device = maybe_local_device(resolve_device(sec.training.device))
    adjust_amp(sec.training, device)
    sec.training.device = device

    model = load_for_sft(cfg.model, sec.finetune, device)

    seq_len = cfg.model.max_seq_len
    train_ds = FinetuneDataset(paths.sft_data_dir, sec.data.split_train, seq_len)
    val_ds = FinetuneDataset(paths.sft_data_dir, sec.data.split_val, seq_len)
    if is_main():
        print(f"Train examples: {len(train_ds):,} | Val examples: {len(val_ds):,}")

    train_loader, val_loader = _make_loaders(train_ds, val_ds, sec.training.batch_size)

    ckpt_dir = paths.checkpoint_dir(f"sft/{sec.finetune.method}")
    meta = {
        **_tokenizer_meta(cfg.tokenizer.path),
        "vocab_size": cfg.model.vocab_size,
        "lora_rank": sec.finetune.lora.rank,
        "lora_alpha": sec.finetune.lora.alpha,
    }
    trainer = Trainer(
        model, train_loader, val_loader, sec.training, ckpt_dir,
        method=sec.finetune.method, meta=meta,
        run_name=cfg.name, run_id=run_id, seq_len=seq_len,
    )
    if resume:
        trainer.step = load_checkpoint(model, trainer.optimizer, resume,
                                       method=sec.finetune.method)
    trainer.train()


def _eval_sft(cfg: RunConfig, paths: RunPaths, checkpoint, stage_log):
    sec = cfg.sft
    device = maybe_local_device(resolve_device(sec.training.device))
    ckpt = checkpoint or _resolve_ckpt(paths.checkpoint_dir(f"sft/{sec.finetune.method}"))
    if ckpt is None:
        print("[eval_sft] no checkpoint found — skipping")
        return

    model = load_for_sft(cfg.model, sec.finetune, device)
    load_checkpoint(model, None, ckpt, method=sec.finetune.method)
    model.eval()

    seq_len = cfg.model.max_seq_len
    val_ds = FinetuneDataset(paths.sft_data_dir, sec.data.split_val, seq_len)
    val_loader = DataLoader(val_ds, batch_size=sec.training.batch_size)
    tokenizer = load_tokenizer(cfg.tokenizer.path)

    results = evaluate_sft(model, sec.finetune, sec.data, paths, val_loader,
                           tokenizer, device, max_seq_len=seq_len)
    _save_eval(paths, "eval_sft", results,
               {"run_name": cfg.name, "method": sec.finetune.method,
                "metric": sec.finetune.eval_metric, "checkpoint": str(ckpt)})


# ── stage dispatch ───────────────────────────────────────────────────
def _run_stages(stages, handlers, stage_log):
    """
    Run stages with DDP-aware rank gating:
      - data + eval stages run on rank 0 only; other ranks wait at a barrier
      - train stages run on every rank (that's where DDP fans out)
    A barrier after each stage keeps all ranks aligned. All of this is a no-op
    when single-process (is_main() True, barrier() does nothing).
    """
    for stage in stages:
        main_only = stage in MAIN_ONLY_STAGES

        # non-main ranks skip main-only stages and wait for rank 0
        if main_only and not is_main():
            barrier()
            continue

        # manifest-based skip (data stages only)
        mget = handlers.get(f"_manifest_{stage}")
        if stage in DATA_STAGES and mget and _complete(mget()):
            print(f"[pipeline] '{stage}' already complete - skipping")
            barrier()
            continue

        if is_main():
            print(f"\n[pipeline] === stage: {stage} ===")
        stage_log.start(stage)
        try:
            summary = handlers[stage]()
        except Exception as e:
            stage_log.error(stage, str(e))
            print(f"[pipeline] FATAL in '{stage}': {e}")
            print(f"[pipeline] Fix and re-run with --start-from {stage}")
            raise
        stage_log.end(stage, summary if isinstance(summary, dict) else {})
        barrier()


# ── public entrypoints ───────────────────────────────────────────────
def run_pretrain(cfg, paths, start_from="ingest_pre", resume=None, cleanup=False,
                 run_id=None, stage_log=None):
    stage_log = stage_log or StageLogger(run_name=cfg.name, run_id=run_id)
    paths.make_dirs()
    stages = PRETRAIN_STAGES[PRETRAIN_STAGES.index(start_from):]

    def do_tokenizer():
        if Path(cfg.tokenizer.path).exists():
            print(f"[train_tokenizer] tokenizer exists at {cfg.tokenizer.path} — skipping")
            return {}
        if not cfg.tokenizer.train_if_missing:
            raise FileNotFoundError(
                f"Tokenizer not found at {cfg.tokenizer.path} and "
                "tokenizer.train_if_missing is false."
            )
        return train_tokenizer(cfg.tokenizer, paths.pretrain_data_dir / "raw", stage_log)

    handlers = {
        "ingest_pre": lambda: ingest_pretrain(cfg.pretrain.data, paths,
                                              paths.manifest("ingest_pre"), stage_log),
        "_manifest_ingest_pre": lambda: paths.manifest("ingest_pre"),
        "train_tokenizer": do_tokenizer,
        "tokenize": lambda: run_tokenize(paths, cfg.tokenizer.path, stage_log),
        "_manifest_tokenize": lambda: paths.manifest("tokenize"),
        "pack": lambda: run_pack(paths, cfg.pretrain.data.seq_len,
                                 cfg.pretrain.data.val_frac, cfg.pretrain.data.test_frac,
                                 stage_log),
        "_manifest_pack": lambda: paths.manifest("pack"),
        "train_pre": lambda: _train_pretrain(cfg, paths, resume, run_id, stage_log),
        "eval_pre": lambda: _eval_pretrain(cfg, paths, None, stage_log),
    }
    _run_stages(stages, handlers, stage_log)

    if cleanup:
        _cleanup_pretrain(paths, cfg.pretrain.data.seq_len)


def run_sft(cfg, paths, start_from="ingest_sft", resume=None, cleanup=False,
            run_id=None, stage_log=None):
    stage_log = stage_log or StageLogger(run_name=cfg.name, run_id=run_id)
    paths.make_dirs()

    if not Path(cfg.tokenizer.path).exists():
        raise FileNotFoundError(
            f"SFT requires an existing tokenizer at {cfg.tokenizer.path}"
        )
    _verify_tokenizer_against_base(cfg)

    stages = SFT_STAGES[SFT_STAGES.index(start_from):]
    handlers = {
        "ingest_sft": lambda: ingest_sft(cfg.sft.data, paths,
                                         paths.manifest("ingest_sft"), stage_log),
        "_manifest_ingest_sft": lambda: paths.manifest("ingest_sft"),
        "prepare_sft": lambda: run_prepare_sft(cfg.sft.data, paths, cfg.tokenizer.path, stage_log),
        "_manifest_prepare_sft": lambda: paths.manifest("prepare_sft"),
        "train_sft": lambda: _train_sft(cfg, paths, resume, run_id, stage_log),
        "eval_sft": lambda: _eval_sft(cfg, paths, None, stage_log),
    }
    _run_stages(stages, handlers, stage_log)


def run_all(cfg, paths, cleanup=False, run_id=None, stage_log=None):
    stage_log = stage_log or StageLogger(run_name=cfg.name, run_id=run_id)
    run_pretrain(cfg, paths, cleanup=cleanup, run_id=run_id, stage_log=stage_log)

    # auto-wire SFT base checkpoint + tokenizer from pretrain outputs
    if cfg.sft.finetune.base_checkpoint is None:
        ckpt = _resolve_ckpt(paths.checkpoint_dir("pretrain"))
        if ckpt is None:
            raise RuntimeError("Pretrain produced no checkpoint to fine-tune from")
        cfg.sft.finetune.base_checkpoint = str(ckpt)
        print(f"[run] SFT base checkpoint auto-wired -> {ckpt}")

    run_sft(cfg, paths, cleanup=cleanup, run_id=run_id, stage_log=stage_log)


# ── tokenizer compatibility check ────────────────────────────────────
def _verify_tokenizer_against_base(cfg: RunConfig):
    base = cfg.sft.finetune.base_checkpoint
    if not base or not Path(base).exists():
        return
    try:
        from ..training.checkpoint import read_meta

        meta = read_meta(base)
    except Exception:
        return
    cur = _tokenizer_meta(cfg.tokenizer.path)
    base_sha = meta.get("tokenizer_sha")
    if base_sha and cur.get("tokenizer_sha") and base_sha != cur["tokenizer_sha"]:
        print(
            "WARNING: SFT tokenizer differs from the one the base checkpoint was "
            "trained with — token ids may not align. Proceeding anyway."
        )


# ── cleanup ──────────────────────────────────────────────────────────
def _cleanup_pretrain(paths: RunPaths, seq_len: int):
    # raw text no longer needed once tokenized; tokenized shards no longer needed once packed
    safe_delete_stage("raw", paths.pretrain_data_dir / "raw",
                      paths.manifest("tokenize"), seq_len=seq_len)
    safe_delete_stage("tokenized", paths.pretrain_data_dir / "tokenized",
                      paths.manifest("pack"), seq_len=seq_len)
