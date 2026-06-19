"""
checkpoint.py — unified, method-aware checkpointing.

A checkpoint is self-describing via its `method`:
  None / "full"      -> {"model": full state_dict, ...}
  "lora" / "bitfit"  -> {"adapter": trainable-only params, ...}

`meta` carries provenance (e.g. tokenizer vocab/sha) so SFT can verify it is
fine-tuning a model trained with the matching tokenizer. A pretrain checkpoint
({"model": ...}) loads cleanly as an SFT base via load_base_weights.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def _adapter_state(model: nn.Module) -> dict:
    return {
        k: v.detach().cpu()
        for k, v in model.named_parameters()
        if v.requires_grad
    }


def save_checkpoint(
    model: nn.Module,
    optimizer,
    step: int,
    path: str | Path,
    *,
    method: str | None = None,
    val_loss: float | None = None,
    meta: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "step": step,
        "method": method,
        "val_loss": val_loss,
        "meta": meta or {},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    if method in ("lora", "bitfit"):
        payload["adapter"] = _adapter_state(model)
    else:
        payload["model"] = model.state_dict()

    torch.save(payload, path)
    return path


def save_best(model, optimizer, step, ckpt_dir, *, method=None, val_loss=None, meta=None) -> Path:
    return save_checkpoint(
        model, optimizer, step, Path(ckpt_dir) / "best.pt",
        method=method, val_loss=val_loss, meta=meta,
    )


def save_last(model, optimizer, step, ckpt_dir, *, method=None, val_loss=None, meta=None) -> Path:
    return save_checkpoint(
        model, optimizer, step, Path(ckpt_dir) / "last.pt",
        method=method, val_loss=val_loss, meta=meta,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer,
    path: str | Path,
    *,
    method: str | None = None,
) -> int:
    """
    Restore model (and optionally optimizer) from a checkpoint. Returns the
    saved step. `method` defaults to whatever the checkpoint recorded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(str(path), map_location="cpu")
    method = method if method is not None else ckpt.get("method")

    if method in ("lora", "bitfit"):
        adapter_state = ckpt["adapter"]
        current = dict(model.named_parameters())
        for k, v in adapter_state.items():
            if k in current:
                current[k].data.copy_(v)
            else:
                print(f"  WARNING: adapter key {k!r} not found in model, skipping.")
    else:
        model.load_state_dict(ckpt["model"])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    step = ckpt.get("step", 0)
    print(f"Loaded checkpoint from step {step}  ({path.name})")
    return step


def load_base_weights(model: nn.Module, path: str | Path):
    """
    Load full base weights into a freshly built model. Accepts both the
    framework's {"model": state_dict, ...} format and a bare state_dict.
    Used to seed SFT from a pretrained checkpoint.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {path}")

    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "adapter" in ckpt:
        raise ValueError(
            f"{path} is an adapter checkpoint, not a full base model. "
            "Point base_checkpoint at a pretrain/full checkpoint."
        )
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)


def read_meta(path: str | Path) -> dict:
    """Return the meta dict stored in a checkpoint (empty if none)."""
    ckpt = torch.load(str(path), map_location="cpu")
    return ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
