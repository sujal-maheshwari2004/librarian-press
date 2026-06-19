"""
load_model.py — method-aware checkpoint loading for inference.

Reconstructs the right model shape from a checkpoint:
  - full / pretrain  : load the full state_dict
  - lora / bitfit    : build base, re-inject the adapter (rank/alpha from meta),
                       then copy the adapter weights in
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..config.schema import ModelConfig, LoRAConfig
from ..model.build import build_model, apply_finetune
from ..training.checkpoint import load_checkpoint


def load_model(model_cfg: ModelConfig, checkpoint_path: str | Path, device: str):
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    method = ckpt.get("method") if isinstance(ckpt, dict) else None
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

    model = build_model(model_cfg)

    if method in ("lora", "bitfit"):
        lora_cfg = LoRAConfig(
            rank=meta.get("lora_rank", 8),
            alpha=meta.get("lora_alpha", 16.0),
            dropout=0.0,
        )
        model = apply_finetune(model, method, lora_cfg)

    load_checkpoint(model, None, checkpoint_path, method=method)
    model.to(device)
    model.eval()
    return model
