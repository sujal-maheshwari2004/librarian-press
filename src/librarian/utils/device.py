"""Device + AMP resolution shared by every train/eval entrypoint."""

from __future__ import annotations

import torch


def resolve_device(requested: str) -> str:
    """Fall back to CPU when CUDA is requested but unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU")
        return "cpu"
    return requested


def adjust_amp(training_cfg, device: str):
    """
    Mixed precision in this codebase is hardcoded to the CUDA autocast/GradScaler
    path. On CPU we must disable it or the scaler silently no-ops in a half-broken
    way. Mutates and returns the training config.
    """
    if device != "cuda" and training_cfg.mixed_precision:
        print(f"WARNING: mixed_precision requires CUDA — disabling on {device}")
        training_cfg.mixed_precision = False
    return training_cfg
