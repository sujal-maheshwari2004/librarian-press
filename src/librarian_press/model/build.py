"""
build.py — construct a GPT, apply a finetune method, and load base weights.

Absorbs the SFT repo's loader.py (which previously reached into the base repo
via a sys.path hack). No path manipulation here — everything is in-package.
"""

from __future__ import annotations

import torch.nn as nn

from ..config.schema import ModelConfig, FinetuneConfig, LoRAConfig
from ..training.checkpoint import load_base_weights
from .gpt import GPT
from .lora import inject_lora, enable_bitfit, print_trainable_parameters


def build_model(model_cfg: ModelConfig) -> GPT:
    return GPT(model_cfg)


def apply_finetune(
    model: nn.Module,
    method: str,
    lora_cfg: LoRAConfig | None = None,
) -> nn.Module:
    if method == "lora":
        lora_cfg = lora_cfg or LoRAConfig()
        model = inject_lora(
            model,
            rank=lora_cfg.rank,
            alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout,
        )
        print(f"LoRA injected (rank={lora_cfg.rank}, alpha={lora_cfg.alpha})")
    elif method == "bitfit":
        model = enable_bitfit(model)
        print("BitFit enabled (bias params only)")
    elif method == "full":
        for param in model.parameters():
            param.requires_grad = True
        print("Full finetune (all parameters trainable)")
    else:
        raise ValueError(f"Unknown method: {method!r}. Use lora | bitfit | full.")

    print_trainable_parameters(model)
    return model


def load_for_sft(
    model_cfg: ModelConfig,
    finetune_cfg: FinetuneConfig,
    device: str,
) -> nn.Module:
    """
    Build a GPT, load pretrained base weights (if given), then apply the
    finetune method. Returns the model on `device`, ready to train.
    """
    model = build_model(model_cfg)

    if finetune_cfg.base_checkpoint:
        load_base_weights(model, finetune_cfg.base_checkpoint)
        print(f"Loaded base weights from: {finetune_cfg.base_checkpoint}")

    model = apply_finetune(model, finetune_cfg.method, finetune_cfg.lora)
    return model.to(device)
