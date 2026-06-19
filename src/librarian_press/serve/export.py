"""
export.py — consolidate a trained run into a portable, self-contained bundle.

Resolves whatever the run produced (full / LoRA / BitFit) into a single plain
set of weights, copies the tokenizer, and writes bundle.json. The result needs
nothing from the original run dir or base checkpoint to chat with.

LoRA is merged into the base weights:  W' = W + (alpha/rank) * (B @ A).
Because a LoRA-adapted lm_head can diverge from the tied embedding, merged LoRA
bundles are exported with tie_embeddings=False so both matrices are stored.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn as nn

from ..config.schema import RunConfig, ModelConfig
from ..config.paths import RunPaths
from ..model.build import build_model, load_for_sft
from ..model.gpt import GPT
from ..model.lora import LoRALinear
from ..model.rmsnorm import RMSNorm
from ..training.checkpoint import load_checkpoint

_FIELD_RE = re.compile(r"\{(\w+(?:\.\w+)*(?:\[\d+\])?)\}")


def _resolve_ckpt(ckpt_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    for cand in ("best.pt", "last.pt"):
        p = ckpt_dir / cand
        if p.exists():
            return p
    raise FileNotFoundError(f"No best.pt/last.pt under {ckpt_dir}")


def _load_trained(cfg: RunConfig, paths: RunPaths, source: str, checkpoint: str | None):
    """Return (model on cpu in eval mode, method)."""
    if source == "sft":
        method = cfg.sft.finetune.method
        ckpt = _resolve_ckpt(paths.checkpoint_dir(f"sft/{method}"), checkpoint)
        model = load_for_sft(cfg.model, cfg.sft.finetune, "cpu")
        load_checkpoint(model, None, ckpt, method=method)
    else:
        method = None
        ckpt = _resolve_ckpt(paths.checkpoint_dir("pretrain"), checkpoint)
        model = build_model(cfg.model)
        load_checkpoint(model, None, ckpt)
    model.eval()
    return model, method


def _consolidate(trained: nn.Module, model_cfg: ModelConfig, method: str | None):
    """Return (full_state_dict, export_model_cfg) for a plain GPT."""
    # full / bitfit keep a plain GPT structure already
    if method in (None, "full", "bitfit"):
        return trained.state_dict(), model_cfg

    # LoRA: merge adapters into an untied plain GPT
    export_cfg = replace(model_cfg, tie_embeddings=False)
    plain = build_model(export_cfg)
    tmods = dict(trained.named_modules())

    for name, pmod in plain.named_modules():
        tmod = tmods.get(name)
        if isinstance(pmod, nn.Linear):
            if isinstance(tmod, LoRALinear):
                W = tmod.linear.weight.data
                delta = tmod.scaling * (tmod.lora_B.data @ tmod.lora_A.data)
                pmod.weight.data.copy_(W + delta)
                if pmod.bias is not None and tmod.linear.bias is not None:
                    pmod.bias.data.copy_(tmod.linear.bias.data)
            elif isinstance(tmod, nn.Linear):
                pmod.weight.data.copy_(tmod.weight.data)
                if pmod.bias is not None and tmod.bias is not None:
                    pmod.bias.data.copy_(tmod.bias.data)
        elif isinstance(pmod, nn.Embedding) and isinstance(tmod, nn.Embedding):
            pmod.weight.data.copy_(tmod.weight.data)
        elif isinstance(pmod, RMSNorm) and isinstance(tmod, RMSNorm):
            pmod.weight.data.copy_(tmod.weight.data)

    return plain.state_dict(), export_cfg


def export_model(
    cfg: RunConfig,
    *,
    name: str,
    checkpoint: str | None = None,
    source: str | None = None,
    temperature: float = 0.8,
    top_k: int = 40,
    max_new_tokens: int = 256,
) -> Path:
    from ..serve.registry import bundle_dir

    paths = RunPaths(cfg.name)

    # decide which half of the run to export
    if source is None:
        source = "sft" if cfg.sft is not None else "pretrain"
    if source == "sft" and cfg.sft is None:
        raise ValueError("config has no sft section to export")
    if source == "pretrain" and cfg.pretrain is None:
        raise ValueError("config has no pretrain section to export")

    # A LoRA/BitFit SFT model is only reconstructable on top of the exact base it
    # was trained with. In mode "both" the base is auto-wired at train time, so the
    # config's base_checkpoint is null — resolve it the same way run_all does.
    if source == "sft" and cfg.sft.finetune.base_checkpoint is None:
        base = _resolve_ckpt(paths.checkpoint_dir("pretrain"), None)
        cfg.sft.finetune.base_checkpoint = str(base)
        print(f"[export] base checkpoint auto-wired -> {base}")

    trained, method = _load_trained(cfg, paths, source, checkpoint)
    full_sd, export_cfg = _consolidate(trained, cfg.model, method)

    # chat prompt template (SFT only): reuse the run's template, routing user
    # input into its first field.
    prompt_template = None
    prompt_field = None
    if source == "sft":
        prompt_template = cfg.sft.data.prompt_template
        fields = _FIELD_RE.findall(prompt_template or "")
        prompt_field = fields[0] if fields else None

    bdir = bundle_dir(name)
    bdir.mkdir(parents=True, exist_ok=True)

    torch.save({"model": full_sd, "method": None, "meta": {"name": name}},
               bdir / "weights.pt")
    shutil.copyfile(cfg.tokenizer.path, bdir / "tokenizer.json")

    bundle = {
        "name": name,
        "created": int(time.time()),
        "source": source,
        "trained_method": method or "full/pretrain",
        "model": asdict(export_cfg),
        "tokenizer": "tokenizer.json",
        "weights": "weights.pt",
        "max_seq_len": export_cfg.max_seq_len,
        "prompt_template": prompt_template,
        "prompt_field": prompt_field,
        "generation": {
            "temperature": temperature,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
        },
    }
    with open(bdir / "bundle.json", "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"\nExported model '{name}' -> {bdir}")
    print(f"  source={source}  method={method or 'full/pretrain'}  "
          f"vocab={export_cfg.vocab_size}  seq_len={export_cfg.max_seq_len}")
    print(f"  chat with:  librarian-press chat {name}")
    return bdir
