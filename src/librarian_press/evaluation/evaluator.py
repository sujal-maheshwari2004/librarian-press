"""evaluator.py — route to the right metric for each mode."""

from __future__ import annotations

from .perplexity import compute_perplexity
from .generation import compute_generation_metrics


def evaluate_pretrain(model, dataloader, device) -> dict:
    return {"perplexity": round(compute_perplexity(model, dataloader, device), 4)}


def evaluate_sft(
    model,
    finetune_cfg,
    data_cfg,
    paths,
    val_loader,
    tokenizer,
    device,
    max_seq_len: int = 512,
) -> dict:
    metric = finetune_cfg.eval_metric
    if metric == "perplexity":
        return {"perplexity": round(compute_perplexity(model, val_loader, device), 4)}
    if metric in ("exact_match", "f1"):
        return compute_generation_metrics(
            model, data_cfg, paths, tokenizer, device, max_seq_len=max_seq_len
        )
    raise ValueError(f"Unknown eval_metric: {metric!r}")
