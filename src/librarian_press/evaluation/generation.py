"""generation.py — greedy decoding + generation metrics (exact match, token F1) for SFT."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from ..config.schema import SFTDataConfig
from ..config.paths import RunPaths
from ..data.ingest import iter_dict_records
from ..data.prepare_sft import render_prompt, resolve_field
from ..tokenizer.load import special_ids


@torch.no_grad()
def greedy_decode(model: nn.Module, prompt_ids, max_new_tokens, eos_id, device,
                  max_seq_len: int = 512):
    idx = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)
    generated = []
    for _ in range(max_new_tokens):
        logits = model(idx[:, -max_seq_len:])
        next_tok = logits[0, -1, :].argmax(-1).item()
        generated.append(next_tok)
        if next_tok == eos_id:
            break
        idx = torch.cat([idx, torch.tensor([[next_tok]], device=device)], dim=1)
    return generated


def token_f1(pred_tokens, gold_tokens) -> float:
    pred_set, gold_set = set(pred_tokens), set(gold_tokens)
    common = pred_set & gold_set
    if not common:
        return 0.0
    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def compute_generation_metrics(
    model: nn.Module,
    cfg: SFTDataConfig,
    paths: RunPaths,
    tokenizer,
    device: str,
    max_seq_len: int = 512,
    max_eval_examples: int = 500,
) -> dict:
    """Greedy-decode each validation prompt and compare to the gold completion."""
    raw_dir = paths.sft_data_dir / "raw" / cfg.split_val
    files = sorted(raw_dir.rglob("*.jsonl")) if raw_dir.exists() else []
    if not files:
        raise FileNotFoundError(f"No raw validation shards under {raw_dir}")

    ids = special_ids(tokenizer)
    bos_id, eos_id = ids["bos"], ids["eos"]
    model.eval()

    exact = 0
    f1_scores = []
    total = 0

    records = list(iter_dict_records(files, "txt"))[:max_eval_examples]
    for rec in tqdm(records, desc="evaluating"):
        prompt_text = render_prompt(rec, cfg.prompt_template)
        gold = resolve_field(rec, cfg.completion_field)
        if not gold:
            continue
        prompt_ids = [bos_id] + tokenizer.encode(prompt_text).ids[: cfg.max_prompt_len]
        pred_ids = greedy_decode(model, prompt_ids, cfg.max_completion_len, eos_id, device, max_seq_len)
        pred_text = tokenizer.decode(pred_ids).strip()
        gold_clean = gold.strip()
        if pred_text == gold_clean:
            exact += 1
        f1_scores.append(token_f1(pred_text.split(), gold_clean.split()))
        total += 1

    return {
        "exact_match": round(exact / max(total, 1), 4),
        "f1": round(sum(f1_scores) / max(len(f1_scores), 1), 4),
        "eval_examples": total,
    }
