"""
prepare_sft.py — tokenize SFT records into masked binary splits.

Reads the ingested raw JSONL shards (tracked by the ingest_sft manifest),
renders prompt_template + completion_field per record, and writes:

    runs/<name>/data/sft/<split>.bin        flat uint16 token ids
    runs/<name>/data/sft/<split>_mask.bin   uint16: 0 = prompt, 1 = completion

Record layout: [bos] prompt... completion... [eos]; loss flows only through
completion + eos (mask 1). render_prompt/resolve_field support dotted field
expressions with a trailing [i] index, e.g. "answers.text[0]".
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..config.schema import SFTDataConfig
from ..config.paths import RunPaths
from ..pipeline.manifest import StageManifest, ShardState, file_checksum
from ..tokenizer.load import load_tokenizer, special_ids
from .ingest import iter_dict_records

DTYPE = np.uint16


# ── field resolution + templating (reused from the SFT repo) ─────────
def resolve_field(example: dict, field_expr: str) -> str:
    index = None
    m = re.match(r"^(.*)\[(\d+)\]$", field_expr)
    if m:
        field_expr = m.group(1)
        index = int(m.group(2))

    value = example
    for part in field_expr.split("."):
        if isinstance(value, dict):
            value = value.get(part, "")
        else:
            return ""

    if index is not None and isinstance(value, (list, tuple)):
        value = value[index] if len(value) > index else ""

    return str(value).strip() if value else ""


def render_prompt(example: dict, template: str) -> str:
    def replacer(m):
        return resolve_field(example, m.group(1)) or ""

    return re.sub(r"\{(\w+(?:\.\w+)*(?:\[\d+\])?)\}", replacer, template)


def tokenize_example(example, ds_cfg: SFTDataConfig, tokenizer, bos_id, eos_id):
    prompt_text = render_prompt(example, ds_cfg.prompt_template)
    completion_text = resolve_field(example, ds_cfg.completion_field)
    if not completion_text:
        return None

    prompt_ids = tokenizer.encode(prompt_text).ids[: ds_cfg.max_prompt_len]
    completion_ids = tokenizer.encode(completion_text).ids[: ds_cfg.max_completion_len]

    full = [bos_id] + prompt_ids + completion_ids + [eos_id]
    prompt_len = 1 + len(prompt_ids)            # bos + prompt
    return full, prompt_len


# ── per-split processing ─────────────────────────────────────────────
def _process_split(split: str, cfg: SFTDataConfig, paths: RunPaths,
                   tokenizer, bos_id, eos_id, stage_log=None) -> dict:
    raw_dir = paths.sft_data_dir / "raw" / split
    if not raw_dir.exists():
        print(f"[prepare_sft] no raw dir for {split!r}, skipping")
        return {"written": 0, "skipped": 0}

    files = sorted(raw_dir.rglob("*.jsonl"))
    tok_path = paths.sft_data_dir / f"{split}.bin"
    mask_path = paths.sft_data_dir / f"{split}_mask.bin"

    all_tokens: list[int] = []
    all_masks: list[int] = []
    written = skipped = 0

    for rec in iter_dict_records(files, "txt"):
        result = tokenize_example(rec, cfg, tokenizer, bos_id, eos_id)
        if result is None:
            skipped += 1
            continue
        full, prompt_len = result
        all_tokens.extend(full)
        all_masks.extend([0] * prompt_len + [1] * (len(full) - prompt_len))
        written += 1

    total = written + skipped
    if total and (skipped / total) > cfg.max_skipped_frac:
        raise RuntimeError(
            f"[prepare_sft] {split}: skipped {skipped}/{total} records "
            f"({skipped / total:.0%}) — likely a wrong completion_field "
            f"({cfg.completion_field!r}). Aborting."
        )

    paths.sft_data_dir.mkdir(parents=True, exist_ok=True)
    np.array(all_tokens, dtype=DTYPE).tofile(tok_path)
    np.array(all_masks, dtype=DTYPE).tofile(mask_path)

    print(f"[prepare_sft] {split}: {written:,} examples | {len(all_tokens):,} tokens "
          f"| skipped {skipped:,}")
    if stage_log:
        stage_log.progress("prepare_sft", {"split": split, "written": written, "skipped": skipped})

    return {"written": written, "skipped": skipped, "tok_path": str(tok_path),
            "tokens": len(all_tokens)}


def run_prepare_sft(cfg: SFTDataConfig, paths: RunPaths, tokenizer_path, stage_log=None) -> dict:
    ingest_manifest = StageManifest(paths.manifest("ingest_sft"))
    if not ingest_manifest.is_complete():
        raise RuntimeError(f"ingest_sft not complete: {ingest_manifest.summary()}")

    tokenizer = load_tokenizer(tokenizer_path)
    sid_specials = special_ids(tokenizer)
    bos_id, eos_id = sid_specials["bos"], sid_specials["eos"]

    prep_manifest = StageManifest(paths.manifest("prepare_sft"))
    prep_manifest.reset_stale()

    summary = {}
    for split in (cfg.split_train, cfg.split_val):
        sid = f"prepare__{split}"
        existing = prep_manifest._entries.get(sid)
        if existing and existing.state == ShardState.DONE:
            continue
        prep_manifest.register_shards([sid], meta={"stage": "prepare_sft"})
        prep_manifest.mark_processing(sid)
        res = _process_split(split, cfg, paths, tokenizer, bos_id, eos_id, stage_log)
        out = res.get("tok_path", "")
        checksum = file_checksum(out) if out and Path(out).exists() else ""
        prep_manifest.mark_verified(sid, out, checksum, res.get("tokens", 0))
        prep_manifest.mark_done(sid)
        summary[split] = res

    return summary
