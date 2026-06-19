"""
shard.py — turn user inputs into manifest-tracked raw shards.

Wraps the format-agnostic ingest readers in the resume infrastructure
(StageManifest + AtomicTextWriter). The downstream contract — a directory of
shard files plus a complete manifest — matches what tokenize/pack/prepare expect.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config.schema import PretrainDataConfig, SFTDataConfig
from ..config.paths import RunPaths
from ..pipeline.manifest import StageManifest, ShardState, file_checksum
from ..pipeline.atomic_writer import AtomicTextWriter
from .ingest import expand_inputs, iter_text_records, iter_dict_records


# ── Pretrain ingest ──────────────────────────────────────────────────
def ingest_pretrain(
    cfg: PretrainDataConfig,
    paths: RunPaths,
    manifest_path: Path,
    stage_log=None,
) -> dict:
    raw_dir = paths.pretrain_data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = StageManifest(manifest_path)
    manifest.reset_stale()

    files = expand_inputs(cfg.inputs)
    if not files:
        raise FileNotFoundError(
            f"No input files matched pretrain.data.inputs={cfg.inputs}"
        )
    print(f"[ingest_pre] {len(files)} input file(s)")

    state = {"shard_idx": 0, "buffer": [], "total": 0}

    def flush():
        sid = f"shard_{state['shard_idx']:06d}"
        path = raw_dir / f"{sid}.txt"
        existing = manifest._entries.get(sid)
        if existing and existing.state == ShardState.DONE:
            state["shard_idx"] += 1
            state["buffer"] = []
            return
        manifest.register_shards([sid])
        manifest.mark_processing(sid)
        with AtomicTextWriter(path) as w:
            for doc in state["buffer"]:
                # keep shards line-delimited (downstream reads line by line)
                w.write(doc.replace("\n", " ").replace("\r", " ") + "\n")
        manifest.mark_verified(sid, str(path), file_checksum(path), len(state["buffer"]))
        manifest.mark_done(sid)
        if stage_log:
            stage_log.progress("ingest_pre", {"shard": state["shard_idx"], "docs": len(state["buffer"])})
        state["shard_idx"] += 1
        state["buffer"] = []

    for text in iter_text_records(files, cfg.format, cfg.text_column, cfg.txt_granularity):
        state["buffer"].append(text)
        state["total"] += 1
        if len(state["buffer"]) >= cfg.docs_per_shard:
            flush()
    if state["buffer"]:
        flush()

    if state["total"] == 0:
        raise RuntimeError("[ingest_pre] No non-empty documents found in inputs")

    return {
        "documents": state["total"],
        "shards": state["shard_idx"],
        "manifest": manifest.summary(),
    }


# ── SFT ingest ───────────────────────────────────────────────────────
def _val_assignment(record: dict, val_frac: float) -> str:
    """Deterministic per-record train/val split for the 'all' inputs case."""
    key = json.dumps(record, sort_keys=True, default=str)
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10000
    return "val" if h < int(val_frac * 10000) else "train"


def ingest_sft(
    cfg: SFTDataConfig,
    paths: RunPaths,
    manifest_path: Path,
    stage_log=None,
) -> dict:
    raw_root = paths.sft_data_dir / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    manifest = StageManifest(manifest_path)
    manifest.reset_stale()

    # map logical split name -> list of records (as an iterator source)
    split_files: dict[str, list[Path]] = {}
    if "all" in cfg.inputs:
        all_files = expand_inputs(cfg.inputs["all"])
        if not all_files:
            raise FileNotFoundError(f"No files matched sft.data.inputs.all={cfg.inputs['all']}")
        return _ingest_sft_from_all(cfg, all_files, paths, manifest, stage_log)
    else:
        if "train" in cfg.inputs:
            split_files[cfg.split_train] = expand_inputs(cfg.inputs["train"])
        if "val" in cfg.inputs:
            split_files[cfg.split_val] = expand_inputs(cfg.inputs["val"])

    summary = {}
    for split, files in split_files.items():
        if not files:
            print(f"[ingest_sft] No files for split {split!r}, skipping")
            continue
        n = _write_split_shards(
            split, iter_dict_records(files, cfg.format), cfg, paths, manifest, stage_log
        )
        summary[f"{split}_records"] = n

    summary["manifest"] = manifest.summary()
    return summary


def _ingest_sft_from_all(cfg, files, paths, manifest, stage_log) -> dict:
    """Split a single pool of records into train/val deterministically."""
    train_split = cfg.split_train
    val_split = cfg.split_val
    counts = {train_split: 0, val_split: 0}
    writers = {
        train_split: _ShardWriter(train_split, paths, manifest, 50000, stage_log),
        val_split: _ShardWriter(val_split, paths, manifest, 50000, stage_log),
    }
    for rec in iter_dict_records(files, cfg.format):
        dest = train_split if _val_assignment(rec, cfg.val_frac) == "train" else val_split
        writers[dest].add(rec)
        counts[dest] += 1
    for w in writers.values():
        w.close()

    return {
        f"{train_split}_records": counts[train_split],
        f"{val_split}_records": counts[val_split],
        "manifest": manifest.summary(),
    }


def _write_split_shards(split, records, cfg, paths, manifest, stage_log) -> int:
    writer = _ShardWriter(split, paths, manifest, 50000, stage_log)
    n = 0
    for rec in records:
        writer.add(rec)
        n += 1
    writer.close()
    return n


class _ShardWriter:
    """Buffers JSON records and flushes manifest-tracked .jsonl shards per split."""

    def __init__(self, split, paths: RunPaths, manifest: StageManifest,
                 docs_per_shard: int, stage_log=None):
        self.split = split
        self.dir = paths.sft_data_dir / "raw" / split
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.docs_per_shard = docs_per_shard
        self.stage_log = stage_log
        self.idx = 0
        self.buffer: list[dict] = []

    def add(self, rec: dict):
        self.buffer.append(rec)
        if len(self.buffer) >= self.docs_per_shard:
            self._flush()

    def _flush(self):
        sid = f"{self.split}__shard_{self.idx:06d}"
        path = self.dir / f"shard_{self.idx:06d}.jsonl"
        existing = self.manifest._entries.get(sid)
        if existing and existing.state == ShardState.DONE:
            self.idx += 1
            self.buffer = []
            return
        self.manifest.register_shards([sid])
        self.manifest.mark_processing(sid)
        with AtomicTextWriter(path) as w:
            for rec in self.buffer:
                w.write(json.dumps(rec, default=str) + "\n")
        self.manifest.mark_verified(sid, str(path), file_checksum(path), len(self.buffer))
        self.manifest.mark_done(sid)
        if self.stage_log:
            self.stage_log.progress("ingest_sft", {"split": self.split, "shard": self.idx, "records": len(self.buffer)})
        self.idx += 1
        self.buffer = []

    def close(self):
        if self.buffer:
            self._flush()
