"""
tokenize.py — streaming per-shard tokenization with manifest tracking.

Reads the ingested raw text shards (tracked by the ingest_pre manifest),
tokenizes each into a uint16 .bin shard written atomically, and records
progress in the tokenize manifest. Paths come from RunPaths — nothing hardcoded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config.paths import RunPaths
from ..pipeline.manifest import StageManifest, ShardState, file_checksum
from ..pipeline.atomic_writer import AtomicBinaryWriter
from ..tokenizer.load import load_tokenizer

DTYPE = np.uint16


def _tokenize_shard(raw_path: Path, out_path: Path, tokenizer) -> int:
    FLUSH_EVERY = 100_000
    buffer: list[int] = []
    total = 0
    with AtomicBinaryWriter(out_path) as w:
        with raw_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                buffer.extend(tokenizer.encode(line).ids)
                if len(buffer) >= FLUSH_EVERY:
                    w.write(np.array(buffer, dtype=DTYPE).tobytes())
                    total += len(buffer)
                    buffer = []
        if buffer:
            w.write(np.array(buffer, dtype=DTYPE).tobytes())
            total += len(buffer)
    return total


def run_tokenize(paths: RunPaths, tokenizer_path: str | Path, stage_log=None) -> dict:
    ingest_manifest = StageManifest(paths.manifest("ingest_pre"))
    if not ingest_manifest.is_complete():
        raise RuntimeError(f"ingest_pre not complete: {ingest_manifest.summary()}")

    tokenizer = load_tokenizer(tokenizer_path)

    out_dir = paths.pretrain_data_dir / "tokenized"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_manifest = StageManifest(paths.manifest("tokenize"))
    tok_manifest.reset_stale()

    entries = ingest_manifest.verified_entries()
    tok_manifest.register_shards([e.shard_id for e in entries], meta={"stage": "tokenize"})

    print(f"\n=== TOKENIZE STAGE ===  {len(entries)} shard(s)")
    total_tokens = 0

    for entry in entries:
        sid = entry.shard_id
        existing = tok_manifest._entries.get(sid)
        if existing and existing.state == ShardState.DONE:
            total_tokens += existing.token_count
            continue

        raw_path = Path(entry.output_path)
        out_path = out_dir / f"{sid}.bin"
        tok_manifest.mark_processing(sid)
        try:
            n = _tokenize_shard(raw_path, out_path, tokenizer)
        except Exception as e:
            tok_manifest.mark_failed(sid, str(e))
            print(f"[tokenize] FAILED {sid}: {e}")
            raise

        tok_manifest.mark_verified(sid, str(out_path), file_checksum(out_path), n)
        tok_manifest.mark_done(sid)
        total_tokens += n
        if stage_log:
            stage_log.progress("tokenize", {"shard": sid, "tokens": n})
        print(f"[tokenize] {sid}: {n:,} tokens")

    print(f"[tokenize] Total tokens: {total_tokens:,}")
    return {"total_tokens": total_tokens, "manifest": tok_manifest.summary()}
