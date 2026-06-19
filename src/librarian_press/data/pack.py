"""
pack.py — streaming token packer with manifest tracking.

Streams tokenized shards (never loading them whole), packs exact `seq_len`
sequences, and carries partial sequences across shard boundaries so no tokens
are lost. Shards are assigned to train/validation/test by a deterministic hash
of their shard_id (reproducible across restarts). val_frac/test_frac and seq_len
come from the config.

Output (under runs/<name>/data/pretrain/):
  train_packed.bin, validation_packed.bin, test_packed.bin
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..config.paths import RunPaths
from ..pipeline.manifest import StageManifest, file_checksum, validate_bin_file
from ..pipeline.atomic_writer import AtomicBinaryWriter

DTYPE = np.uint16
DTYPE_BYTES = np.dtype(DTYPE).itemsize


def _iter_tokens(path: Path):
    CHUNK = 32 * 1024 * 1024 // DTYPE_BYTES
    with open(path, "rb") as f:
        while True:
            raw = f.read(CHUNK * DTYPE_BYTES)
            if not raw:
                break
            yield np.frombuffer(raw, dtype=DTYPE)


def _split_for(shard_id: str, val_frac: float, test_frac: float) -> str:
    h = int(hashlib.sha256(shard_id.encode()).hexdigest(), 16) % 10000
    if h < int(test_frac * 10000):
        return "test"
    if h < int((val_frac + test_frac) * 10000):
        return "validation"
    return "train"


class _Packer:
    def __init__(self, out_path: Path, seq_len: int):
        self.out_path = out_path
        self.seq_len = seq_len
        self._carry = np.array([], dtype=DTYPE)
        self._writer = None
        self._n_seqs = 0

    def __enter__(self):
        self._writer = AtomicBinaryWriter(self.out_path).__enter__()
        return self

    def feed(self, tokens: np.ndarray):
        if len(tokens) == 0:
            return
        combined = np.concatenate([self._carry, tokens])
        n_complete = len(combined) // self.seq_len
        if n_complete > 0:
            usable = combined[: n_complete * self.seq_len]
            self._writer.write(usable.tobytes())
            self._n_seqs += n_complete
        self._carry = combined[n_complete * self.seq_len:]

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._writer.__exit__(exc_type, exc_val, exc_tb)

    @property
    def sequences_written(self) -> int:
        return self._n_seqs


def run_pack(
    paths: RunPaths,
    seq_len: int,
    val_frac: float = 0.005,
    test_frac: float = 0.0,
    stage_log=None,
) -> dict:
    tok_manifest = StageManifest(paths.manifest("tokenize"))
    if not tok_manifest.is_complete():
        raise RuntimeError(f"tokenize not complete: {tok_manifest.summary()}")

    out_dir = paths.pretrain_data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = tok_manifest.verified_entries()
    entries.sort(key=lambda e: e.shard_id)

    splits: dict[str, list] = {"train": [], "validation": [], "test": []}
    for entry in entries:
        splits[_split_for(entry.shard_id, val_frac, test_frac)].append(entry)

    print(f"\n=== PACK STAGE ===  seq_len={seq_len}")
    for split, es in splits.items():
        print(f"  {split}: {len(es)} shards")

    results: dict = {}
    pack_manifest = StageManifest(paths.manifest("pack"))
    pack_manifest.reset_stale()

    for split, es in splits.items():
        if not es:
            continue
        out_path = out_dir / f"{split}_packed.bin"

        if out_path.exists():
            valid, n_tok, _ = validate_bin_file(out_path, seq_len)
            if valid:
                print(f"[pack] {out_path.name} already valid ({n_tok:,} tokens) — skipping")
                results[split] = {"sequences": n_tok // seq_len, "tokens": n_tok}
                continue

        print(f"[pack] Packing {split}…")
        with _Packer(out_path, seq_len) as packer:
            for entry in es:
                for chunk in _iter_tokens(Path(entry.output_path)):
                    packer.feed(chunk)
                if stage_log:
                    stage_log.progress("pack", {"split": split, "shard": entry.shard_id,
                                                "seqs": packer.sequences_written})

        valid, n_tok, err = validate_bin_file(out_path, seq_len)
        if not valid:
            raise RuntimeError(f"Packed output invalid ({split}): {err}")
        results[split] = {"sequences": packer.sequences_written, "tokens": n_tok}
        print(f"[pack] {split}: {packer.sequences_written:,} sequences")

    # record a pack manifest so the stage is resumable / skippable
    sids = [f"packed_{s}" for s in results]
    pack_manifest.register_shards(sids, meta={"stage": "pack"})
    for split in results:
        sid = f"packed_{split}"
        out_path = out_dir / f"{split}_packed.bin"
        pack_manifest.mark_processing(sid)
        pack_manifest.mark_verified(sid, str(out_path), file_checksum(out_path),
                                    results[split]["tokens"])
        pack_manifest.mark_done(sid)

    return results
