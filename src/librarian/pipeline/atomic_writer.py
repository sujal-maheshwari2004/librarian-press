"""
atomic_writer.py — Atomic streaming file writer.

Writes data to a .tmp file, then os.replace() on clean close.
If the process dies mid-write, the destination file is either the previous
valid version or absent — never partially overwritten.
"""

from __future__ import annotations

import os
from pathlib import Path


class AtomicBinaryWriter:
    """Streams bytes to a .tmp file and atomically renames on clean exit."""

    def __init__(self, target: str | Path, buffer_size: int = 16 * 1024 * 1024):
        self.target      = Path(target)
        self.buffer_size = buffer_size
        self._tmp_path   = self.target.with_suffix(".tmp")
        self._fh         = None
        self._bytes_written = 0

    def __enter__(self) -> "AtomicBinaryWriter":
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._tmp_path, "wb", buffering=self.buffer_size)
        self._bytes_written = 0
        return self

    def write(self, data: bytes) -> int:
        n = self._fh.write(data)
        self._bytes_written += n
        return n

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fh.close()
        if exc_type is None:
            os.replace(self._tmp_path, self.target)
        return False   # don't suppress exceptions

    @property
    def bytes_written(self) -> int:
        return self._bytes_written


class AtomicTextWriter:
    """Same contract but for text files (e.g. ingested .txt / .jsonl shards)."""

    def __init__(self, target: str | Path, encoding: str = "utf-8"):
        self.target    = Path(target)
        self._tmp_path = self.target.with_suffix(".tmp")
        self._fh       = None
        self._encoding = encoding

    def __enter__(self) -> "AtomicTextWriter":
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._tmp_path, "w", encoding=self._encoding)
        return self

    def write(self, text: str):
        self._fh.write(text)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fh.close()
        if exc_type is None:
            os.replace(self._tmp_path, self.target)
        return False


def recover_stranded_tmps(
    root: str | Path,
    src_ext: str = ".tmp",
    dst_ext: str = ".txt",
    verbose: bool = True,
) -> int:
    """
    Rename all files matching src_ext under root to dst_ext.

    Called at the start of any stage that reads the output of a previous
    stage using an atomic writer, to recover from a mid-write crash where
    the atomic rename never completed. If a completed destination already
    exists alongside a .tmp, the .tmp is removed (stale partial write).

    Returns the number of files renamed.
    """
    root = Path(root)
    if not root.exists():
        return 0

    renamed = 0
    for tmp in sorted(root.rglob(f"*{src_ext}")):
        dst = tmp.with_suffix(dst_ext)
        if dst.exists():
            tmp.unlink()
            if verbose:
                print(f"[recover_tmps] Removed stale {tmp.name} (dst exists)")
        else:
            os.replace(tmp, dst)
            renamed += 1

    if renamed and verbose:
        print(f"[recover_tmps] Renamed {renamed} .tmp → {dst_ext} under {root}")

    return renamed
