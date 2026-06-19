"""
ingest.py — format-agnostic readers for user-supplied clean data.

Replaces the base repo's HuggingFace download + quality-cleaning. The user owns
data cleanliness; we only parse local Parquet and txt files.

  iter_text_records  -> pretraining: a stream of text strings
  iter_dict_records  -> SFT: a stream of record dicts (nested structs preserved)
"""

from __future__ import annotations

import glob as _glob
import json
from pathlib import Path
from typing import Iterator


def expand_inputs(inputs: list[str]) -> list[Path]:
    """Expand a list of globs / paths into a sorted, de-duplicated file list."""
    files: list[str] = []
    for pattern in inputs:
        matched = _glob.glob(pattern, recursive=True)
        if matched:
            files.extend(matched)
        elif Path(pattern).is_file():
            files.append(pattern)
    return [Path(f) for f in sorted(set(files)) if Path(f).is_file()]


def _detect_format(path: Path, fmt: str) -> str:
    if fmt and fmt != "auto":
        return fmt
    ext = path.suffix.lower()
    if ext == ".parquet":
        return "parquet"
    if ext in (".txt", ".jsonl", ".json"):
        return "txt"
    raise ValueError(
        f"Cannot auto-detect format for {path} (ext {ext!r}); "
        "set data.format to 'txt' or 'parquet'."
    )


# ── pretraining: text records ────────────────────────────────────────
def iter_text_records(
    files: list[Path],
    fmt: str = "auto",
    text_column: str = "text",
    granularity: str = "line",
) -> Iterator[str]:
    for path in files:
        f = _detect_format(path, fmt)
        if f == "parquet":
            yield from _iter_parquet_text(path, text_column)
        else:
            yield from _iter_txt_text(path, granularity)


def _iter_txt_text(path: Path, granularity: str) -> Iterator[str]:
    if granularity == "document":
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            yield text
        return
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def _iter_parquet_text(path: Path, text_column: str) -> Iterator[str]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    columns = pf.schema_arrow.names
    if text_column not in columns:
        raise ValueError(
            f"text_column {text_column!r} not found in {path}. "
            f"Available columns: {columns}"
        )
    for batch in pf.iter_batches(columns=[text_column]):
        for val in batch.column(text_column):
            v = val.as_py()
            if v is None:
                continue
            s = str(v).strip()
            if s:
                yield s


# ── SFT: dict records ────────────────────────────────────────────────
def iter_dict_records(files: list[Path], fmt: str = "auto") -> Iterator[dict]:
    for path in files:
        f = _detect_format(path, fmt)
        if f == "parquet":
            yield from _iter_parquet_dict(path)
        else:
            yield from _iter_jsonl_dict(path)


def _iter_parquet_dict(path: Path) -> Iterator[dict]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches():
        # to_pylist preserves nested struct/list values so dotted field
        # resolution (e.g. "answers.text[0]") keeps working.
        for row in batch.to_pylist():
            yield row


def _iter_jsonl_dict(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{i + 1} is not valid JSON. SFT .txt inputs must be "
                    f"JSON-per-line (one record per line). Original error: {e}"
                ) from e
