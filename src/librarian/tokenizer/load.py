"""load.py — load a saved tokenizer and resolve its special-token ids."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    return Tokenizer.from_file(str(path))


def special_ids(tokenizer: Tokenizer) -> dict[str, int]:
    """Return {bos, eos, pad, unk} ids with sane fallbacks."""
    return {
        "bos": tokenizer.token_to_id("<bos>") if tokenizer.token_to_id("<bos>") is not None else 1,
        "eos": tokenizer.token_to_id("<eos>") if tokenizer.token_to_id("<eos>") is not None else 2,
        "pad": tokenizer.token_to_id("<pad>") if tokenizer.token_to_id("<pad>") is not None else 0,
        "unk": tokenizer.token_to_id("<unk>") if tokenizer.token_to_id("<unk>") is not None else 3,
    }
