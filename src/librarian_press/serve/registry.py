"""
registry.py — local store of exported, ready-to-run model bundles.

A bundle is a self-contained folder (like an Ollama model) holding everything
inference needs:

    <models_dir>/<name>/
      bundle.json      metadata + model config + generation/chat defaults
      tokenizer.json   the tokenizer
      weights.pt       consolidated full weights ({"model": ..., "method": null})

Default location: ~/.librarian-press/models  (override with LIBRARIAN_PRESS_HOME).
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def models_dir() -> Path:
    home = os.environ.get("LIBRARIAN_PRESS_HOME")
    root = Path(home) if home else (Path.home() / ".librarian-press")
    return root / "models"


def bundle_dir(name: str) -> Path:
    return models_dir() / name


def list_models() -> list[str]:
    d = models_dir()
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if (p / "bundle.json").exists())


def load_bundle(name: str) -> dict:
    """Return the parsed bundle.json for a model, with absolute paths resolved."""
    bdir = bundle_dir(name)
    meta_path = bdir / "bundle.json"
    if not meta_path.exists():
        available = list_models()
        hint = f" Available: {', '.join(available)}" if available else " (no models exported yet)"
        raise FileNotFoundError(f"Model {name!r} not found in {models_dir()}.{hint}")
    with meta_path.open() as f:
        meta = json.load(f)
    meta["_dir"] = str(bdir)
    meta["_tokenizer_path"] = str(bdir / meta["tokenizer"])
    meta["_weights_path"] = str(bdir / meta["weights"])
    return meta
