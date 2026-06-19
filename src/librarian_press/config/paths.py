"""
RunPaths — all on-disk locations for a run, namespaced by config `name`.

Layout
------
runs/<name>/
  data/
    pretrain/raw/shard_XXXXXX.txt
    pretrain/tokenized/shard_XXXXXX.bin
    pretrain/{train,validation,test}_packed.bin
    sft/raw/<split>/shard_XXXXXX.jsonl
    sft/{train,validation}.bin  +  {train,validation}_mask.bin
  manifests/<stage>.json
  tokenizer/tokenizer.json          (when trained by the framework)
  checkpoints/pretrain/{best,last,step_*}.pt
  checkpoints/sft/<method>/{best,last,step_*}.pt
  logs/
"""

from __future__ import annotations

from pathlib import Path


class RunPaths:
    def __init__(self, name: str):
        self.name = name
        self.run_dir = Path("runs") / name

    # ── data ─────────────────────────────────────────────────────────
    @property
    def data_dir(self) -> Path:
        return self.run_dir / "data"

    @property
    def pretrain_data_dir(self) -> Path:
        return self.data_dir / "pretrain"

    @property
    def sft_data_dir(self) -> Path:
        return self.data_dir / "sft"

    # ── manifests ────────────────────────────────────────────────────
    @property
    def manifest_dir(self) -> Path:
        return self.run_dir / "manifests"

    def manifest(self, stage: str) -> Path:
        return self.manifest_dir / f"{stage}.json"

    # ── tokenizer ────────────────────────────────────────────────────
    @property
    def tokenizer_dir(self) -> Path:
        return self.run_dir / "tokenizer"

    # ── checkpoints ──────────────────────────────────────────────────
    def checkpoint_dir(self, namespace: str) -> Path:
        return self.run_dir / "checkpoints" / namespace

    # ── logs ─────────────────────────────────────────────────────────
    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"

    # ── setup ────────────────────────────────────────────────────────
    def make_dirs(self):
        for d in (
            self.data_dir,
            self.manifest_dir,
            self.tokenizer_dir,
            self.log_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
