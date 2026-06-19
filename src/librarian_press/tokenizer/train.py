"""
train.py — byte-level BPE tokenizer training.

Reads the ingested pretrain text shards (a directory of line-delimited .txt
files) and trains a HuggingFace BPE tokenizer with NFKC normalization, a
ByteLevel pre-tokenizer, and <bos> $A <eos> post-processing. Vocab size,
min frequency, and special tokens come from TokenizerConfig.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from ..config.schema import TokenizerConfig
from ..pipeline.atomic_writer import recover_stranded_tmps


def _iter_documents(shards_dir: Path):
    for shard_file in sorted(shards_dir.rglob("*.txt")):
        with shard_file.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.strip():
                    yield line


def _write_temp_corpus(tmp_path: Path, shards_dir: Path, stage_log=None) -> int:
    if not shards_dir.exists():
        raise FileNotFoundError(
            f"Ingested text shards not found: {shards_dir}\n"
            "Run the ingest_pre stage before training the tokenizer."
        )

    recovered = recover_stranded_tmps(shards_dir, src_ext=".tmp", dst_ext=".txt")
    if recovered:
        print(f"[train_tokenizer] Recovered {recovered} stranded .tmp shards.")

    shard_files = sorted(shards_dir.rglob("*.txt"))
    if not shard_files:
        raise FileNotFoundError(f"No .txt shards under {shards_dir}")

    print(f"[train_tokenizer] Streaming {len(shard_files)} shard(s) into temp corpus…")
    lines = 0
    with tmp_path.open("w", encoding="utf-8") as out:
        for doc in _iter_documents(shards_dir):
            out.write(doc + "\n")
            lines += 1
            if lines % 500_000 == 0 and stage_log:
                stage_log.progress("train_tokenizer", {"lines_streamed": lines})
    print(f"[train_tokenizer] Corpus ready: {lines:,} lines")
    return lines


def train_tokenizer(
    cfg: TokenizerConfig,
    shards_dir: Path,
    stage_log=None,
) -> dict:
    out_path = Path(cfg.path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== Training Tokenizer ===")
    if stage_log:
        stage_log.progress("train_tokenizer", {
            "vocab_size": cfg.vocab_size,
            "min_frequency": cfg.min_frequency,
        })

    with tempfile.TemporaryDirectory(prefix="librarian_tok_") as tmp_dir:
        corpus_path = Path(tmp_dir) / "corpus.txt"
        lines = _write_temp_corpus(corpus_path, shards_dir, stage_log)

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

        trainer = BpeTrainer(
            vocab_size=cfg.vocab_size,
            min_frequency=cfg.min_frequency,
            special_tokens=list(cfg.special_tokens),
        )

        print("Training tokenizer…")
        t0 = time.time()
        tokenizer.train([str(corpus_path)], trainer)
        train_elapsed = time.time() - t0

    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    tokenizer.decoder = ByteLevelDecoder()

    tokenizer.save(str(out_path))

    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "model": "BPE",
        "pre_tokenizer": "ByteLevel",
        "normalizer": "NFKC",
        "min_frequency": cfg.min_frequency,
        "special_tokens": list(cfg.special_tokens),
    }
    with open(out_path.parent / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=4)

    final_vocab = tokenizer.get_vocab_size()
    print(f"\nTokenizer training complete. vocab={final_vocab}  lines={lines:,}  "
          f"elapsed={train_elapsed:.1f}s")

    return {
        "final_vocab_size": final_vocab,
        "lines_trained_on": lines,
        "train_elapsed_s": round(train_elapsed, 2),
    }
