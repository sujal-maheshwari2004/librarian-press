"""
Unified config schema for the librarian framework.

One JSON config drives everything. Top-level `mode` selects pretrain / sft / both.
Shared sections: `model`, `tokenizer`. Each mode carries its own `data` + `training`
(and `finetune` for sft). Dataclasses below mirror the JSON one-to-one.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Model ────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # tokenizer
    vocab_size: int = 16000
    # architecture
    dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    hidden_dim: int = 2048
    # sequence
    max_seq_len: int = 512
    # dropout
    dropout: float = 0.1
    # rope
    rope_theta: float = 10000.0
    # weight tying
    tie_embeddings: bool = True


# ── Tokenizer ────────────────────────────────────────────────────────
@dataclass
class TokenizerConfig:
    path: str
    train_if_missing: bool = False
    vocab_size: int = 32000
    min_frequency: int = 2
    special_tokens: list[str] = field(
        default_factory=lambda: ["<pad>", "<bos>", "<eos>", "<unk>"]
    )


# ── Training (shared shape, per-mode instance) ───────────────────────
@dataclass
class TrainingConfig:
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    total_steps: int = 50000
    batch_size: int = 32
    grad_accum: int = 4
    weight_decay: float = 0.1
    mixed_precision: bool = True
    eval_interval: int = 1000
    save_interval: int = 5000
    device: str = "cuda"


# ── Pretrain data ────────────────────────────────────────────────────
@dataclass
class PretrainDataConfig:
    inputs: list[str]                       # globs / file paths to clean text
    format: str = "auto"                    # "txt" | "parquet" | "auto"
    text_column: str = "text"               # parquet column holding text
    txt_granularity: str = "line"           # "line" | "document"
    seq_len: int = 512
    val_frac: float = 0.005
    test_frac: float = 0.0
    split_strategy: str = "hash"            # "hash" (only strategy for now)
    docs_per_shard: int = 50000


# ── SFT data ─────────────────────────────────────────────────────────
@dataclass
class SFTDataConfig:
    # inputs: {"train": [...], "val": [...]}  OR  {"all": [...]}
    inputs: dict[str, list[str]]
    format: str = "auto"                    # "txt" (jsonl) | "parquet" | "auto"
    prompt_template: str = "{prompt}"
    completion_field: str = "completion"
    max_prompt_len: int = 384
    max_completion_len: int = 128
    val_frac: float = 0.05                  # used only when inputs has "all"
    split_train: str = "train"
    split_val: str = "validation"
    max_skipped_frac: float = 0.5           # fail loudly if more records skipped


# ── LoRA ─────────────────────────────────────────────────────────────
@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05


# ── Finetune method ──────────────────────────────────────────────────
@dataclass
class FinetuneConfig:
    method: str = "lora"                    # "lora" | "bitfit" | "full"
    base_checkpoint: str | None = None      # None -> auto-wired in mode "both"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    eval_metric: str = "perplexity"         # "perplexity" | "exact_match" | "f1"


# ── Mode sections ────────────────────────────────────────────────────
@dataclass
class PretrainSection:
    data: PretrainDataConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass
class SFTSection:
    data: SFTDataConfig
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ── Root config ──────────────────────────────────────────────────────
@dataclass
class RunConfig:
    name: str
    mode: str                               # "pretrain" | "sft" | "both"
    model: ModelConfig
    tokenizer: TokenizerConfig
    pretrain: PretrainSection | None = None
    sft: SFTSection | None = None
