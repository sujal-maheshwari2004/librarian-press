"""Stage orderings for each mode. The orchestrator runs the slice from --start-from."""

PRETRAIN_STAGES = [
    "ingest_pre",
    "train_tokenizer",
    "tokenize",
    "pack",
    "train_pre",
    "eval_pre",
]

SFT_STAGES = [
    "ingest_sft",
    "prepare_sft",
    "train_sft",
    "eval_sft",
]

# Stages whose completion is tracked by a shard manifest (skippable on resume).
DATA_STAGES = {"ingest_pre", "train_tokenizer", "tokenize", "pack", "ingest_sft", "prepare_sft"}

# Evaluation stages. Under DDP these run on rank 0 only (the train stages are the
# ones that fan out across ranks). Data stages also run on rank 0 only.
EVAL_STAGES = {"eval_pre", "eval_sft"}

# Stages that run on rank 0 only when distributed (everything except train_*).
MAIN_ONLY_STAGES = DATA_STAGES | EVAL_STAGES
