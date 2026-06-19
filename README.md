# Librarian

A config-driven framework to **pretrain** and **fine-tune** Librarian GPT models
from your own cleaned data. You bring clean **Parquet/`.txt`** files and one JSON
config; the framework handles tokenizer training, tokenization, packing, training,
evaluation, and inference.

Pretraining and SFT are separable: run one, the other, or both end-to-end.

## Install

```bash
uv sync            # or: pip install -e .
```

## Usage

```bash
# Pretrain a base model from raw text
librarian pretrain --config configs/pretrain_130M.json

# Supervised fine-tune an existing base model
librarian sft --config configs/sft_qa_lora.json

# Both, end-to-end (SFT auto-consumes the pretrain checkpoint + tokenizer)
librarian run --config configs/run_both.json     # config mode must be "both"

# Other commands
librarian tokenizer --config <cfg>                # train tokenizer only
librarian eval      --config <cfg> [--checkpoint CKPT]
librarian infer     --config <cfg> --checkpoint CKPT [--prompt "..."]
```

All commands accept `--start-from <stage>` (data stages resume via per-stage
manifests) and the train pipelines accept `--resume <checkpoint>`.

### Multi-GPU (DDP)

Launch any training command under `torchrun` to data-parallelize across GPUs:

```bash
torchrun --nproc_per_node=4 --module librarian.cli.main pretrain --config configs/pretrain_130M.json
torchrun --nproc_per_node=4 --module librarian.cli.main run      --config configs/run_both.json
```

Each GPU holds a full model copy; gradients all-reduce at the accumulation
boundary, so the effective batch is `batch_size × grad_accum × num_gpus`. Data
stages (ingest/tokenize/pack/prepare) and evaluation run on rank 0; only rank 0
writes checkpoints and logs. Running without `torchrun` is unchanged single-GPU.
DDP scales throughput — the model must still fit on one GPU (sharding/FSDP for
larger-than-one-GPU models is not built in yet).

## Data you provide

- **Pretraining**: `.txt` (one document per line, or whole-file) and/or `.parquet`
  with a configurable `text_column`. Already cleaned — no quality filtering is done.
- **SFT**: `.parquet` rows or JSON-per-line `.txt`, mapped via `prompt_template`
  (e.g. `"Context: {context}\nQuestion: {question}\nAnswer:"`) and
  `completion_field` (supports dotted/array fields like `answers.text[0]`).
  Loss flows only through completion tokens.

## Config

One JSON, `mode ∈ {pretrain, sft, both}`, with shared `model` + `tokenizer`
sections and per-mode `data`/`training` (and `finetune` for SFT). See
[configs/](configs/) for runnable examples, including a CPU-only
`run_dummy.json` for a fast end-to-end smoke test.

## Layout

Everything for a run lives under `runs/<name>/`: ingested shards, manifests,
the tokenizer, packed splits, checkpoints (`checkpoints/pretrain/`,
`checkpoints/sft/<method>/`), and eval results.
