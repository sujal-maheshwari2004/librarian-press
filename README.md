# librarian-press

A config-driven framework that turns clean data into trained LLMs. Like the
printing press did for books, `librarian-press` puts **pretraining and
fine-tuning of Librarian-family GPT models — at virtually any size — in
anyone's hands**: you bring clean **Parquet/`.txt`** files and one JSON config,
and it handles tokenizer training, tokenization, packing, training, evaluation,
and inference.

Pretraining and SFT are separable: run one, the other, or both end-to-end.

## Install

```bash
uv sync            # or: pip install -e .
```

This installs the `librarian-press` command (short alias: `lpress`).

## Usage

```bash
# Pretrain a base model from raw text
librarian-press pretrain --config configs/pretrain_130M.json

# Supervised fine-tune an existing base model
librarian-press sft --config configs/sft_qa_lora.json

# Both, end-to-end (SFT auto-consumes the pretrain checkpoint + tokenizer)
librarian-press run --config configs/run_both.json     # config mode must be "both"

# Other commands  (lpress is a shorthand for librarian-press)
lpress tokenizer --config <cfg>                # train tokenizer only
lpress eval      --config <cfg> [--checkpoint CKPT]
lpress infer     --config <cfg> --checkpoint CKPT [--prompt "..."]
```

### Export & chat (Ollama-style)

Package a trained run into a portable, self-contained model folder, then chat
with it from the terminal:

```bash
# 1. bundle the trained model into the local registry (~/.librarian-press/models/<name>)
librarian-press export --config configs/run_both.json --name my-bot

# 2. chat with it — streams tokens until you type /bye
librarian-press chat my-bot

# list everything you've exported
librarian-press models
```

`export` consolidates whatever the run produced into a single plain weights file
(**LoRA/BitFit adapters are merged into the base**), copies the tokenizer, and
writes a `bundle.json` with the model config + chat prompt template + generation
defaults. The resulting folder needs nothing from the original run to chat with,
so it's easy to copy or share. Override the source/checkpoint with
`--from pretrain|sft` and `--checkpoint`, and the sampling defaults with
`--temperature` / `--top-k` / `--max-new-tokens`. The registry location can be
moved with the `LIBRARIAN_PRESS_HOME` environment variable.

All commands accept `--start-from <stage>` (data stages resume via per-stage
manifests) and the train pipelines accept `--resume <checkpoint>`.

### Multi-GPU (DDP)

Launch any training command under `torchrun` to data-parallelize across GPUs:

```bash
torchrun --nproc_per_node=4 --module librarian_press.cli.main pretrain --config configs/pretrain_130M.json
torchrun --nproc_per_node=4 --module librarian_press.cli.main run      --config configs/run_both.json
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
