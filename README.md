# librarian-press

> Bring clean data, get a trained LLM. A config-driven framework for **pretraining
> and fine-tuning** Librarian-family GPT models — from a few-million-parameter toy
> to billion-parameter models — driven entirely by one JSON file.

Like the printing press did for the written word, `librarian-press` takes LLM
training out of the lab: you supply cleaned **Parquet/`.txt`** data and a config,
and it handles tokenizer training → tokenization → packing → training →
evaluation → inference. Pretraining and supervised fine-tuning (SFT) are
separable — run one, the other, or both end-to-end.

---

## Highlights

- **One config, three modes** — `pretrain`, `sft`, or `both` from a single JSON file.
- **Bring-your-own data** — local Parquet and `.txt`; no scraping, no hidden cleaning.
- **Modern architecture** — GPT with RoPE, RMSNorm, SwiGLU MLP, weight tying.
- **Fine-tuning built in** — LoRA, BitFit, or full fine-tune, with completion-only masked loss.
- **Resumable pipeline** — every data stage is shard-tracked with atomic, restart-safe manifests.
- **Multi-GPU** — data-parallel (DDP) via `torchrun`, single-GPU path unchanged.
- **Export & chat** — bundle a trained run into a portable folder and chat with it, Ollama-style.
- **Observable** — opt-in Prometheus-style `/metrics` endpoint for Grafana, with zero extra dependencies.

---

## Installation

```bash
pip install librarian-press
```

This provides the `librarian-press` command (short alias **`lpress`**).

**From source (development):**

```bash
git clone <repo-url> && cd librarian-fw
uv sync          # or:  pip install -e .
```

**Requirements:** Python 3.12–3.13, PyTorch ≥ 2.6 (CUDA build recommended). All
other dependencies (`numpy`, `tokenizers`, `pyarrow`, `tqdm`, `tensorboard`,
`requests`) install automatically.

---

## Quickstart

A complete CPU-only run is included for a fast end-to-end smoke test:

```bash
# pretrain a tiny model, then SFT it, all from bundled fixtures
librarian-press run --config configs/run_dummy.json

# package the result into a portable model and chat with it
librarian-press export --config configs/run_dummy.json --name demo
librarian-press chat demo
```

For a real run, point a config at your data and go:

```bash
librarian-press pretrain --config configs/pretrain_130M.json
```

---

## Concepts

**Modes.** The top-level `mode` selects the pipeline:

| Mode | Pipeline |
|---|---|
| `pretrain` | ingest → train tokenizer → tokenize → pack → train → eval |
| `sft` | ingest → prepare (masked) → train → eval |
| `both` | pretrain, then SFT auto-wired onto the fresh checkpoint + tokenizer |

**Run directory.** Everything for a run lives under `runs/<name>/`:

```text
runs/<name>/
  data/          ingested shards, tokenized shards, packed splits
  manifests/     per-stage progress (resume-safe)
  tokenizer/     trained tokenizer
  checkpoints/   pretrain/  and  sft/<method>/   (best.pt, last.pt, step_*.pt)
  logs/          eval results
```

Data stages are skipped automatically when their manifest is already complete,
so re-running a command resumes rather than recomputes.

---

## CLI reference

```text
librarian-press <command> [options]      # alias: lpress
```

| Command | Purpose |
|---|---|
| `pretrain --config <cfg>` | Run the pretraining pipeline |
| `sft --config <cfg>` | Run the supervised fine-tuning pipeline |
| `run --config <cfg>` | Pretrain then SFT end-to-end (config `mode` must be `both`) |
| `tokenizer --config <cfg>` | Train the tokenizer only |
| `eval --config <cfg> [--checkpoint CKPT]` | Evaluate a trained checkpoint |
| `infer --config <cfg> --checkpoint CKPT [--prompt "..."]` | One-off generation / ad-hoc chat |
| `export --config <cfg> --name NAME` | Bundle a trained run into a portable model |
| `chat <model-name>` | Interactive streaming chat with an exported model |
| `models` | List exported models |

### Common options

| Option | Applies to | Meaning |
|---|---|---|
| `--start-from <stage>` | `pretrain`, `sft` | Resume the pipeline from a specific stage |
| `--resume <checkpoint>` | `pretrain`, `sft` | Resume training from a checkpoint |
| `--cleanup` | `pretrain`, `run` | Delete intermediate artifacts after success (opt-in) |
| `--metrics-port <port>` | training + `chat` | Expose Prometheus-style metrics (see [Monitoring](#monitoring)) |
| `--from pretrain\|sft`, `--checkpoint` | `export` | Choose which half / which checkpoint to export |
| `--temperature`, `--top-k`, `--max-new-tokens` | `export` | Default sampling parameters baked into the bundle |

---

## Configuration

A single JSON file. Shared `model` and `tokenizer` sections, plus a `pretrain`
and/or `sft` section depending on `mode`.

### Top level

| Field | Type | Notes |
|---|---|---|
| `name` | string | Run name; defines `runs/<name>/` |
| `mode` | enum | `pretrain` · `sft` · `both` |
| `model` | object | Architecture (below), or `{"config_path": "model_130M.json"}` |
| `tokenizer` | object | Tokenizer config (below) |
| `pretrain` | object | Required for `pretrain`/`both` |
| `sft` | object | Required for `sft`/`both` |

### `model`

| Field | Default | Notes |
|---|---|---|
| `vocab_size` | 16000 | Must be `< 65536` (tokens stored as uint16) |
| `dim` | 512 | Must be divisible by `n_heads`; head dim should be even |
| `n_layers` | 12 | |
| `n_heads` | 8 | |
| `hidden_dim` | 2048 | MLP inner size (commonly `4 × dim`) |
| `max_seq_len` | 512 | Context length |
| `dropout` | 0.1 | |
| `rope_theta` | 10000.0 | RoPE base |
| `tie_embeddings` | true | Share input/output embeddings |

### `tokenizer`

| Field | Default | Notes |
|---|---|---|
| `path` | — | Where the tokenizer is / will be written |
| `train_if_missing` | false | Pretrain may train one; SFT must reuse an existing one |
| `vocab_size` | 32000 | Must equal `model.vocab_size` |
| `min_frequency` | 2 | BPE merge threshold |
| `special_tokens` | `["<pad>","<bos>","<eos>","<unk>"]` | |

### `training` (per mode)

`lr`, `min_lr`, `warmup_steps`, `total_steps`, `batch_size`, `grad_accum`,
`weight_decay`, `mixed_precision`, `eval_interval`, `save_interval`, `device`.
Cosine schedule with warmup; AdamW with decoupled weight decay; AMP + gradient
accumulation + gradient clipping.

### `pretrain.data`

| Field | Default | Notes |
|---|---|---|
| `inputs` | — | List of file paths / globs |
| `format` | `auto` | `txt` · `parquet` · `auto` |
| `text_column` | `text` | Parquet column holding text |
| `txt_granularity` | `line` | `line` (one doc per line) or `document` (whole file) |
| `seq_len` | 512 | Packed sequence length (`≤ model.max_seq_len`) |
| `val_frac` / `test_frac` | 0.005 / 0.0 | Deterministic hash split |

### `sft.data`

| Field | Default | Notes |
|---|---|---|
| `inputs` | — | `{"train": [...], "val": [...]}` or `{"all": [...]}` |
| `format` | `auto` | Parquet rows, or JSON-per-line `.txt` |
| `prompt_template` | `"{prompt}"` | e.g. `"Context: {context}\nQuestion: {question}\nAnswer:"` |
| `completion_field` | `"completion"` | Dotted/array fields supported, e.g. `answers.text[0]` |
| `max_prompt_len` / `max_completion_len` | 384 / 128 | Token caps |

### `sft.finetune`

| Field | Default | Notes |
|---|---|---|
| `method` | `lora` | `lora` · `bitfit` · `full` |
| `base_checkpoint` | null | Pretrained weights; auto-wired in `both` |
| `lora` | `{rank:8, alpha:16, dropout:0.05}` | LoRA hyperparameters |
| `eval_metric` | `perplexity` | `perplexity` · `exact_match` · `f1` |

### Example (`mode: both`)

```jsonc
{
  "name": "my-model",
  "mode": "both",
  "model": { "vocab_size": 32000, "dim": 768, "n_layers": 12, "n_heads": 12,
             "hidden_dim": 3072, "max_seq_len": 1024 },
  "tokenizer": { "path": "runs/my-model/tokenizer/tokenizer.json",
                 "train_if_missing": true, "vocab_size": 32000 },
  "pretrain": {
    "data": { "inputs": ["./data/corpus/*.parquet"], "text_column": "text",
              "seq_len": 1024, "val_frac": 0.005 },
    "training": { "lr": 3e-4, "total_steps": 100000, "batch_size": 32, "grad_accum": 4 }
  },
  "sft": {
    "finetune": { "method": "lora", "base_checkpoint": null, "eval_metric": "f1" },
    "data": { "inputs": { "train": ["./sft/train/*.parquet"], "val": ["./sft/val/*.parquet"] },
              "prompt_template": "Q: {question}\nA:", "completion_field": "answer" },
    "training": { "lr": 2e-4, "total_steps": 5000, "batch_size": 16, "grad_accum": 4 }
  }
}
```

See [`configs/`](configs/) for runnable examples.

---

## Data you provide

You own data cleanliness — the framework parses, it does not scrape or quality-filter.

- **Pretraining** — `.txt` (one document per line, or the whole file as one
  document) and/or `.parquet` with a configurable `text_column`.
- **SFT** — `.parquet` rows or JSON-per-line `.txt`, mapped through
  `prompt_template` + `completion_field`. Loss is computed on completion tokens
  only; prompt tokens are masked out.

---

## Inference: export & chat

Consolidate a trained run into a portable, self-contained model folder, then chat
with it from the terminal:

```bash
# bundle -> ~/.librarian-press/models/<name>/  (override with LIBRARIAN_PRESS_HOME)
librarian-press export --config configs/run_both.json --name my-bot

# stream tokens until you type /bye
librarian-press chat my-bot

librarian-press models     # list exported models
```

`export` merges **LoRA/BitFit adapters into the base weights**, writing a single
plain weights file plus the tokenizer and a `bundle.json` (model config, chat
prompt template, sampling defaults). The folder is fully standalone — copy or
share it freely.

---

## Multi-GPU (DDP)

Launch any training command under `torchrun` to data-parallelize across GPUs:

```bash
torchrun --nproc_per_node=4 --module librarian_press.cli.main \
  pretrain --config configs/pretrain_130M.json
```

Each GPU holds a full model copy; gradients all-reduce at the accumulation
boundary, so the effective batch is `batch_size × grad_accum × num_gpus`. Data
stages and evaluation run on rank 0; only rank 0 writes checkpoints and logs.
Running without `torchrun` is the unchanged single-GPU path. DDP scales
throughput — the model must still fit on one GPU (FSDP/sharding is not built in).

---

## Monitoring

Add `--metrics-port <port>` to any training or `chat` command to expose a
Prometheus-style metrics endpoint — pull model, plain text exposition format, **no
`prometheus_client` dependency**:

```bash
librarian-press run  --config configs/run_both.json --metrics-port 9099
curl http://localhost:9099/metrics
```

Point Prometheus (or Grafana Alloy/Agent) at `/metrics` and dashboard it in
Grafana. Under DDP only rank 0 serves. Configurable via
`LIBRARIAN_PRESS_METRICS_PORT` / `LIBRARIAN_PRESS_METRICS_HOST`.

| Metric | Type | Labels |
|---|---|---|
| `librarian_train_loss`, `librarian_val_loss` | gauge | `run`, `mode` |
| `librarian_train_learning_rate`, `librarian_train_grad_norm`, `librarian_train_step` | gauge | `run`, `mode` |
| `librarian_train_tokens_per_second`, `librarian_gpu_memory_bytes` | gauge | `run`, `mode` |
| `librarian_train_steps_total` | counter | `run`, `mode` |
| `librarian_inference_requests_total`, `librarian_inference_generated_tokens_total` | counter | `model` |
| `librarian_inference_tokens_per_second`, `librarian_inference_latency_seconds` | gauge | `model` |

---

## Project layout

```text
src/librarian_press/
  cli/          command-line entrypoint
  config/       JSON schema, loading, run paths
  model/        GPT, attention, RoPE, RMSNorm, MLP, LoRA, build/load
  tokenizer/    BPE training + loading
  data/         ingest, shard, tokenize, pack, prepare_sft, datasets
  pipeline/     manifests, atomic writers, cleanup, stage orchestration
  training/     trainer, optimizer, scheduler, checkpoints
  evaluation/   perplexity, generation metrics, router
  inference/    sampling, generation, method-aware loading
  serve/        export bundles, registry, chat REPL
  metrics/      Prometheus-style registry + HTTP server
  utils/        logging, device, distributed
```

---

## License

See [LICENSE](LICENSE).
