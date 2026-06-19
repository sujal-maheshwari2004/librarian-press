# Contributing to librarian-press

Thanks for your interest in improving **librarian-press**! This project was
created and is maintained by **Sujal Maheshwari**. Contributions — bug reports,
fixes, docs, and features — are welcome.

## Getting started

```bash
git clone <repo-url> && cd librarian-fw
uv sync            # or:  pip install -e .
```

**Requirements:** Python 3.12–3.13, PyTorch ≥ 2.6.

Run the bundled CPU-only smoke test to confirm your environment works
end-to-end (ingest → tokenizer → pack → pretrain → SFT → eval):

```bash
librarian-press run --config configs/run_dummy.json
```

It finishes in seconds and writes everything under `runs/librarian-dummy/`.

## Project layout

The package lives in `src/librarian_press/`. Each subpackage has one job:

| Area | Where |
|---|---|
| CLI entrypoint | `cli/` |
| Config schema, loading, run paths | `config/` |
| Model (GPT, RoPE, RMSNorm, MLP, LoRA) | `model/` |
| Tokenizer training + loading | `tokenizer/` |
| Data ingest, tokenize, pack, SFT prepare, datasets | `data/` |
| Manifests, atomic writers, stage orchestration | `pipeline/` |
| Trainer, optimizer, scheduler, checkpoints | `training/` |
| Evaluation (perplexity, generation metrics) | `evaluation/` |
| Inference (sampling, generation, loading) | `inference/` |
| Export bundles + chat REPL | `serve/` |
| Prometheus-style metrics | `metrics/` |
| Logging, device, distributed helpers | `utils/` |

## Guidelines

- **Match the surrounding style.** Keep modules small and single-purpose;
  prefer relative imports within the package.
- **The config is the contract.** New capabilities should be driven by config
  fields, validated in `config/load.py`, not hardcoded.
- **Keep the single-GPU path intact.** Distributed features must degrade to a
  no-op when not launched under `torchrun` (see `utils/distributed.py`).
- **Resumability matters.** New data stages should be shard-tracked through a
  `StageManifest` and use the atomic writers so they survive interruption.
- **Verify before you submit.** At minimum, `librarian-press run --config
  configs/run_dummy.json` must pass. For distributed changes, run the DDP smoke
  test in `tests/_ddp_smoke.py`.

## Submitting changes

1. Create a feature branch off the default branch.
2. Make focused commits with clear messages.
3. Ensure the smoke test passes and the package still imports
   (`python -c "import librarian_press"`).
4. Open a pull request describing **what** changed and **why**, and how you
   verified it.

## Reporting issues

Open an issue with:

- what you ran (the exact command + a minimal config),
- what you expected vs. what happened,
- the full error output and your environment (OS, Python, PyTorch, GPU).

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE) of this project.
