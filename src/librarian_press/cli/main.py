"""
librarian-press CLI  (command: `librarian-press` or short alias `lpress`)

  librarian-press pretrain  --config x.json [--start-from S] [--resume CKPT] [--cleanup]
  librarian-press sft       --config y.json [--start-from S] [--resume CKPT]
  librarian-press run       --config z.json [--cleanup]          # pretrain -> sft
  librarian-press tokenizer --config x.json                       # train tokenizer only
  librarian-press eval      --config c.json [--checkpoint CKPT]
  librarian-press infer     --config c.json --checkpoint CKPT [--prompt "..."]
"""

from __future__ import annotations

import argparse
import os
import time

from ..config.load import load_config
from ..config.paths import RunPaths
from ..pipeline.stages import PRETRAIN_STAGES, SFT_STAGES


def _run_id() -> int:
    return int(os.environ.get("RUN_ID", 0)) or int(time.time())


def _maybe_start_metrics(args, *, main_only: bool):
    """Start the Prometheus-style /metrics server if requested (flag or env).
    For distributed training only rank 0 serves (avoids port clashes)."""
    port = getattr(args, "metrics_port", None) or int(os.environ.get("LIBRARIAN_PRESS_METRICS_PORT", "0")) or None
    if not port:
        return
    if main_only:
        from ..utils.distributed import is_main
        if not is_main():
            return
    from ..metrics import start_metrics_server
    host = getattr(args, "metrics_host", None) or os.environ.get("LIBRARIAN_PRESS_METRICS_HOST", "0.0.0.0")
    start_metrics_server(port, host)


def _cmd_pretrain(args):
    from ..pipeline.orchestrator import run_pretrain

    cfg = load_config(args.config)
    _require_mode(cfg, "pretrain")
    _maybe_start_metrics(args, main_only=True)
    run_pretrain(cfg, RunPaths(cfg.name), start_from=args.start_from,
                 resume=args.resume, cleanup=args.cleanup, run_id=_run_id())


def _cmd_sft(args):
    from ..pipeline.orchestrator import run_sft

    cfg = load_config(args.config)
    _require_mode(cfg, "sft")
    _maybe_start_metrics(args, main_only=True)
    run_sft(cfg, RunPaths(cfg.name), start_from=args.start_from,
            resume=args.resume, run_id=_run_id())


def _cmd_run(args):
    from ..pipeline.orchestrator import run_all

    cfg = load_config(args.config)
    if cfg.mode != "both":
        raise SystemExit(f"`run` requires mode='both' (got {cfg.mode!r})")
    _maybe_start_metrics(args, main_only=True)
    run_all(cfg, RunPaths(cfg.name), cleanup=args.cleanup, run_id=_run_id())


def _cmd_tokenizer(args):
    cfg = load_config(args.config)
    _require_mode(cfg, "pretrain")
    _tokenizer_only(cfg)


def _tokenizer_only(cfg):
    paths = RunPaths(cfg.name)
    paths.make_dirs()
    # ingest then tokenizer by trimming the stage list via monkey-free approach:
    from ..data.shard import ingest_pretrain
    from ..tokenizer.train import train_tokenizer
    from ..utils.logging import StageLogger
    from pathlib import Path

    log = StageLogger(run_name=cfg.name, run_id=_run_id())
    log.start("ingest_pre")
    ingest_pretrain(cfg.pretrain.data, paths, paths.manifest("ingest_pre"), log)
    log.end("ingest_pre")
    if Path(cfg.tokenizer.path).exists():
        print(f"[tokenizer] exists at {cfg.tokenizer.path} — nothing to do")
        return
    log.start("train_tokenizer")
    train_tokenizer(cfg.tokenizer, paths.pretrain_data_dir / "raw", log)
    log.end("train_tokenizer")


def _cmd_eval(args):
    cfg = load_config(args.config)
    paths = RunPaths(cfg.name)
    from ..pipeline.orchestrator import _eval_pretrain, _eval_sft
    from ..utils.logging import StageLogger

    log = StageLogger(run_name=cfg.name, run_id=_run_id())
    if cfg.mode in ("pretrain", "both"):
        _eval_pretrain(cfg, paths, args.checkpoint, log)
    if cfg.mode in ("sft", "both"):
        _eval_sft(cfg, paths, args.checkpoint, log)


def _cmd_infer(args):
    from ..inference.load_model import load_model
    from ..inference.generate import generate
    from ..inference.chat import chat
    from ..tokenizer.load import load_tokenizer
    from ..utils.device import resolve_device
    import torch

    cfg = load_config(args.config)
    device = resolve_device(
        (cfg.sft or cfg.pretrain).training.device if (cfg.sft or cfg.pretrain) else "cpu"
    )
    tokenizer = load_tokenizer(cfg.tokenizer.path)
    model = load_model(cfg.model, args.checkpoint, device)

    if args.prompt is None:
        chat(model, tokenizer)
        return
    ids = tokenizer.encode(args.prompt).ids
    idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    out = generate(model, idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))


def _cmd_export(args):
    from ..serve.export import export_model

    cfg = load_config(args.config)
    name = args.name or cfg.name
    export_model(
        cfg, name=name, checkpoint=args.checkpoint, source=getattr(args, "from"),
        temperature=args.temperature, top_k=args.top_k, max_new_tokens=args.max_new_tokens,
    )


def _cmd_chat(args):
    from ..serve.chat import run_chat

    _maybe_start_metrics(args, main_only=False)
    run_chat(args.model)


def _cmd_models(args):
    from ..serve.registry import list_models, models_dir

    names = list_models()
    if not names:
        print(f"No models in {models_dir()}. Export one: librarian-press export --config <cfg> --name <name>")
        return
    print(f"Models in {models_dir()}:")
    for n in names:
        print(f"  {n}")


def _require_mode(cfg, needed):
    if needed not in (cfg.mode, "both") and cfg.mode != needed:
        # allow 'both' configs to run a single phase
        if not (cfg.mode == "both" and needed in ("pretrain", "sft")):
            raise SystemExit(
                f"config mode={cfg.mode!r} does not support `{needed}`"
            )


def _add_metrics_args(parser):
    parser.add_argument("--metrics-port", type=int, default=None, dest="metrics_port",
                        help="expose Prometheus-style metrics at http://host:PORT/metrics")
    parser.add_argument("--metrics-host", default=None, dest="metrics_host",
                        help="bind host for the metrics server (default 0.0.0.0)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="librarian-press",
                                description="Pretrain and fine-tune Librarian-family GPT models from clean local data")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("pretrain", help="run the pretraining pipeline")
    pt.add_argument("--config", required=True)
    pt.add_argument("--start-from", choices=PRETRAIN_STAGES, default="ingest_pre")
    pt.add_argument("--resume", default=None)
    pt.add_argument("--cleanup", action="store_true")
    _add_metrics_args(pt)
    pt.set_defaults(func=_cmd_pretrain)

    sf = sub.add_parser("sft", help="run the supervised fine-tuning pipeline")
    sf.add_argument("--config", required=True)
    sf.add_argument("--start-from", choices=SFT_STAGES, default="ingest_sft")
    sf.add_argument("--resume", default=None)
    _add_metrics_args(sf)
    sf.set_defaults(func=_cmd_sft)

    rn = sub.add_parser("run", help="pretrain then SFT end-to-end (mode=both)")
    rn.add_argument("--config", required=True)
    rn.add_argument("--cleanup", action="store_true")
    _add_metrics_args(rn)
    rn.set_defaults(func=_cmd_run)

    tk = sub.add_parser("tokenizer", help="train the tokenizer only")
    tk.add_argument("--config", required=True)
    tk.set_defaults(func=_cmd_tokenizer)

    ev = sub.add_parser("eval", help="evaluate a trained checkpoint")
    ev.add_argument("--config", required=True)
    ev.add_argument("--checkpoint", default=None)
    ev.set_defaults(func=_cmd_eval)

    inf = sub.add_parser("infer", help="generate text / chat with a checkpoint")
    inf.add_argument("--config", required=True)
    inf.add_argument("--checkpoint", required=True)
    inf.add_argument("--prompt", default=None)
    inf.add_argument("--max-new-tokens", type=int, default=100, dest="max_new_tokens")
    inf.add_argument("--temperature", type=float, default=1.0)
    inf.add_argument("--top-k", type=int, default=50, dest="top_k")
    inf.set_defaults(func=_cmd_infer)

    ex = sub.add_parser("export", help="bundle a trained run into a portable model folder")
    ex.add_argument("--config", required=True)
    ex.add_argument("--name", default=None, help="model name in the registry (default: config name)")
    ex.add_argument("--checkpoint", default=None, help="checkpoint to export (default: best/last)")
    ex.add_argument("--from", dest="from", choices=["pretrain", "sft"], default=None,
                    help="which half of the run to export (default: sft if present)")
    ex.add_argument("--temperature", type=float, default=0.8)
    ex.add_argument("--top-k", type=int, default=40, dest="top_k")
    ex.add_argument("--max-new-tokens", type=int, default=256, dest="max_new_tokens")
    ex.set_defaults(func=_cmd_export)

    ch = sub.add_parser("chat", help="chat with an exported model (Ollama-style REPL)")
    ch.add_argument("model", help="exported model name")
    _add_metrics_args(ch)
    ch.set_defaults(func=_cmd_chat)

    ls = sub.add_parser("models", help="list exported models")
    ls.set_defaults(func=_cmd_models)

    return p


def main(argv=None):
    from ..utils.distributed import init_distributed, cleanup

    parser = build_parser()
    args = parser.parse_args(argv)
    # no-op unless launched with WORLD_SIZE>1 (e.g. torchrun); enables DDP
    init_distributed()
    try:
        args.func(args)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
