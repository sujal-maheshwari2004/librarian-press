"""
librarian CLI

  librarian pretrain  --config x.json [--start-from S] [--resume CKPT] [--cleanup]
  librarian sft       --config y.json [--start-from S] [--resume CKPT]
  librarian run       --config z.json [--cleanup]          # pretrain -> sft
  librarian tokenizer --config x.json                       # train tokenizer only
  librarian eval      --config c.json [--checkpoint CKPT]
  librarian infer     --config c.json --checkpoint CKPT [--prompt "..."]
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


def _cmd_pretrain(args):
    from ..pipeline.orchestrator import run_pretrain

    cfg = load_config(args.config)
    _require_mode(cfg, "pretrain")
    run_pretrain(cfg, RunPaths(cfg.name), start_from=args.start_from,
                 resume=args.resume, cleanup=args.cleanup, run_id=_run_id())


def _cmd_sft(args):
    from ..pipeline.orchestrator import run_sft

    cfg = load_config(args.config)
    _require_mode(cfg, "sft")
    run_sft(cfg, RunPaths(cfg.name), start_from=args.start_from,
            resume=args.resume, run_id=_run_id())


def _cmd_run(args):
    from ..pipeline.orchestrator import run_all

    cfg = load_config(args.config)
    if cfg.mode != "both":
        raise SystemExit(f"`run` requires mode='both' (got {cfg.mode!r})")
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


def _require_mode(cfg, needed):
    if needed not in (cfg.mode, "both") and cfg.mode != needed:
        # allow 'both' configs to run a single phase
        if not (cfg.mode == "both" and needed in ("pretrain", "sft")):
            raise SystemExit(
                f"config mode={cfg.mode!r} does not support `{needed}`"
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="librarian",
                                description="Pretrain and fine-tune Librarian GPT models from clean local data")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("pretrain", help="run the pretraining pipeline")
    pt.add_argument("--config", required=True)
    pt.add_argument("--start-from", choices=PRETRAIN_STAGES, default="ingest_pre")
    pt.add_argument("--resume", default=None)
    pt.add_argument("--cleanup", action="store_true")
    pt.set_defaults(func=_cmd_pretrain)

    sf = sub.add_parser("sft", help="run the supervised fine-tuning pipeline")
    sf.add_argument("--config", required=True)
    sf.add_argument("--start-from", choices=SFT_STAGES, default="ingest_sft")
    sf.add_argument("--resume", default=None)
    sf.set_defaults(func=_cmd_sft)

    rn = sub.add_parser("run", help="pretrain then SFT end-to-end (mode=both)")
    rn.add_argument("--config", required=True)
    rn.add_argument("--cleanup", action="store_true")
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
