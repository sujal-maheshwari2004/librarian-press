"""
logging.py — console + optional remote dashboard logging.

Unified from the base and SFT loggers: both StageLogger and TrainingLogger
accept an optional `run_name` (used by SFT-style runs) and a `run_id`.
Remote posting is best-effort and never interrupts a run.
"""

from __future__ import annotations

import os
import time

import requests
import torch
from tqdm import tqdm

from .distributed import is_main

GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"

_DASHBOARD_KEY = os.environ.get("DASHBOARD_KEY", "")

STAGE_LABELS = {
    "ingest_pre": "Pretrain Ingest",
    "train_tokenizer": "Tokenizer Training",
    "tokenize": "Tokenization",
    "pack": "Token Packing",
    "train_pre": "Pretrain",
    "eval_pre": "Pretrain Eval",
    "ingest_sft": "SFT Ingest",
    "prepare_sft": "SFT Prepare",
    "train_sft": "Finetune",
    "eval_sft": "Finetune Eval",
    "train": "Training",
    "evaluate": "Evaluation",
}


class _BaseSender:
    API_BASE = "https://librarian-logging-api-point.vercel.app"

    def __init__(self):
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_DASHBOARD_KEY}",
        }

    def _post(self, endpoint: str, payload: dict):
        if not _DASHBOARD_KEY or not getattr(self, "enabled", True):
            return
        try:
            requests.post(
                f"{self.API_BASE}/{endpoint}",
                json=payload,
                headers=self._headers,
                timeout=5,
            )
        except Exception:
            pass


# -- StageLogger ------------------------------------------------------
class StageLogger(_BaseSender):
    def __init__(self, run_name: str | None = None, run_id: int | None = None):
        super().__init__()
        self.enabled = is_main()
        self.run_name = run_name
        self.run_id = run_id or int(time.time())
        self._stage_start: dict[str, float] = {}

        if not self.enabled:
            return
        tqdm.write("----------------------------------------")
        label = f"  {run_name}" if run_name else ""
        tqdm.write(f"{CYAN}PIPELINE{RESET}{label}  run_id: {self.run_id}")
        tqdm.write("----------------------------------------")

    def start(self, stage: str):
        self._stage_start[stage] = time.time()
        if not self.enabled:
            return
        tqdm.write(f"\n{CYAN}> START   {RESET}{STAGE_LABELS.get(stage, stage)}")
        self._post("stage_metrics", {
            "run_id": self.run_id, "run_name": self.run_name,
            "stage": stage, "event": "start", "timestamp": time.time(), "metrics": {},
        })

    def progress(self, stage: str, metrics: dict):
        if not self.enabled:
            return
        parts = "  ".join(f"{k} {YELLOW}{_fmt(v)}{RESET}" for k, v in metrics.items())
        tqdm.write(f"  {DIM}>{RESET} {DIM}{STAGE_LABELS.get(stage, stage)}{RESET}  {parts}")
        self._post("stage_metrics", {
            "run_id": self.run_id, "run_name": self.run_name,
            "stage": stage, "event": "progress", "timestamp": time.time(),
            "metrics": _serialise(metrics),
        })

    def end(self, stage: str, metrics: dict | None = None):
        elapsed = time.time() - self._stage_start.get(stage, time.time())
        metrics = metrics or {}
        if not self.enabled:
            return
        parts = "  ".join(f"{k} {GREEN}{_fmt(v)}{RESET}" for k, v in metrics.items())
        tqdm.write(
            f"  {GREEN}[OK] DONE   {RESET}{STAGE_LABELS.get(stage, stage)}  "
            f"{DIM}elapsed {elapsed:.1f}s{RESET}  {parts}"
        )
        self._post("stage_metrics", {
            "run_id": self.run_id, "run_name": self.run_name,
            "stage": stage, "event": "end", "elapsed_s": round(elapsed, 2),
            "timestamp": time.time(), "metrics": _serialise(metrics),
        })

    def error(self, stage: str, message: str):
        if not self.enabled:
            return
        tqdm.write(f"  {RED}[ERR] ERROR  {RESET}{STAGE_LABELS.get(stage, stage)}  {message}")
        self._post("stage_metrics", {
            "run_id": self.run_id, "run_name": self.run_name,
            "stage": stage, "event": "error", "timestamp": time.time(),
            "metrics": {"error": message},
        })


# -- TrainingLogger ---------------------------------------------------
class TrainingLogger(_BaseSender):
    def __init__(
        self,
        seq_len: int,
        batch_size: int,
        run_name: str | None = None,
        run_id: int | None = None,
    ):
        super().__init__()
        self.enabled = is_main()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.run_name = run_name
        self.run_id = run_id or int(time.time())
        self.last_time = time.time()
        self.tokens_seen = 0

        if not self.enabled:
            return
        tqdm.write("----------------------------------------")
        tqdm.write(f"RUN START  {run_name or ''}")
        tqdm.write(f"run_id:  {self.run_id}")
        tqdm.write(f"seq_len: {seq_len}  batch: {batch_size}")
        tqdm.write("----------------------------------------")

    def throughput(self) -> int:
        now = time.time()
        elapsed = max(now - self.last_time, 1e-9)
        self.last_time = now
        tokens = self.seq_len * self.batch_size
        self.tokens_seen += tokens
        return int(tokens / elapsed)

    def gpu_mem(self) -> float:
        return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    def train(self, step, loss, lr, grad):
        if not self.enabled:
            return
        tok_s = self.throughput()
        gpu = self.gpu_mem()
        self._post("train_metrics", {
            "run_id": self.run_id, "run_name": self.run_name, "type": "train",
            "step": step, "loss": float(loss), "lr": float(lr),
            "grad_norm": float(grad), "tokens_per_sec": tok_s,
            "gpu_mem_gb": gpu, "timestamp": time.time(),
        })
        tqdm.write(
            f"{GREEN}TRAIN{RESET} step {step:06d} | loss {loss:6.4f} | "
            f"lr {lr:.2e} | grad {grad:5.2f} | tok/s {tok_s:6d} | gpu {gpu:4.2f}GB"
        )

    def eval(self, step, val_loss):
        if not self.enabled:
            return
        self._post("train_metrics", {
            "run_id": self.run_id, "run_name": self.run_name, "type": "eval",
            "step": step, "val_loss": float(val_loss), "timestamp": time.time(),
        })
        tqdm.write(f"{BLUE}EVAL {RESET} step {step:06d} | val_loss {val_loss:6.4f}")

    def checkpoint(self, step, val_loss):
        if not self.enabled:
            return
        self._post("train_metrics", {
            "run_id": self.run_id, "run_name": self.run_name, "type": "checkpoint",
            "step": step, "val_loss": float(val_loss), "timestamp": time.time(),
        })
        tqdm.write(f"{YELLOW}CKPT {RESET} step {step:06d} | new_best {val_loss:6.4f}")


# -- helpers ----------------------------------------------------------
def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, int) and v > 1_000_000:
        return f"{v / 1e6:.2f}M"
    if isinstance(v, int) and v > 1_000:
        return f"{v / 1e3:.1f}K"
    return str(v)


def _serialise(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        out[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
    return out
