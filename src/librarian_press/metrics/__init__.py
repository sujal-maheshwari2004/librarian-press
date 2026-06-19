"""
metrics — Prometheus-style metrics for training and inference.

Metrics are always recorded into an in-process registry (cheap, no-op if nothing
scrapes them). Call start_metrics_server(port) to expose them over HTTP for a
Prometheus / Grafana Alloy scraper. Recording helpers below are the only API the
rest of the codebase needs.
"""

from __future__ import annotations

from .core import Counter, Gauge, REGISTRY
from .server import start_server

# ── training metrics (labelled by run + mode) ────────────────────────
_TRAIN_LOSS = REGISTRY.register(Gauge("librarian_train_loss", "Training loss at the last log step"))
_TRAIN_LR = REGISTRY.register(Gauge("librarian_train_learning_rate", "Current learning rate"))
_TRAIN_GRAD = REGISTRY.register(Gauge("librarian_train_grad_norm", "Gradient L2 norm at the last step"))
_TRAIN_STEP = REGISTRY.register(Gauge("librarian_train_step", "Current training step"))
_TRAIN_TOKS = REGISTRY.register(Gauge("librarian_train_tokens_per_second", "Training throughput (tokens/sec)"))
_TRAIN_GPU = REGISTRY.register(Gauge("librarian_gpu_memory_bytes", "Allocated GPU memory in bytes"))
_TRAIN_VAL = REGISTRY.register(Gauge("librarian_val_loss", "Most recent validation loss"))
_TRAIN_STEPS_TOTAL = REGISTRY.register(Counter("librarian_train_steps_total", "Total logged training steps"))

# ── inference metrics (labelled by model) ────────────────────────────
_INFER_REQS = REGISTRY.register(Counter("librarian_inference_requests_total", "Total inference/chat requests"))
_INFER_TOKS = REGISTRY.register(Counter("librarian_inference_generated_tokens_total", "Total generated tokens"))
_INFER_TPS = REGISTRY.register(Gauge("librarian_inference_tokens_per_second", "Throughput of the last generation"))
_INFER_LAT = REGISTRY.register(Gauge("librarian_inference_latency_seconds", "Latency of the last generation"))


def start_metrics_server(port: int, host: str = "0.0.0.0"):
    return start_server(port, host)


def record_train(run, mode, step, loss, lr, grad, tokens_per_sec, gpu_bytes):
    lbl = {"run": run or "default", "mode": mode or "train"}
    _TRAIN_LOSS.set(loss, **lbl)
    _TRAIN_LR.set(lr, **lbl)
    _TRAIN_GRAD.set(grad, **lbl)
    _TRAIN_STEP.set(step, **lbl)
    _TRAIN_TOKS.set(tokens_per_sec, **lbl)
    _TRAIN_GPU.set(gpu_bytes, **lbl)
    _TRAIN_STEPS_TOTAL.inc(1, **lbl)


def record_val(run, mode, val_loss):
    _TRAIN_VAL.set(val_loss, run=run or "default", mode=mode or "train")


def record_inference(model, generated_tokens, latency_seconds):
    lbl = {"model": model or "default"}
    _INFER_REQS.inc(1, **lbl)
    _INFER_TOKS.inc(generated_tokens, **lbl)
    _INFER_LAT.set(latency_seconds, **lbl)
    _INFER_TPS.set(generated_tokens / latency_seconds if latency_seconds > 0 else 0.0, **lbl)


__all__ = ["start_metrics_server", "record_train", "record_val", "record_inference", "REGISTRY"]
