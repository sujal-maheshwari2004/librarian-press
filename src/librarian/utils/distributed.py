"""
distributed.py — minimal DDP helpers.

No-ops unless the process was launched with WORLD_SIZE > 1 (e.g. via
`torchrun --nproc_per_node=N`). This keeps the single-GPU / CPU path byte-for-byte
the same: is_main() is True, is_distributed() is False, barrier() does nothing.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

_STATE = {"initialized": False, "rank": 0, "world_size": 1, "local_rank": 0}


def init_distributed() -> dict:
    """Initialize the process group if launched distributed. Safe to call always."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return dict(_STATE)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    _STATE.update(initialized=True, rank=rank, world_size=world_size, local_rank=local_rank)
    if rank == 0:
        print(f"[distributed] {backend} world_size={world_size}")
    return dict(_STATE)


def is_distributed() -> bool:
    return _STATE["initialized"] and _STATE["world_size"] > 1


def get_rank() -> int:
    return _STATE["rank"]


def get_world_size() -> int:
    return _STATE["world_size"]


def get_local_rank() -> int:
    return _STATE["local_rank"]


def is_main() -> bool:
    return _STATE["rank"] == 0


def barrier():
    if is_distributed():
        dist.barrier()


def cleanup():
    if _STATE["initialized"] and dist.is_initialized():
        dist.destroy_process_group()
        _STATE["initialized"] = False


def maybe_local_device(device: str) -> str:
    """Map a 'cuda' request to this rank's specific GPU when distributed."""
    if is_distributed() and device.startswith("cuda") and torch.cuda.is_available():
        return f"cuda:{_STATE['local_rank']}"
    return device


def all_reduce_mean(value: float, device: str) -> float:
    """Average a scalar across all ranks (returns value unchanged if single-process)."""
    if not is_distributed():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / _STATE["world_size"])
