"""2-process DDP smoke test (gloo/CPU). Exercises the real distributed code path
without torchrun (Windows PyTorch here lacks libuv, so torchrun's rendezvous fails).

CUDA is hidden so a 1-GPU box can run 2 ranks; the real multi-GPU path maps one
rank per GPU (nccl) under torchrun.
"""
import os
# must be set BEFORE torch is imported anywhere
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["USE_LIBUV"] = "0"

import torch.multiprocessing as mp


def worker(rank: int, world_size: int):
    os.environ.update(
        CUDA_VISIBLE_DEVICES="-1", USE_LIBUV="0",
        RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29519",
    )
    from librarian.cli.main import main
    main(["pretrain", "--config", "configs/pretrain_dummy.json"])


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
    print("DDP_SMOKE_OK")
