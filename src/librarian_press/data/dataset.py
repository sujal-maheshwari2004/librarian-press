"""
dataset.py — memmap datasets for both modes.

  PackedDataset   : pretraining, yields (x, y)            from a packed .bin
  FinetuneDataset : SFT,         yields (x, y, mask)       from .bin + _mask.bin

Both are uint16 on disk, windowed identically (one extra token for the target
shift). The trainer treats a 2-tuple as unmasked and a 3-tuple as masked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedDataset(Dataset):
    def __init__(self, path, seq_len):
        self.seq_len = seq_len
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.num_sequences = max(0, len(self.data) // seq_len - 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        tokens = self.data[start:end].astype(np.int64)
        x = torch.from_numpy(tokens[:-1])
        y = torch.from_numpy(tokens[1:])
        return x, y


class FinetuneDataset(Dataset):
    def __init__(self, data_dir: Path, split_name: str, seq_len: int):
        self.seq_len = seq_len
        data_dir = Path(data_dir)
        tok_path = data_dir / f"{split_name}.bin"
        mask_path = data_dir / f"{split_name}_mask.bin"

        if not tok_path.exists():
            raise FileNotFoundError(
                f"Token file not found: {tok_path}\nRun the prepare_sft stage first."
            )

        self.tokens = np.memmap(tok_path, dtype=np.uint16, mode="r")
        self.masks = np.memmap(mask_path, dtype=np.uint16, mode="r")
        self.num_sequences = max(0, len(self.tokens) // seq_len - 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        tokens = self.tokens[start:end].astype(np.int64)
        masks = self.masks[start:end].astype(np.float32)
        x = torch.from_numpy(tokens[:-1])
        y = torch.from_numpy(tokens[1:])
        mask = torch.from_numpy(masks[1:])   # aligns with the target
        return x, y, mask
