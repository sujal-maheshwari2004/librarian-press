"""perplexity.py — token-level perplexity for both modes.

Handles (x, y) pretrain batches (every token counts) and (x, y, mask) SFT
batches (only completion tokens count) transparently.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_perplexity(model, dataloader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        if len(batch) == 3:
            x, y, mask = batch
            mask = mask.to(device)
        else:
            x, y = batch
            mask = None
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        B, T, V = logits.shape
        loss_per_token = F.cross_entropy(
            logits.reshape(B * T, V), y.reshape(B * T), reduction="none"
        )
        if mask is None:
            total_loss += loss_per_token.sum().item()
            total_tokens += y.numel()
        else:
            mask_flat = mask.reshape(B * T)
            total_loss += (loss_per_token * mask_flat).sum().item()
            total_tokens += mask_flat.sum().item()

    return math.exp(total_loss / max(total_tokens, 1))
