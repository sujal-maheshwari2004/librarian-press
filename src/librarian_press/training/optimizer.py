import torch


def build_optimizer(model, config):
    """AdamW with decoupled weight decay: decay 2D+ params, not biases/norms.
    Only parameters with requires_grad are included (LoRA/BitFit safe)."""

    decay = []
    no_decay = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    optim_groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.lr,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    return optimizer
