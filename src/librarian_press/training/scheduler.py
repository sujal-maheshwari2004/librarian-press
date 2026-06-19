import math


def cosine_lr(step, config) -> float:
    """Linear warmup then cosine decay to min_lr. Returns a multiplier on config.lr."""

    if step < config.warmup_steps:
        return step / max(config.warmup_steps, 1)

    progress = (step - config.warmup_steps) / max(
        config.total_steps - config.warmup_steps, 1
    )

    cosine = 0.5 * (1 + math.cos(math.pi * progress))

    return config.min_lr / config.lr + cosine * (1 - config.min_lr / config.lr)
