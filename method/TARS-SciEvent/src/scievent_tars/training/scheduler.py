"""Linear warmup + linear decay, applied uniformly to every parameter group."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: Optimizer, total_steps: int, warmup_ratio: float = 0.1
) -> LambdaLR:
    warmup_steps = max(1, int(math.ceil(total_steps * warmup_ratio)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)
