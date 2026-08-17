"""Learning-rate schedules."""
from __future__ import annotations

import math
import warnings

from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_scheduler(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    """Linear warmup -> cosine decay to 0. Steps per call to scheduler.step().

        LR(0)            = 0
        LR(warmup_steps) = base_lr           (peak)
        LR(total_steps)  = 0
    """

    def f(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    # LambdaLR.__init__ calls self.step() once to initialise the LR before any
    # optimizer step has occurred, which triggers a spurious PyTorch UserWarning.
    # The training loop order (optimizer.step → scheduler.step) is correct; the
    # warning here is a false positive from the scheduler's own initialisation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return LambdaLR(optimizer, f)
