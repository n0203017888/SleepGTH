"""SleepGTH classification loss.

Weighted cross-entropy on epoch-level logits vs epoch labels, with optional
class weighting, label smoothing, and focal loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ClassificationLoss(nn.Module):
    def __init__(self,
                 ignore_index: int = -1,
                 class_weight=None,
                 label_smoothing: float = 0.0,
                 use_focal: bool = False,
                 focal_alpha=None,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        if class_weight is not None:
            self.register_buffer('class_weight',
                                 torch.as_tensor(class_weight, dtype=torch.float32))
        else:
            self.class_weight = None
        if focal_alpha is not None:
            self.register_buffer('focal_alpha',
                                 torch.as_tensor(focal_alpha, dtype=torch.float32))
        else:
            self.focal_alpha = None

    def _focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Multi-class focal loss: FL = -alpha_t * (1 - p_t)^gamma * log(p_t)."""
        mask = labels != self.ignore_index
        if not mask.any():
            return logits.sum() * 0.0
        logits = logits[mask]
        labels = labels[mask]
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_factor = (1.0 - pt).pow(self.focal_gamma)
        loss = -focal_factor * log_pt
        if self.focal_alpha is not None:
            alpha_t = self.focal_alpha[labels]
            loss = alpha_t * loss
        return loss.mean()

    def forward(self, epoch_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Scalar classification loss on epoch-level logits."""
        if self.use_focal:
            return self._focal_loss(epoch_logits, labels)
        return F.cross_entropy(epoch_logits, labels,
                               weight=self.class_weight,
                               ignore_index=self.ignore_index,
                               label_smoothing=self.label_smoothing)
