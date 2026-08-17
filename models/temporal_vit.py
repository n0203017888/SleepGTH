"""SleepGTH Temporal-ViT Encoder with dual classification heads.

Input  : (B, N=30, d_model=64)   — patch tokens from the Source Fusion block.
Output : dict
    'epoch_logits' : (B, 5)       — epoch-level logits (CCE / classification head)
    'epoch_feat'   : (B, 64)      — GAP-pooled epoch embedding

Pipeline:
    x  (B, 30, 64)
      + sinusoidal PE  ──► z0  (B, 30, 64)
      4 × Pre-LN Transformer blocks (4 heads, FFN=256, dropout=0.1, GELU)
                       ──► zL  (B, 30, 64)
    epoch_logits = Linear(64 -> 5)(zL.mean(dim=1))  # GAP over seq -> (B, 64) -> (B, 5)
"""
from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE:
        PE[pos, 2i]   = sin(pos / 10000^(2i/d))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
    Added once to the input; not learned.
    """

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                             * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]


class TemporalViT(nn.Module):
    def __init__(self, d_model: int = 200, n_patches: int = 30,
                 num_layers: int = 4, num_heads: int = 4,
                 d_ffn: int = 800, dropout: float = 0.1,
                 num_classes: int = 5, use_vit: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_patches = n_patches
        self.num_classes = num_classes
        # use_vit=False -> ablation: skip the temporal transformer entirely and just
        # GAP the raw patch tokens. PE + encoder are not built (cleaner param count).
        self.use_vit = use_vit

        if use_vit:
            self.pe = SinusoidalPositionalEncoding(d_model, max_len=max(n_patches, 64))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ffn,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,        # Pre-LN
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.epoch_head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        if self.use_vit:
            z0 = self.pe(x)                      # (B, 30, d_model)
            zL = self.encoder(z0)                # (B, 30, d_model)
        else:
            zL = x                               # ablation: no temporal modeling

        epoch_feat   = zL.mean(dim=1)            # GAP over seq: (B, d_model)
        epoch_logits = self.epoch_head(epoch_feat)  # (B, 5)

        return {'epoch_logits': epoch_logits,
                'epoch_feat': epoch_feat}
