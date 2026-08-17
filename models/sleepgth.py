"""End-to-end SleepGTH model (EEG-only).

Combines patchify -> SpatialEncoder (per-patch) -> TemporalViT (sequence over 30 patches).

Forward accepts a dict from CinC2018EpochDataset:
    {'eeg': (B, 6, 6000), ...}                       # single epoch
    {'eeg': (B, L, 6, 6000), ...}                    # multi-epoch context
A bare tensor of the same shape is also accepted.

Key dimensions:
    l_p     = 200  : raw patch length (samples), used only for patchify
    d_node  = 64   : node embedding dim output by SpatialEncoder -> TemporalViT input

Returns:
    {'epoch_logits': (B, 5)}        # for CCE classification loss
"""
from __future__ import annotations

import torch
from torch import nn

from .spatial_encoder import MultiScaleNodeEncoder, SpatialEncoder
from .temporal_vit import SinusoidalPositionalEncoding, TemporalViT


class SequenceTransformer(nn.Module):
    """Cross-epoch context: takes (B, L, d_model) and outputs (B, L, d_model).

    2-layer Pre-LN Transformer with sinusoidal PE over the L positions.
    """

    def __init__(self, d_model: int = 64, n_pos: int = 5, num_layers: int = 2,
                 num_heads: int = 4, d_ffn: int = 128, dropout: float = 0.1):
        super().__init__()
        self.pe = SinusoidalPositionalEncoding(d_model, max_len=max(n_pos, 16))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.pe(x))


class SleepGTH(nn.Module):
    def __init__(self,
                 n_eeg: int = 6,
                 l_p: int = 200, n_patches: int = 30, num_classes: int = 5,
                 # spatial config
                 d_node: int = 64,
                 d_reg: int = 4, sp_layers: int = 2, sp_heads: int = 4,
                 sp_d_attn: int = 64, conv_hidden: int = 32, conv_kernel1: int = 19,
                 conv_kernel2: int = 7, n_pool: int = 8, graph_seed: int = 0,
                 use_gnn: bool = True, use_vit: bool = True, readout: str = 'fusion',
                 use_multiscale_eeg: bool = True, ms_dropout: float = 0.2,
                 pool_type: str = 'mean_max', deep_branch: bool = True,
                 # temporal config
                 vit_layers: int = 4, vit_heads: int = 4,
                 vit_d_ffn: int = 256, vit_dropout: float = 0.1,
                 # multi-epoch context
                 context_size: int = 1, seq_layers: int = 2, seq_heads: int = 4,
                 seq_d_ffn: int = 128, seq_dropout: float = 0.1):
        super().__init__()
        self.n_eeg = n_eeg
        self.n_channels = n_eeg
        self.n_patches = n_patches
        self.l_p = l_p

        self.use_multiscale_eeg = use_multiscale_eeg
        if use_multiscale_eeg:
            self.multiscale = MultiScaleNodeEncoder(
                n_eeg=n_eeg, l=n_patches * l_p, l_p=l_p, n_patches=n_patches,
                d_node=d_node, hidden=conv_hidden, n_pool=n_pool, dropout=ms_dropout,
                pool_type=pool_type, deep_branch=deep_branch,
            )
        self.spatial = SpatialEncoder(
            n_eeg=n_eeg, l_p=l_p, d_node=d_node, d_reg=d_reg,
            num_layers=sp_layers, num_heads=sp_heads, d_attn=sp_d_attn,
            conv_hidden=conv_hidden, conv_kernel1=conv_kernel1, conv_kernel2=conv_kernel2,
            n_pool=n_pool, seed=graph_seed, use_gnn=use_gnn, readout=readout,
        )
        self.temporal = TemporalViT(
            d_model=d_node, n_patches=n_patches,
            num_layers=vit_layers, num_heads=vit_heads,
            d_ffn=vit_d_ffn, dropout=vit_dropout, num_classes=num_classes,
            use_vit=use_vit,
        )

        self.context_size = context_size
        if context_size > 1:
            self.sequence = SequenceTransformer(
                d_model=d_node, n_pos=context_size,
                num_layers=seq_layers, num_heads=seq_heads,
                d_ffn=seq_d_ffn, dropout=seq_dropout,
            )

    @staticmethod
    def _as_tensor(x) -> torch.Tensor:
        return x['eeg'] if isinstance(x, dict) else x

    def _process_epochs(self, x: torch.Tensor) -> dict:
        """Process (B, n_eeg, 6000) through MS-CNN + SpatialEncoder + TemporalViT."""
        b = x.shape[0]
        eeg_nodes = None
        if self.use_multiscale_eeg:
            eeg_nodes = self.multiscale(x[:, :self.n_eeg])           # (B, 30, n_eeg, d_node)
            eeg_nodes = eeg_nodes.reshape(b * self.n_patches, self.n_eeg, -1)
        patches = (x.reshape(b, self.n_channels, self.n_patches, self.l_p)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .reshape(b * self.n_patches, self.n_channels, self.l_p))
        z = self.spatial(patches, eeg_nodes=eeg_nodes).reshape(b, self.n_patches, -1)  # (B, 30, 64)
        return self.temporal(z)

    def forward(self, x) -> dict:
        x = self._as_tensor(x)                                       # (B, C, 6000) or (B, L, C, 6000)

        if x.dim() == 3:
            # Single-epoch path
            out = self._process_epochs(x)
            return {'epoch_logits': out['epoch_logits']}

        # Multi-epoch context path: (B, L, C, 6000)
        b, l = x.shape[0], x.shape[1]
        x_flat = x.reshape(b * l, *x.shape[2:])                      # (B*L, C, 6000)

        out = self._process_epochs(x_flat)                           # dict with (B*L, ...) shapes

        feat = out['epoch_feat'].reshape(b, l, -1)                   # (B, L, d_node)
        feat = self.sequence(feat)                                   # (B, L, d_node)
        center = l // 2
        center_feat = feat[:, center]                                # (B, d_node)
        epoch_logits = self.temporal.epoch_head(center_feat)         # (B, 5)
        return {'epoch_logits': epoch_logits}
