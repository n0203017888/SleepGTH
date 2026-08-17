"""End-to-end SleepGTH model.

Combines patchify -> SpatialEncoder (per-patch) -> TemporalViT (sequence over 30 patches).

Forward accepts a dict from CinC2018EpochDataset:
    {'eeg': (B, 6, 6000), 'eog': (B, 1, 6000), 'emg': (B, 1, 6000), ...}

Use ``use_eog`` / ``use_emg`` to run ablations with fewer modalities.

Key dimensions:
    l_p     = 200  : raw patch length (samples), used only for patchify
    d_node  = 64   : node embedding dim output by SpatialEncoder -> TemporalViT input

Returns:
    {'epoch_logits': (B, 5)}        # for CCE classification loss
"""
from __future__ import annotations

import torch
from torch import nn

from .spatial_encoder import EOGEncoder, MultiScaleNodeEncoder, SpatialEncoder
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
                 n_eeg: int = 6, n_eog: int = 1, n_emg: int = 1,
                 use_eog: bool = True, use_emg: bool = True,
                 use_eog_encoder: bool = False, eog_only: bool = False,
                 use_emg_encoder: bool = False,
                 fusion: str = 'concat', eog_dim: int = 64, emg_dim: int = 16,
                 eog_kernels: tuple[int, int] = (11, 201),
                 emg_kernels: tuple[int, int] = (11, 201),
                 fusion_level: str = 'epoch',
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
        self.n_eog = n_eog if use_eog else 0
        self.n_emg = n_emg if use_emg else 0
        self.use_eog = use_eog
        self.use_emg = use_emg
        self.n_channels = self.n_eeg + self.n_eog + self.n_emg
        self.n_patches = n_patches
        self.l_p = l_p

        # EOG-only diagnostic mode: bypass the entire EEG pipeline, classify from
        # EOGEncoder features alone. Used to check how much signal EOG carries on its own.
        self.eog_only = eog_only

        # Skip building EEG modules (multiscale / spatial / temporal) when eog_only.
        self.use_multiscale_eeg = use_multiscale_eeg and not eog_only
        if not eog_only:
            if use_multiscale_eeg:
                self.multiscale = MultiScaleNodeEncoder(
                    n_eeg=n_eeg, l=n_patches * l_p, l_p=l_p, n_patches=n_patches,
                    d_node=d_node, hidden=conv_hidden, n_pool=n_pool, dropout=ms_dropout,
                    pool_type=pool_type, deep_branch=deep_branch,
                )
            self.spatial = SpatialEncoder(
                n_eeg=n_eeg, n_eog=self.n_eog, n_emg=self.n_emg,
                l_p=l_p, d_node=d_node, d_reg=d_reg,
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

        # Auxiliary 1-channel encoders (EOG / EMG) + fusion with EEG.
        #   fusion_level='epoch': fuse AFTER TemporalViT (aux=1 epoch vector). concat / hwgate.
        #   fusion_level='patch': fuse BEFORE TemporalViT (aux=30 patch tokens, time-aligned),
        #                         so aux signals also go through temporal modeling.
        # EMG (chin) mirrors EOG: same 1-ch EOGEncoder, fused the same way. EMG carries muscle
        # tone (REM atonia / Wake high tone), a summary-level feature -> small emg_dim + epoch fuse.
        self.use_eog_encoder = use_eog_encoder and not eog_only
        self.use_emg_encoder = use_emg_encoder and not eog_only
        self.fusion_type = fusion
        self.fusion_level = fusion_level
        if self.use_eog_encoder or self.use_emg_encoder:
            if fusion not in ('concat', 'hwgate'):
                raise ValueError(f"fusion must be 'concat' or 'hwgate', got {fusion!r}")
            if fusion_level not in ('epoch', 'patch'):
                raise ValueError(f"fusion_level must be 'epoch' or 'patch', got {fusion_level!r}")
            # n_patches>0 makes the aux encoders emit 30 tokens (patch level); 0 = one vector (epoch)
            aux_np = n_patches if fusion_level == 'patch' else 0
            aux_dim = 0
            if self.use_eog_encoder:
                self.eog_encoder = EOGEncoder(out_dim=eog_dim, kernels=eog_kernels,
                                              n_patches=aux_np, l_p=l_p)
                aux_dim += eog_dim
            if self.use_emg_encoder:
                self.emg_encoder = EOGEncoder(out_dim=emg_dim, kernels=emg_kernels,
                                              n_patches=aux_np, l_p=l_p)
                aux_dim += emg_dim
            cat_dim = d_node + aux_dim                               # eeg(64) + eog_dim + emg_dim
            if fusion_level == 'patch':
                # aux produce 30 tokens; fuse per-patch then feed fused tokens to TemporalViT
                self.patch_fusion = nn.Sequential(
                    nn.Linear(cat_dim, d_node),                     # (B,30,cat_dim) -> (B,30,64)
                    nn.GELU(),
                )
            elif fusion == 'concat':
                # concat: fused = GELU(Linear(cat(eeg, aux...)))
                self.fusion = nn.Sequential(
                    nn.Linear(cat_dim, d_node),                     # (B,cat_dim) -> (B,64)
                    nn.GELU(),
                )
            else:  # 'hwgate'
                # Highway gate: slide per-dim between pure EEG (safe fallback) and a learned mix.
                #   h = GELU(Linear(cat));  gate = sigmoid(Linear(cat));  fused = gate*h + (1-gate)*eeg
                self.fusion_mix  = nn.Sequential(
                    nn.Linear(cat_dim, d_node),
                    nn.GELU(),
                )
                self.fusion_gate = nn.Linear(cat_dim, d_node)
                nn.init.constant_(self.fusion_gate.bias, -1.0)      # sigmoid(-1)~0.27: start near EEG-only

        if eog_only:
            # eog_only feeds SequenceTransformer (d_model=d_node), so output d_node (not eog_dim);
            # eog_dim is a fusion concept and doesn't apply here. Kernels still configurable.
            self.eog_encoder = EOGEncoder(out_dim=d_node, kernels=eog_kernels)   # (B,1,6000)->(B,d_node)
            self.eog_head = nn.Linear(d_node, num_classes)          # (B,d_node) -> (B,5)

    def _stack(self, x) -> torch.Tensor:
        if isinstance(x, dict):
            parts = [x['eeg']]
            if self.use_eog:
                parts.append(x['eog'])
            if self.use_emg:
                parts.append(x['emg'])
            return torch.cat(parts, dim=-2)                         # cat over channel dim (works for both 3-D and 4-D)
        return x

    def _process_epochs(self, x: torch.Tensor, eog: torch.Tensor | None = None,
                        emg: torch.Tensor | None = None) -> dict:
        """Process (B, C, 6000) through MS-CNN + SpatialEncoder + TemporalViT.

        Auxiliary 1-ch signals (eog, emg) are each encoded by their own EOGEncoder and
        fused with the EEG features — at patch level (before ViT) or epoch level (after ViT).
        """
        b = x.shape[0]
        has_aux = self.use_eog_encoder or self.use_emg_encoder
        eeg_nodes = None
        if self.use_multiscale_eeg:
            eeg_nodes = self.multiscale(x[:, :self.n_eeg])           # (B, 30, n_eeg, d_node)
            eeg_nodes = eeg_nodes.reshape(b * self.n_patches, self.n_eeg, -1)
        patches = (x.reshape(b, self.n_channels, self.n_patches, self.l_p)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .reshape(b * self.n_patches, self.n_channels, self.l_p))
        z = self.spatial(patches, eeg_nodes=eeg_nodes).reshape(b, self.n_patches, -1)  # (B, 30, 64) EEG tokens

        # --- patch-level fusion: fuse aux tokens into the 30 EEG tokens BEFORE TemporalViT ---
        if has_aux and self.fusion_level == 'patch':
            tok = [z]                                                # (B, 30, 64)
            if self.use_eog_encoder and eog is not None:
                tok.append(self.eog_encoder(eog))                   # (B, 30, eog_dim) time-aligned
            if self.use_emg_encoder and emg is not None:
                tok.append(self.emg_encoder(emg))                   # (B, 30, emg_dim)
            z = self.patch_fusion(torch.cat(tok, dim=-1))           # (B, 30, cat_dim) -> (B, 30, 64)

        out = self.temporal(z)

        # --- epoch-level fusion: fuse aux into the single epoch vector AFTER TemporalViT ---
        if has_aux and self.fusion_level == 'epoch':
            eeg_feat = out['epoch_feat']                            # (B, 64)
            parts = [eeg_feat]
            if self.use_eog_encoder and eog is not None:
                parts.append(self.eog_encoder(eog))                 # (B, eog_dim)
            if self.use_emg_encoder and emg is not None:
                parts.append(self.emg_encoder(emg))                 # (B, emg_dim)
            cat = torch.cat(parts, dim=-1)                          # (B, 64 + eog_dim + emg_dim)
            if self.fusion_type == 'concat':
                fused = self.fusion(cat)                            # (B, 64)
            else:  # 'hwgate'
                h    = self.fusion_mix(cat)                         # (B, 64) aligned mix
                gate = torch.sigmoid(self.fusion_gate(cat))         # (B, 64) per-dim 0..1
                fused = gate * h + (1.0 - gate) * eeg_feat          # pure EEG <-> rich mix
            # Recompute epoch_logits from the fused epoch embedding
            epoch_logits = self.temporal.epoch_head(fused)          # (B, 5)
            out = {**out, 'epoch_feat': fused, 'epoch_logits': epoch_logits}

        return out

    def _forward_eog_only(self, x) -> dict:
        """EOG-only diagnostic path: EOGEncoder (+ SequenceTransformer if ctx>1) -> head."""
        eog = x['eog'] if isinstance(x, dict) else x        # (B,1,6000) or (B,L,1,6000)

        if eog.dim() == 3:
            # Single-epoch path: (B, 1, 6000)
            feat = self.eog_encoder(eog)                    # (B, 64)
            epoch_logits = self.eog_head(feat)              # (B, 5)
            return {'epoch_logits': epoch_logits}

        # Multi-epoch context path: (B, L, 1, 6000)
        b, l = eog.shape[0], eog.shape[1]
        eog_flat = eog.reshape(b * l, 1, eog.shape[-1])     # (B*L, 1, 6000)
        feat = self.eog_encoder(eog_flat).reshape(b, l, -1) # (B, L, 64)
        feat = self.sequence(feat)                          # (B, L, 64)
        center = l // 2
        epoch_logits = self.eog_head(feat[:, center])       # (B, 5)
        return {'epoch_logits': epoch_logits}

    def forward(self, x) -> dict:
        if self.eog_only:
            return self._forward_eog_only(x)

        # Extract aux signals (EOG/EMG) before _stack merges the dict into a channel tensor
        eog = x.get('eog') if (isinstance(x, dict) and self.use_eog_encoder) else None  # (B,1,6000)/(B,L,1,6000)
        emg = x.get('emg') if (isinstance(x, dict) and self.use_emg_encoder) else None

        x = self._stack(x)                                           # (B, C, 6000) or (B, L, C, 6000)

        if x.dim() == 3:
            # Single-epoch path
            out = self._process_epochs(x, eog, emg)                 # aux: (B, 1, 6000) or None
            return {'epoch_logits': out['epoch_logits']}

        # Multi-epoch context path: (B, L, C, 6000)
        b, l = x.shape[0], x.shape[1]
        x_flat = x.reshape(b * l, *x.shape[2:])                      # (B*L, C, 6000)

        # aux: (B, L, 1, 6000) -> (B*L, 1, 6000) for per-epoch encoding
        eog_flat = eog.reshape(b * l, 1, eog.shape[-1]) if eog is not None else None
        emg_flat = emg.reshape(b * l, 1, emg.shape[-1]) if emg is not None else None

        out = self._process_epochs(x_flat, eog_flat, emg_flat)       # dict with (B*L, ...) shapes

        feat = out['epoch_feat'].reshape(b, l, -1)                   # (B, L, d_node)
        feat = self.sequence(feat)                                   # (B, L, d_node)
        center = l // 2
        center_feat = feat[:, center]                                # (B, d_node)
        epoch_logits = self.temporal.epoch_head(center_feat)         # (B, 5)
        return {'epoch_logits': epoch_logits}
