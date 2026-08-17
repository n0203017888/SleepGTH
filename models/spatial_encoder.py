"""SleepGTH Spatial Encoder (per-patch, CinC 2018 200 Hz setup).

EEG-only: the 6 scalp electrodes are the graph nodes.

Per-patch input shape:  (B, 6, L_p=200)  channels = 6 EEG
Per-patch output shape: (B, d_node=64)

Pipeline:
    EEG  (B, 6, 200) --NodeEncoder (Conv1d x2 + Pool)--> (B, 6, 64)
                     --GTN (2 layers, masked attn, d=64)--> (B, 6, 64)
    readout='global': take the GTN global node            -> (B, 64)
    readout='fusion': Linear(6->1) over the channel axis  -> (B, 64)
"""
from __future__ import annotations

import math

import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
# Adjacency / attention mask
# --------------------------------------------------------------------------- #

def _random_regular_adj(n: int, d: int, seed: int) -> torch.Tensor:
    """Boolean (n, n) adjacency for a random d-regular graph (no self-loops yet).

    Kept for optional graph-structure ablations (random vs anatomical); not used by
    the default model, which uses the anatomical 10-20 adjacency below.
    """
    g = nx.random_regular_graph(d=d, n=n, seed=seed)
    adj = torch.zeros(n, n, dtype=torch.bool)
    for u, v in g.edges():
        adj[u, v] = True
        adj[v, u] = True
    return adj


# Anatomical 10-20 adjacency for the 6 EEG electrodes, in cache order
# [F3(0), F4(1), C3(2), C4(3), O1(4), O2(5)] — a 3x2 grid plus long-range F–O skips,
# giving a 3-regular graph (every electrode has degree 3, diameter 2):
#     F3 — F4        horizontal: within-row L↔R pairs
#     |  X |         vertical:   left chain  F3–C3–O1, right chain F4–C4–O2
#     C3 — C4        long-range: F3–O1, F4–O2 (skip central → diameter drops 3→2)
#     |  X |
#     O1 — O2
_EEG_GRID_EDGES = [(0, 1), (2, 3), (4, 5),          # horizontal (F, C, O rows)
                   (0, 2), (2, 4),                  # left chain  F3–C3–O1
                   (1, 3), (3, 5),                  # right chain F4–C4–O2
                   (0, 4), (1, 5)]                  # long-range frontal–occipital (F3–O1, F4–O2)


def _anatomical_adj(n: int = 6) -> torch.Tensor:
    """Boolean (n, n) adjacency for the fixed 10-20 grid (no self-loops yet)."""
    if n != 6:
        raise ValueError(f'_anatomical_adj is defined for the 6-electrode montage, got n={n}')
    adj = torch.zeros(n, n, dtype=torch.bool)
    for u, v in _EEG_GRID_EDGES:
        adj[u, v] = True
        adj[v, u] = True
    return adj


def build_attention_mask(n_real: int, d_reg: int, seed: int) -> torch.Tensor:
    """(n_real+1, n_real+1) attention mask.

    Real-real: anatomical 10-20 grid adjacency + self-loops.
    Global node (index n_real): star-connected to every node; dropped after the GTN.
    ``d_reg`` / ``seed`` are accepted for signature compatibility but unused by the
    anatomical graph (swap in ``_random_regular_adj`` here for a random-graph ablation).
    """
    n = n_real + 1
    mask = torch.zeros(n, n, dtype=torch.bool)
    mask[:n_real, :n_real] = _anatomical_adj(n_real)
    mask[:n_real, :n_real] |= torch.eye(n_real, dtype=torch.bool)   # self-loops
    mask[n_real, :] = True   # global node attends to all
    mask[:, n_real] = True   # all attend to global node
    return mask


# --------------------------------------------------------------------------- #
# Per-modality encoders
# --------------------------------------------------------------------------- #

class NodeEncoder(nn.Module):
    """Per-EEG-channel CNN encoder: (B, n_nodes, L_p) -> (B, n_nodes, d_out).

    Conv1d(1->hidden, k=kernel1) -> ReLU -> Conv1d(hidden->d_out, k=kernel2) -> ReLU
    -> AdaptiveAvgPool1d(1)   # collapse time axis -> compact node embedding
    """

    def __init__(self, d_out: int = 64, hidden: int = 32, kernel1: int = 19, kernel2: int = 7,
                 n_pool: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden, kernel_size=kernel1, padding=kernel1 // 2)
        self.conv2 = nn.Conv1d(hidden, d_out, kernel_size=kernel2, padding=kernel2 // 2)
        self.pool = nn.AdaptiveAvgPool1d(n_pool)
        self.proj = nn.Linear(d_out * n_pool, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, l = x.shape
        x = x.reshape(b * n, 1, l)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)         # (B*n, d_out * n_pool)
        x = self.proj(x)                    # (B*n, d_out)
        return x.reshape(b, n, -1)          # (B, n, d_out)


class _Branch(nn.Module):
    """Two-conv block. The first conv's kernel size varies per branch
    (large/medium/small receptive field); the second is fixed at k=11.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel1: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel1, padding=kernel1 // 2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=11, padding=5)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self.conv1(x)))
        return F.gelu(self.bn2(self.conv2(h)))


class MultiScaleNodeEncoder(nn.Module):
    """Multi-scale CNN applied to the full-length EEG signal BEFORE patchify.

    Input:  (B, n_eeg, L=6000)
    Output: (B, n_patches=30, n_eeg, d_node=64)

    Three parallel depthwise-style branches (shared weights across the 6 EEG
    channels via reshape) capture different time scales, then a 1x11 integration
    conv fuses them. Patchify is done after the multi-scale CNN so each patch
    sees features extracted from the full 30-second context.
    """

    POOL_TYPES = ('mean_max', 'avgpool8', 'mean')

    def __init__(self, n_eeg: int = 6, l: int = 6000, l_p: int = 200,
                 n_patches: int = 30, d_node: int = 64,
                 hidden: int = 32, n_pool: int = 8, dropout: float = 0.2,
                 pool_type: str = 'mean_max', deep_branch: bool = True):
        super().__init__()
        assert l == n_patches * l_p
        if pool_type not in self.POOL_TYPES:
            raise ValueError(f'pool_type must be one of {self.POOL_TYPES}, got {pool_type!r}')
        self.n_eeg = n_eeg
        self.l = l
        self.l_p = l_p
        self.n_patches = n_patches
        self.d_node = d_node
        self.pool_type = pool_type
        self.deep_branch = deep_branch

        if deep_branch:
            self.branch_a = _Branch(1, hidden, kernel1=25)
            self.branch_b = _Branch(1, hidden, kernel1=101)
            self.branch_c = _Branch(1, hidden, kernel1=201)
            self.shortcut_proj = nn.Conv1d(3 * hidden, d_node, kernel_size=1)
            self.integration_conv = nn.Conv1d(3 * hidden, d_node, kernel_size=11, padding=5)
            self.bn_int = nn.BatchNorm1d(d_node)
        else:
            self.branch_a = nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=25, padding=12),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
            )
            self.branch_b = nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=101, padding=50),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
            )
            self.branch_c = nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=201, padding=100),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
            )
            self.integrate = nn.Sequential(
                nn.Conv1d(3 * hidden, d_node, kernel_size=11, padding=5),
                nn.BatchNorm1d(d_node),
                nn.GELU(),
            )
        self.dropout = nn.Dropout(dropout)
        if pool_type == 'avgpool8':
            self.pool = nn.AdaptiveAvgPool1d(n_pool)
            self.proj = nn.Linear(d_node * n_pool, d_node)
        elif pool_type == 'mean_max':
            self.proj = nn.Linear(d_node * 2, d_node)
        else:  # 'mean'
            self.proj = nn.Linear(d_node, d_node)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = x.reshape(b * self.n_eeg, 1, self.l)
        a = self.branch_a(x)
        m = self.branch_b(x)
        c = self.branch_c(x)
        x_cat = torch.cat([a, m, c], dim=1)                            # (B*n_eeg, 3H, L)
        if self.deep_branch:
            shortcut = self.shortcut_proj(x_cat)                       # projection residual
            h = self.bn_int(self.integration_conv(x_cat))
            x = F.gelu(h + shortcut)                                   # (B*n_eeg, d_node, L)
        else:
            x = self.integrate(x_cat)                                  # (B*n_eeg, d_node, L)
        x = self.dropout(x)
        x = x.reshape(b, self.n_eeg, self.d_node, self.n_patches, self.l_p)
        x = x.permute(0, 3, 1, 2, 4).contiguous()                      # (B, P, n_eeg, d_node, l_p)
        x = x.reshape(b * self.n_patches * self.n_eeg, self.d_node, self.l_p)
        if self.pool_type == 'avgpool8':
            feat = self.pool(x).flatten(1)                             # (B*P*n_eeg, d_node*n_pool)
        elif self.pool_type == 'mean_max':
            mean_f = F.adaptive_avg_pool1d(x, 1).squeeze(-1)           # (B*P*n_eeg, d_node)
            max_f = F.adaptive_max_pool1d(x, 1).squeeze(-1)            # (B*P*n_eeg, d_node)
            feat = torch.cat([mean_f, max_f], dim=-1)                  # (B*P*n_eeg, 2*d_node)
        else:  # 'mean'
            feat = F.adaptive_avg_pool1d(x, 1).squeeze(-1)             # (B*P*n_eeg, d_node)
        x = self.proj(feat)                                            # (B*P*n_eeg, d_node)
        return x.reshape(b, self.n_patches, self.n_eeg, self.d_node)


# --------------------------------------------------------------------------- #
# Graph Transformer Network
# --------------------------------------------------------------------------- #

class MaskedMHA(nn.Module):
    """Multi-head self-attention with a fixed boolean (N, N) connectivity mask."""

    def __init__(self, d_model: int, num_heads: int, d_attn: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_attn = d_attn
        h_total = num_heads * d_attn
        self.q = nn.Linear(d_model, h_total)
        self.k = nn.Linear(d_model, h_total)
        self.v = nn.Linear(d_model, h_total)
        self.out = nn.Linear(h_total, d_model)
        self.scale = 1.0 / math.sqrt(d_attn)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.q(x).view(b, n, self.num_heads, self.d_attn).transpose(1, 2)
        k = self.k(x).view(b, n, self.num_heads, self.d_attn).transpose(1, 2)
        v = self.v(x).view(b, n, self.num_heads, self.d_attn).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, self.num_heads * self.d_attn)
        return self.out(out)


class GTNBlock(nn.Module):
    """Pre-LN graph-transformer block: masked MHA + FFN, both with residual."""

    def __init__(self, d_model: int, num_heads: int, d_attn: int,
                 d_ffn: int | None = None):
        super().__init__()
        d_ffn = d_ffn if d_ffn is not None else 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MaskedMHA(d_model, num_heads, d_attn)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class EEGGraphEncoder(nn.Module):
    """6 real EEG nodes + 1 virtual node.

    NodeEncoder compresses (B, 6, L_p) -> (B, 6, d_node) first,
    then GTN operates on the compact d_node-dim embeddings.
    Virtual node is initialised as the mean of real nodes and dropped after.
    """

    def __init__(self, n_eeg: int = 6, d_reg: int = 4, l_p: int = 200,
                 d_node: int = 64, num_layers: int = 2, num_heads: int = 4,
                 d_attn: int = 64, conv_hidden: int = 32, conv_kernel1: int = 19,
                 conv_kernel2: int = 7, n_pool: int = 8, seed: int = 0,
                 use_gnn: bool = True):
        super().__init__()
        self.n_eeg = n_eeg
        # use_gnn=False -> ablation: skip the GTN graph attention (and virtual node);
        # node embeddings pass straight through to Source Fusion. Blocks/mask not built.
        self.use_gnn = use_gnn
        self.node_enc = NodeEncoder(d_out=d_node, hidden=conv_hidden, kernel1=conv_kernel1,
                                    kernel2=conv_kernel2, n_pool=n_pool)
        if use_gnn:
            self.blocks = nn.ModuleList([
                GTNBlock(d_model=d_node, num_heads=num_heads, d_attn=d_attn)
                for _ in range(num_layers)
            ])
            self.register_buffer('attn_mask', build_attention_mask(n_eeg, d_reg, seed))

    def forward(self, x: torch.Tensor, return_global: bool = False) -> torch.Tensor:
        h = self.node_enc(x)                       # (B, 6, d_node)
        return self.apply_gtn(h, return_global=return_global)

    def apply_gtn(self, h: torch.Tensor, return_global: bool = False) -> torch.Tensor:
        """Run GTN on pre-computed node embeddings (skip the internal NodeEncoder).

        return_global=False: (B, n_eeg, d_node) electrode embeddings (drop global node).
        return_global=True : (B, d_node) the global node, used as the graph-level readout.
        """
        if not self.use_gnn:
            if return_global:
                raise RuntimeError("global-node readout requires use_gnn=True (no global node without GTN)")
            return h                               # ablation: no graph attention
        v = h.mean(dim=1, keepdim=True)            # (B, 1, d_node) — global node init
        h = torch.cat([h, v], dim=1)               # (B, 7, d_node)
        for blk in self.blocks:
            h = blk(h, self.attn_mask)
        if return_global:
            return h[:, self.n_eeg]                # (B, d_node) global node as readout
        return h[:, :self.n_eeg]                   # drop global -> (B, n_eeg, d_node)


# --------------------------------------------------------------------------- #
# Top-level spatial encoder
# --------------------------------------------------------------------------- #

class SpatialEncoder(nn.Module):
    """Per-patch spatial encoder over the EEG electrode graph.

    Input:  (B, n_eeg, L_p=200)
    Output: (B, d_node=64)
    """

    def __init__(self, n_eeg: int = 6,
                 l_p: int = 200, d_node: int = 64, d_reg: int = 4,
                 num_layers: int = 2, num_heads: int = 4, d_attn: int = 16,
                 conv_hidden: int = 32, conv_kernel1: int = 19, conv_kernel2: int = 7,
                 n_pool: int = 8, seed: int = 0, use_gnn: bool = True,
                 readout: str = 'fusion'):
        super().__init__()
        # readout='fusion': Linear(n_eeg->1) static channel fusion over the electrodes.
        # readout='global': use the GTN global node as the patch token (requires use_gnn=True).
        if readout not in ('fusion', 'global'):
            raise ValueError(f"readout must be 'fusion' or 'global', got {readout!r}")
        if readout == 'global' and not use_gnn:
            raise ValueError("readout='global' requires use_gnn=True (no global node without GTN)")
        self.readout = readout
        self.n_eeg = n_eeg
        self.n_channels = n_eeg
        self.eeg = EEGGraphEncoder(
            n_eeg=n_eeg, d_reg=d_reg, l_p=l_p, d_node=d_node,
            num_layers=num_layers, num_heads=num_heads, d_attn=d_attn,
            conv_hidden=conv_hidden, conv_kernel1=conv_kernel1, conv_kernel2=conv_kernel2,
            n_pool=n_pool, seed=seed, use_gnn=use_gnn,
        )
        self.fuse = nn.Linear(self.n_channels, 1) if readout == 'fusion' else None

    def forward(self, x: torch.Tensor, eeg_nodes: torch.Tensor | None = None) -> torch.Tensor:
        """If ``eeg_nodes`` is provided (B*30, n_eeg, d_node), skip the per-patch
        NodeEncoder and feed them straight into the GTN.
        """
        if self.readout == 'global':
            # global node IS the patch token (use_gnn=True enforced at init)
            if eeg_nodes is not None:
                return self.eeg.apply_gtn(eeg_nodes, return_global=True)   # (B, d_node)
            return self.eeg(x[:, :self.n_eeg], return_global=True)         # (B, d_node)

        # 'fusion' readout: GTN-refined electrodes -> Linear(n_eeg->1)
        if eeg_nodes is not None:
            h = self.eeg.apply_gtn(eeg_nodes)    # (B, n_eeg, d_node)
        else:
            h = self.eeg(x[:, :self.n_eeg])      # (B, n_eeg, d_node)
        return self.fuse(h.transpose(1, 2)).squeeze(-1)   # (B, d_node)
