"""Visualize SleepGTH internal attention for ONE representative epoch per stage.

For each sleep stage (W, N1, N2, N3, R) the script picks the highest-confidence
*correctly classified* epoch from a small random subject pool, then produces:

  1. GTN spatial edge weights  -> figures/attn_gtn.png
       The 7x7 masked-attention matrix (6 electrodes + 1 global node) inside the
       Graph-Transformer, averaged over the center epoch's 30 patches, the 2 GTN
       layers and the 4 heads. Electrode-electrode edges show how strongly the two
       electrodes attend to each other; node color = how much the GLOBAL readout
       node attends to that electrode (i.e. the electrode the decision leans on).

  2. Temporal importance over the 30 patches (= 30 seconds) -> figures/attn_temporal.png
       (A) Gradient saliency : ||d logit_pred / d patch_token_t||   (drives the decision)
       (B) Attention rollout : flow of attention across the 4 TemporalViT layers

Nothing in the model is modified on disk: attention is captured at runtime by
monkey-patching the relevant forward() methods.

Run (matching your best EEG-only ctx=7 checkpoint):
    python scripts/visualize_attention.py \
        --cache-dir ./cache_dataset \
        --ckpt ./ckpt_eeg_full/best.pt \
        --context-size 7 --n-subjects 20 --seed 11 \
        --exclude-splits ./ckpt_eeg_full/splits.json
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# NOTE: torch / dataset / model imports are done lazily inside the functions that
# need them, so the --replot-from path (numpy + matplotlib only) never touches
# torch — it stays usable even when torch's DLL load intermittently fails.

STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'R']         # internal codes / npz keys (keep 'R')
DISPLAY_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']     # labels shown on the figures
ELECTRODES = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']
# anatomical 10-20 grid edges (electrode-electrode), same as build_attention_mask
GRID_EDGES = [(0, 1), (2, 3), (4, 5), (0, 2), (2, 4), (1, 3), (3, 5), (0, 4), (1, 5)]
# 2-D layout for drawing the electrode graph
POS = {0: (0.0, 2.0), 1: (1.0, 2.0),       # F3 F4
       2: (0.0, 1.0), 3: (1.0, 1.0),       # C3 C4
       4: (0.0, 0.0), 5: (1.0, 0.0),       # O1 O2
       6: (-1.1, 1.0)}                      # global node 'G'

# Hexagon layout for the edges figure: C3/C4 pushed outward so the long-range
# F3-O1 / F4-O2 edges can bow out on the sides instead of hiding behind the
# vertical F-C-O chains. Value = arc curvature (sign = bulge direction).
POS_HEX = {0: (0.0, 2.0), 1: (1.0, 2.0),
           2: (-0.4, 1.0), 3: (1.4, 1.0),
           4: (0.0, 0.0), 5: (1.0, 0.0)}
LONG_RANGE_RAD = {(0, 4): 1.05, (1, 5): -1.05}  # F3-O1 bows out left, F4-O2 bows out right

# Radial star layout for the global-node figure: G at the center, 6 electrodes
# arranged around it (F3/F4 top, C3/C4 sides, O1/O2 bottom).
POS_STAR = {0: (-0.5, 0.87), 1: (0.5, 0.87),    # F3 F4
            2: (-1.0, 0.0), 3: (1.0, 0.0),      # C3 C4
            4: (-0.5, -0.87), 5: (0.5, -0.87),  # O1 O2
            6: (0.0, 0.0)}                        # global node 'G' (center)


# --------------------------------------------------------------------------- #
# Runtime attention capture (monkey-patching, no source edits)
# --------------------------------------------------------------------------- #

def _masked_mha_capture(self, x, mask):
    """Re-implementation of MaskedMHA.forward that stores the softmax attention."""
    import torch
    b, n, _ = x.shape
    q = self.q(x).view(b, n, self.num_heads, self.d_attn).transpose(1, 2)
    k = self.k(x).view(b, n, self.num_heads, self.d_attn).transpose(1, 2)
    v = self.v(x).view(b, n, self.num_heads, self.d_attn).transpose(1, 2)
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    self._cap_attn = attn.detach()                       # (B, heads, 7, 7)
    out = (attn @ v).transpose(1, 2).reshape(b, n, self.num_heads * self.d_attn)
    return self.out(out)


def _vit_layer_capture(layer, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
    """Pre-LN TransformerEncoderLayer forward that captures per-head attention.

    Reimplemented explicitly so we never hit PyTorch's fused fast-path (which would
    skip self_attn and give us no weights). Matches norm_first=True, GELU.
    """
    x = src
    a, w = layer.self_attn(layer.norm1(x), layer.norm1(x), layer.norm1(x),
                           need_weights=True, average_attn_weights=False)
    layer._cap_attn = w.detach()                          # (B, heads, 30, 30)
    x = x + layer.dropout1(a)
    y = layer.norm2(x)
    y = layer.linear2(layer.dropout(layer.activation(layer.linear1(y))))
    x = x + layer.dropout2(y)
    return x


def patch_for_capture(model):
    """Install capture hooks; return (gtn_mhas, vit_layers)."""
    from models.spatial_encoder import MaskedMHA
    gtn_mhas, vit_layers = [], []
    for m in model.modules():
        if isinstance(m, MaskedMHA):
            m.forward = types.MethodType(_masked_mha_capture, m)
            gtn_mhas.append(m)
    if getattr(model.temporal, 'use_vit', False):
        for layer in model.temporal.encoder.layers:
            layer.forward = types.MethodType(_vit_layer_capture, layer)
            vit_layers.append(layer)
    return gtn_mhas, vit_layers


# --------------------------------------------------------------------------- #
# Model build (auto-detect config from checkpoint keys)
# --------------------------------------------------------------------------- #

def build_model(ckpt_path, context_size, device):
    import torch
    from models.sleepgth import SleepGTH
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt['model']
    keys = list(sd.keys())
    use_gnn = any(k.startswith('spatial.eeg.blocks') for k in keys)
    use_vit = any(k.startswith('temporal.encoder') for k in keys)
    readout = 'fusion' if any('spatial.fuse' in k for k in keys) else 'global'
    has_seq = any(k.startswith('sequence') for k in keys)
    print(f'[config] use_gnn={use_gnn} use_vit={use_vit} readout={readout} '
          f'sequence={has_seq} (context_size={context_size})')
    if has_seq and context_size == 1:
        print('  [warn] checkpoint has a SequenceTransformer but context_size=1; '
              'pass --context-size 7 to match the Full model.')
    model = SleepGTH(use_gnn=use_gnn, use_vit=use_vit, readout=readout,
                     context_size=context_size).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f'[load_state_dict] missing={missing}  unexpected={unexpected}')
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Pass 1: pick one high-confidence correct epoch per stage
# --------------------------------------------------------------------------- #

def pick_examples(model, ds, device, n_per_stage, batch_size=32):
    """Return {stage_idx: [dataset_index, ...]} — the top-N highest-confidence
    correctly-classified epochs per stage (for multi-epoch averaging)."""
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    cand = {c: [] for c in range(5)}                       # (conf, dataset_index)
    seen = 0
    with torch.no_grad():
        for batch in loader:
            eeg = batch['eeg'].to(device)
            labels = batch['label']
            logits = model({'eeg': eeg})['epoch_logits']
            prob = torch.softmax(logits, dim=-1).cpu()
            pred = prob.argmax(-1)
            conf = prob.max(-1).values
            for b in range(eeg.shape[0]):
                y = int(labels[b])
                if pred[b] == y:
                    cand[y].append((float(conf[b]), seen + b))
            seen += eeg.shape[0]
    picks = {}
    for c in range(5):
        cand[c].sort(reverse=True)
        picks[c] = [gi for _, gi in cand[c][:n_per_stage]]
        if picks[c]:
            hi = cand[c][0][0]; lo = cand[c][len(picks[c]) - 1][0]
            print(f'  {STAGE_NAMES[c]}: {len(picks[c])} epochs (conf {hi:.3f}..{lo:.3f})')
        else:
            print(f'  {STAGE_NAMES[c]}: NONE FOUND')
    return picks


# --------------------------------------------------------------------------- #
# Pass 2: capture attention for one item
# --------------------------------------------------------------------------- #

def analyze_item(model, ds, gi, context_size, gtn_mhas, vit_layers, device,
                 n_patches=30):
    """Run one epoch with capture; return dict of GTN matrix + temporal importances."""
    import torch
    item = ds[gi]
    eeg = item['eeg'].unsqueeze(0).to(device)              # (1, L, 6, 6000) or (1, 6, 6000)
    center = context_size // 2

    # forward with grad (for saliency)
    captured = {}
    def pre_hook(module, args):
        z = args[0]
        z.retain_grad()
        captured['z'] = z
        return None
    h = model.temporal.register_forward_pre_hook(pre_hook)

    model.zero_grad(set_to_none=True)
    out = model({'eeg': eeg})
    logits = out['epoch_logits']                           # (1, 5)
    pred = int(logits.argmax(-1))

    # --- (A) gradient saliency over the center epoch's 30 patch tokens ---
    logits[0, pred].backward()
    z = captured['z']                                      # (L, 30, 64), L=context_size
    grad = z.grad                                          # (L, 30, 64)
    sal = grad[center].norm(dim=-1).detach().cpu().numpy() # (30,)
    h.remove()

    # --- GTN edge weights: center epoch's 30 patches, avg layers+heads ---
    rows = slice(center * n_patches, center * n_patches + n_patches)
    mats = []
    for mha in gtn_mhas:
        a = mha._cap_attn[rows]                            # (30, heads, 7, 7)
        mats.append(a.mean(dim=(0, 1)).cpu().numpy())      # (7, 7)
    gtn = np.mean(mats, axis=0) if mats else None          # (7, 7)

    # --- (B) attention rollout over the 4 ViT layers (center epoch) ---
    rollout = None
    if vit_layers:
        R = np.eye(n_patches)
        for layer in vit_layers:
            A = layer._cap_attn[center].mean(dim=0).cpu().numpy()   # (30, 30) avg heads
            A = 0.5 * A + 0.5 * np.eye(n_patches)                   # residual
            A = A / A.sum(axis=-1, keepdims=True)
            R = A @ R
        rollout = R.mean(axis=0)                            # importance of each token

    return {'pred': pred, 'gtn': gtn, 'saliency': sal, 'rollout': rollout}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _norm(v):
    v = np.asarray(v, dtype=float)
    rng = v.max() - v.min()
    return (v - v.min()) / rng if rng > 1e-12 else np.zeros_like(v)


def plot_gtn_edges(results, out_path):
    """Figure 1: importance of the connections BETWEEN the 6 electrodes."""
    fig, axes = plt.subplots(1, 5, figsize=(13, 4.4), gridspec_kw={'wspace': 0.15})
    cmap = plt.cm.plasma
    for ax, c in zip(axes, range(5)):
        r = results.get(c)
        ax.set_title(DISPLAY_NAMES[c]); ax.axis('off')
        ax.set_xlim(-1.4, 2.4); ax.set_ylim(-0.6, 2.6); ax.set_aspect('equal')
        if r is None or r['gtn'] is None:
            ax.text(0.5, 1, 'n/a', ha='center'); continue
        A = r['gtn']                                       # (7,7)
        ew = np.array([0.5 * (A[u, v] + A[v, u]) for u, v in GRID_EDGES])
        ewn = _norm(ew)                                    # per-stage min-max
        for (u, v), w in zip(GRID_EDGES, ewn):
            col = cmap(w); lw = 4.0                         # uniform width; color = attention
            if (u, v) in LONG_RANGE_RAD:                   # F3-O1 / F4-O2 as side arcs
                p = FancyArrowPatch(POS_HEX[u], POS_HEX[v], arrowstyle='-',
                                    connectionstyle=f'arc3,rad={LONG_RANGE_RAD[(u, v)]}',
                                    color=col, lw=lw, zorder=1)
                ax.add_patch(p)
            else:
                x0, y0 = POS_HEX[u]; x1, y1 = POS_HEX[v]
                ax.plot([x0, x1], [y0, y1], color=col, lw=lw, zorder=1)
        for e in range(6):
            x, y = POS_HEX[e]
            ax.scatter([x], [y], s=500, c='lightgray', edgecolors='k', zorder=2)
            ax.text(x, y, ELECTRODES[e], ha='center', va='center', fontsize=7, zorder=3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); sm.set_array([])
    fig.colorbar(sm, ax=list(axes.ravel()), fraction=0.012,
                 label='electrode-electrode attention (per-stage min-max)')
    fig.suptitle('GTN — connection importance between the 6 electrodes '
                 '(edge color = mutual attention)', y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {out_path}', flush=True)


def plot_gtn_global(results, out_path):
    """Figure 2: importance between each electrode and the global readout node G."""
    fig, axes = plt.subplots(1, 5, figsize=(13, 4.4), gridspec_kw={'wspace': 0.05})
    cmap = plt.cm.viridis
    for ax, c in zip(axes, range(5)):
        r = results.get(c)
        ax.set_title(DISPLAY_NAMES[c]); ax.axis('off')
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.4, 1.4); ax.set_aspect('equal')
        if r is None or r['gtn'] is None:
            ax.text(0, 0, 'n/a', ha='center'); continue
        A = r['gtn']
        gw = A[6, :6]; gwn = _norm(gw)                     # global -> each electrode
        for e in range(6):                                 # solid spokes G -> electrode
            x0, y0 = POS_STAR[6]; x1, y1 = POS_STAR[e]
            ax.plot([x0, x1], [y0, y1], color=cmap(gwn[e]), lw=4.0, zorder=1)
        for e in range(6):
            x, y = POS_STAR[e]
            ax.scatter([x], [y], s=900, c='lightgray', edgecolors='k', zorder=2)
            ax.text(x, y, ELECTRODES[e], ha='center', va='center',
                    color='k', fontsize=9, zorder=3)
        gx, gy = POS_STAR[6]
        ax.scatter([gx], [gy], s=1100, c='gold', edgecolors='k', zorder=2)
        ax.text(gx, gy, 'G', ha='center', va='center', fontsize=11, zorder=3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); sm.set_array([])
    fig.colorbar(sm, ax=list(axes.ravel()), fraction=0.012,
                 label='global-readout attention (per-stage min-max)')
    fig.suptitle('GTN — importance between each electrode and the global node G '
                 '(node & edge color = how much G aggregates that electrode)', y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved -> {out_path}', flush=True)


def plot_temporal(results, out_path):
    """TemporalViT-only view: attention rollout across the 4 ViT layers."""
    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
    secs = np.arange(30)
    for ax, c in zip(axes, range(5)):
        r = results.get(c)
        ax.set_ylabel(f'{DISPLAY_NAMES[c]}\nattention (rollout, norm.)')
        if r is None or r['rollout'] is None:
            ax.text(15, 0.5, 'n/a', ha='center'); continue
        ax.bar(secs, _norm(r['rollout']), color='steelblue', width=0.85)
        ax.set_ylim(0, 1.05)
    axes[-1].set_xlabel('time within 30-s epoch (s)')
    fig.suptitle('TemporalViT attention rollout over the 30 patches (= 30 seconds)', y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'saved -> {out_path}', flush=True)


# --------------------------------------------------------------------------- #
# Save / load raw attention + figure driver (so plotting can run standalone)
# --------------------------------------------------------------------------- #

def save_results_npz(results, out_dir):
    empty = np.array([])
    np.savez(out_dir / 'attn_raw.npz',
             **{f'{STAGE_NAMES[c]}_gtn': (results[c]['gtn'] if results[c] else empty)
                for c in range(5)},
             **{f'{STAGE_NAMES[c]}_saliency': (results[c]['saliency'] if results[c] else empty)
                for c in range(5)},
             **{f'{STAGE_NAMES[c]}_rollout':
                (results[c]['rollout'] if (results[c] and results[c]['rollout'] is not None) else empty)
                for c in range(5)},
             **{f'{STAGE_NAMES[c]}_n': np.array([results[c].get('n', 1)]) if results[c] else empty
                for c in range(5)})
    print(f"raw arrays -> {out_dir / 'attn_raw.npz'}", flush=True)


def load_results_npz(npz_path):
    """Rebuild the results dict from a saved attn_raw.npz (no model/GPU needed)."""
    data = np.load(npz_path)
    results = {}
    for c in range(5):
        s = STAGE_NAMES[c]
        gtn = data[f'{s}_gtn']
        if gtn.size == 0:
            results[c] = None
            continue
        roll = data[f'{s}_rollout']
        nkey = f'{s}_n'
        n = int(data[nkey][0]) if (nkey in data and data[nkey].size) else None
        results[c] = {'gtn': gtn, 'saliency': data[f'{s}_saliency'],
                      'rollout': roll if roll.size else None, 'pred': c, 'n': n}
    return results


def write_figures(results, out_dir):
    print('\n[plotting] writing figures ...', flush=True)
    import traceback
    for fn, name in [(plot_gtn_global, 'attn_gtn_global.png'),
                     (plot_gtn_edges, 'attn_gtn_edges.png'),
                     (plot_temporal, 'attn_temporal.png')]:
        try:
            fn(results, out_dir / name)
        except Exception:
            print(f'[ERROR] {fn.__name__} failed:'); traceback.print_exc()


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-dir', type=Path, default=Path('./cache_dataset'))
    p.add_argument('--ckpt', type=Path, default=None)
    p.add_argument('--context-size', type=int, default=7)
    p.add_argument('--n-subjects', type=int, default=20)
    p.add_argument('--n-per-stage', type=int, default=30,
                   help='Average attention over the top-N highest-confidence correct '
                        'epochs per stage (1 = single representative epoch).')
    p.add_argument('--batch-size', type=int, default=16,
                   help='Pass-1 inference batch size (lower if GPU VRAM is tight).')
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--exclude-splits', type=Path, default=None)
    p.add_argument('--out-dir', type=Path, default=Path('./figures'))
    p.add_argument('--replot-from', type=Path, default=None,
                   help='Skip model/data entirely: reload a saved attn_raw.npz and just '
                        'redraw the figures (tiny RAM footprint — use to iterate on layout).')
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # fast path: redraw from cached raw attention, no torch / no dataset preload
    if args.replot_from is not None:
        print(f'[replot] loading {args.replot_from} (no model/data)', flush=True)
        write_figures(load_results_npz(args.replot_from), args.out_dir)
        return

    if args.ckpt is None:
        p.error('--ckpt is required unless --replot-from is given')

    import torch
    from datasets.cinc2018 import CinC2018EpochDataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')

    # subject pool (exclude train/val so we visualize on held-out subjects)
    idx = np.load(args.cache_dir / 'index.npz', allow_pickle=False)
    all_records = sorted(set(idx['record_ids'].tolist()))
    if args.exclude_splits is not None and args.exclude_splits.exists():
        import json
        split = json.loads(args.exclude_splits.read_text())
        excluded = set(split.get('train', [])) | set(split.get('val', []))
        all_records = [r for r in all_records if r not in excluded]
        print(f'pool after exclusion: {len(all_records)} subjects')
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(all_records, size=min(args.n_subjects, len(all_records)),
                        replace=False).tolist()

    ds = CinC2018EpochDataset(args.cache_dir, record_ids=chosen,
                              context_size=args.context_size)
    print(f'dataset : {len(ds):,} epochs from {len(chosen)} subjects')

    model = build_model(args.ckpt, args.context_size, device)

    print(f'\n[pass 1] selecting top-{args.n_per_stage} epochs per stage ...')
    picks = pick_examples(model, ds, device, args.n_per_stage, batch_size=args.batch_size)

    print('\n[pass 2] capturing + averaging attention ...')
    gtn_mhas, vit_layers = patch_for_capture(model)
    results = {}
    for c in range(5):
        if not picks[c]:
            results[c] = None
            continue
        gtns, sals, rolls = [], [], []
        for gi in picks[c]:
            r = analyze_item(model, ds, gi, args.context_size,
                             gtn_mhas, vit_layers, device)
            gtns.append(r['gtn'])
            sals.append(r['saliency'])
            if r['rollout'] is not None:
                rolls.append(r['rollout'])
        # average RAW values across epochs; per-stage normalization is done at plot time
        results[c] = {'gtn': np.mean(gtns, axis=0) if gtns else None,
                      'saliency': np.mean(sals, axis=0),
                      'rollout': np.mean(rolls, axis=0) if rolls else None,
                      'pred': c, 'n': len(picks[c])}
        print(f'  {STAGE_NAMES[c]}: averaged over {len(picks[c])} epochs')

    # Save raw attention FIRST — the GPU work is expensive; never lose it to a
    # plotting crash (this box has flaky RAM that can hard-kill matplotlib).
    save_results_npz(results, args.out_dir)
    write_figures(results, args.out_dir)


if __name__ == '__main__':
    main()
