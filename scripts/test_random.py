"""用指定權重對隨機抽取的 N 個受試者做 test。

用法：
    python scripts/test_random.py \
        --cache-dir ./cache_dataset \
        --ckpt ./checkpoints_kfold_500_ctx5_eeg_k101_201/fold_03_best.pt \
        --n-subjects 200 \
        --context-size 5 \
        --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.cinc2018 import CinC2018EpochDataset
from engine.train import evaluate
from models.sleepgth import SleepGTH


STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'R']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-dir', type=Path, default=Path('./cache_dataset'))
    p.add_argument('--ckpt', type=Path, required=True,
                   help='Path to fold_XX_best.pt checkpoint.')
    p.add_argument('--n-subjects', type=int, default=200,
                   help='Number of subjects to randomly sample.')
    p.add_argument('--context-size', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--modalities', choices=['eeg', 'eeg_eog', 'eeg_eog_emg'], default='eeg')
    p.add_argument('--use-eog', action='store_true', default=False,
                   help='Enable epoch-level EOGEncoder (must match training config).')
    p.add_argument('--fusion', choices=['concat', 'hwgate'], default='concat',
                   help='EOG fusion type (must match the checkpoint training config).')
    p.add_argument('--eog-dim', type=int, default=64,
                   help='EOGEncoder output dim (must match the checkpoint training config).')
    p.add_argument('--eog-kernels', type=int, nargs=2, default=(11, 201), metavar=('SHORT', 'LONG'),
                   help='EOGEncoder two-branch kernel sizes (must match training config).')
    p.add_argument('--fusion-level', choices=['epoch', 'patch'], default='epoch',
                   help='EOG/EMG fusion level (must match the checkpoint training config).')
    p.add_argument('--use-emg', action='store_true', default=False,
                   help='Enable epoch-level EMG encoder (must match training config).')
    p.add_argument('--emg-dim', type=int, default=16,
                   help='EMG encoder output dim (must match training config).')
    p.add_argument('--emg-kernels', type=int, nargs=2, default=(11, 201), metavar=('SHORT', 'LONG'),
                   help='EMG encoder two-branch kernel sizes (must match training config).')
    p.add_argument('--readout', choices=['fusion', 'global'], default='fusion',
                   help='Patch-token readout (must match training config).')
    p.add_argument('--no-gnn', dest='use_gnn', action='store_false', default=True,
                   help='Ablation: remove the GNN (must match training config).')
    p.add_argument('--no-vit', dest='use_vit', action='store_false', default=True,
                   help='Ablation: remove the TemporalViT (must match training config).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exclude-splits', type=Path, default=None,
                   help='Path to splits.json; train+val subjects will be excluded from the pool.')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')

    # 隨機抽取 N 個受試者
    idx = np.load(args.cache_dir / 'index.npz', allow_pickle=False)
    all_records = sorted(set(idx['record_ids'].tolist()))

    if args.exclude_splits is not None:
        import json
        split = json.loads(args.exclude_splits.read_text())
        excluded = set(split.get('train', [])) | set(split.get('val', []))
        # support k-fold format (list of fold dicts) as well
        if isinstance(split, list):
            excluded = set()
            for fold in split:
                excluded |= set(fold.get('train', [])) | set(fold.get('val', []))
        all_records = [r for r in all_records if r not in excluded]
        print(f'pool after exclusion: {len(all_records)} subjects '
              f'(excluded {len(excluded)} train+val)')

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(all_records, size=args.n_subjects, replace=False).tolist()
    print(f'sampled {len(chosen)} subjects (seed={args.seed})')

    # Only preload channels the model uses (saves RAM)
    load_eog = args.modalities in ('eeg_eog', 'eeg_eog_emg') or args.use_eog
    load_emg = args.modalities == 'eeg_eog_emg' or args.use_emg
    ds = CinC2018EpochDataset(args.cache_dir, record_ids=chosen,
                              context_size=args.context_size,
                              load_eog=load_eog, load_emg=load_emg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    print(f'dataset : {len(ds):,} epochs')

    # 建立模型並載入權重
    use_eog = args.modalities in ('eeg_eog', 'eeg_eog_emg')
    use_emg = args.modalities == 'eeg_eog_emg'
    model = SleepGTH(use_eog=use_eog, use_emg=use_emg,
                         use_eog_encoder=args.use_eog, use_emg_encoder=args.use_emg,
                         fusion=args.fusion, eog_dim=args.eog_dim, emg_dim=args.emg_dim,
                         eog_kernels=tuple(args.eog_kernels), emg_kernels=tuple(args.emg_kernels),
                         fusion_level=args.fusion_level,
                         use_gnn=args.use_gnn, use_vit=args.use_vit, readout=args.readout,
                         context_size=args.context_size).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    # strict=False tolerates checkpoints trained before the patch_head/continuity removal
    # (extra 'temporal.patch_head.*' keys). Shape mismatches still raise.
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing or unexpected:
        print(f'[load_state_dict] missing={missing}  unexpected={unexpected}')
    print(f'loaded  : {args.ckpt}')

    # 評估
    result = evaluate(model, loader, device, progress=True)
    print(f'\n--- Test Results ({args.n_subjects} subjects) ---')
    print(f'accuracy : {result["accuracy"]:.4f}')
    print(f'kappa    : {result["kappa"]:.4f}')
    print(f'macro_F1 : {result["macro_f1"]:.4f}')
    print(f'per-class F1:')
    for name, f1 in zip(STAGE_NAMES, result['per_class_f1']):
        print(f'  {name}: {f1:.4f}')
    print(f'\nConfusion matrix (rows=true, cols=pred):')
    print(f'       W     N1    N2    N3    R')
    for name, row in zip(STAGE_NAMES, result['confusion_matrix']):
        print(f'  {name}  ' + '  '.join(f'{v:5d}' for v in row))

    # 儲存混淆矩陣圖片
    cm = np.array(result['confusion_matrix'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(STAGE_NAMES); ax.set_yticklabels(STAGE_NAMES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Test ({args.n_subjects} subjects, seed={args.seed})\n'
                 f'acc={result["accuracy"]:.4f}  kappa={result["kappa"]:.4f}  '
                 f'mF1={result["macro_f1"]:.4f}')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.2f})',
                    ha='center', va='center', fontsize=8,
                    color='white' if cm_norm[i, j] > 0.5 else 'black')
    plt.tight_layout()
    out_path = args.ckpt.parent / f'test_random_n{args.n_subjects}_seed{args.seed}_confusion.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'\nconfusion matrix saved -> {out_path}')


if __name__ == '__main__':
    main()
