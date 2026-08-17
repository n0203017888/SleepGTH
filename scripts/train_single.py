"""Single train/val run (no k-fold, no test split).

Randomly samples n_train + n_val subjects from cache_dataset, trains SleepGTH,
saves best checkpoint by val kappa. Use test_random.py to evaluate separately.

用法：
    python scripts/train_single.py \
        --cache-dir ./cache_dataset \
        --ckpt-dir ./checkpoints_single_ctx5 \
        --n-train 400 \
        --n-val 80 \
        --context-size 5 \
        --epochs 100 \
        --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.augment import default_train_augment, label_conditional_augment, six_pack_augment
from datasets.cinc2018 import CinC2018EpochDataset
from engine.train import evaluate, train_one_epoch
from losses.classification import ClassificationLoss
from models.sleepgth import SleepGTH
from utils.scheduler import cosine_warmup_scheduler


STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'R']


def make_split(cache_dir: Path, n_train: int, n_val: int, seed: int) -> dict:
    idx = np.load(cache_dir / 'index.npz', allow_pickle=False)
    all_records = sorted(set(idx['record_ids'].tolist()))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_records, size=n_train + n_val, replace=False).tolist()
    rng.shuffle(chosen)
    return {
        'train': sorted(chosen[:n_train]),
        'val':   sorted(chosen[n_train:]),
    }


def save_confusion_matrix(cm: list, ckpt_dir: Path, tag: str = 'best') -> None:
    cm = np.array(cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(STAGE_NAMES); ax.set_yticklabels(STAGE_NAMES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Val confusion matrix ({tag})')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.2f})',
                    ha='center', va='center', fontsize=8,
                    color='white' if cm_norm[i, j] > 0.5 else 'black')
    plt.tight_layout()
    path = ckpt_dir / f'val_confusion_{tag}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  confusion matrix saved -> {path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-dir', type=Path, default=Path('./cache_dataset'))
    p.add_argument('--ckpt-dir', type=Path, required=True)
    p.add_argument('--splits-file', type=Path, default=None,
                   help='JSON splits file. Auto-generated if not specified.')
    p.add_argument('--n-train', type=int, default=400)
    p.add_argument('--n-val', type=int, default=80)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--context-size', type=int, default=1)
    p.add_argument('--modalities', choices=['eeg', 'eeg_eog', 'eeg_eog_emg'], default='eeg')
    p.add_argument('--use-class-weights', action='store_true')
    p.add_argument('--class-weights', type=float, nargs=5, default=None,
                   metavar=('W', 'N1', 'N2', 'N3', 'R'))
    p.add_argument('--label-smoothing', type=float, default=0.0)
    p.add_argument('--use-focal', action='store_true')
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--augment', action='store_true')
    p.add_argument('--augment-mode', choices=['six_pack', 'label_conditional', 'default'],
                   default='six_pack')
    p.add_argument('--early-stop-patience', type=int, default=15)
    p.add_argument('--resume-ckpt', type=Path, default=None)
    p.add_argument('--pool-type', choices=['mean_max', 'avgpool8', 'mean'], default='mean_max')
    p.add_argument('--deep-branch', dest='deep_branch', action='store_true', default=True)
    p.add_argument('--no-deep-branch', dest='deep_branch', action='store_false')
    # --- architecture ablations ---
    p.add_argument('--readout', choices=['fusion', 'global'], default='fusion',
                   help="Patch-token readout. fusion = Linear(n_ch->1) over electrodes (default); "
                        "global = use the GTN global node as the token (EEG-only, requires GNN).")
    p.add_argument('--no-gnn', dest='use_gnn', action='store_false', default=True,
                   help='Ablation: remove the GNN (GTN graph attention over electrodes); '
                        'node embeddings go straight to Source Fusion.')
    p.add_argument('--no-vit', dest='use_vit', action='store_false', default=True,
                   help='Ablation: remove the TemporalViT (within-epoch patch transformer); '
                        '30 patch tokens are GAP-pooled directly into epoch_feat.')
    p.add_argument('--use-eog', action='store_true', default=False,
                   help='Enable epoch-level EOGEncoder (full 6000-sample EOG fused with EEG epoch_feat).')
    p.add_argument('--eog-only', action='store_true', default=False,
                   help='Diagnostic: bypass the EEG pipeline entirely and classify from EOG alone '
                        '(EOGEncoder + SequenceTransformer if ctx>1). Use to measure EOG signal.')
    p.add_argument('--fusion', choices=['concat', 'hwgate'], default='concat',
                   help='EOG fusion type (only used with --use-eog). '
                        'concat = GELU(Linear(cat(eeg,eog))); '
                        'hwgate = gate*mix + (1-gate)*eeg (selective, avoids N2/N3 pollution).')
    p.add_argument('--eog-dim', type=int, default=64,
                   help='EOGEncoder output dim (EOG footprint in the fusion concat). '
                        'Smaller (e.g. 16/32) matches EOG 1-channel info content vs 6-channel EEG.')
    p.add_argument('--eog-kernels', type=int, nargs=2, default=(11, 201), metavar=('SHORT', 'LONG'),
                   help='EOGEncoder two-branch kernel sizes (short, long) at 200Hz. '
                        'Default 11(~0.1s)/201(~1s); try 51(~0.25s)/401(~2s) to fit eye-movement scales.')
    p.add_argument('--fusion-level', choices=['epoch', 'patch'], default='epoch',
                   help='Where to fuse EOG/EMG. epoch = after TemporalViT (aux=1 vector, default); '
                        'patch = before TemporalViT (aux=30 tokens, joins temporal modeling).')
    p.add_argument('--use-emg', action='store_true', default=False,
                   help='Enable epoch-level EMG encoder (chin EMG, mirrors --use-eog). '
                        'EMG carries muscle tone (REM atonia / Wake high tone).')
    p.add_argument('--emg-dim', type=int, default=16,
                   help='EMG encoder output dim (EMG footprint in the fusion). Default 16 (like EOG).')
    p.add_argument('--emg-kernels', type=int, nargs=2, default=(11, 201), metavar=('SHORT', 'LONG'),
                   help='EMG encoder two-branch kernel sizes.')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.lr is None:
        args.lr = 5e-5 if args.context_size > 1 else 2e-4
    print(f'device : {device}')
    print(f'config : {vars(args)}')

    # --- splits ---
    if args.splits_file is None:
        args.splits_file = args.ckpt_dir / 'splits.json'
    if args.splits_file.exists():
        split = json.loads(args.splits_file.read_text())
        print(f'loaded splits : {args.splits_file}')
    else:
        split = make_split(args.cache_dir, args.n_train, args.n_val, args.seed)
        args.splits_file.parent.mkdir(parents=True, exist_ok=True)
        args.splits_file.write_text(json.dumps(split, indent=2))
        print(f'generated splits -> {args.splits_file}')
    print(f'train={len(split["train"])}  val={len(split["val"])}')

    # --- datasets ---
    if args.augment:
        if args.augment_mode == 'six_pack':
            train_aug = six_pack_augment()
        elif args.augment_mode == 'label_conditional':
            train_aug = label_conditional_augment()
        else:
            train_aug = default_train_augment()
    else:
        train_aug = None

    # Only preload channels the model actually uses (saves RAM; e.g. drop EMG if unused).
    load_eog = args.modalities in ('eeg_eog', 'eeg_eog_emg') or args.use_eog
    load_emg = args.modalities == 'eeg_eog_emg' or args.use_emg
    print(f'load channels: EEG=1 EOG={int(load_eog)} EMG={int(load_emg)}')
    train_ds = CinC2018EpochDataset(args.cache_dir, record_ids=split['train'],
                                    transform=train_aug, context_size=args.context_size,
                                    eog_only=args.eog_only, load_eog=load_eog, load_emg=load_emg)
    val_ds   = CinC2018EpochDataset(args.cache_dir, record_ids=split['val'],
                                    context_size=args.context_size,
                                    eog_only=args.eog_only, load_eog=load_eog, load_emg=load_emg)
    print(f'train epochs={len(train_ds):,}  val epochs={len(val_ds):,}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0,
                              prefetch_factor=2 if args.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # --- model ---
    use_eog_spatial = args.modalities in ('eeg_eog', 'eeg_eog_emg')
    use_emg_spatial = args.modalities == 'eeg_eog_emg'
    model = SleepGTH(use_eog=use_eog_spatial, use_emg=use_emg_spatial,
                         use_eog_encoder=args.use_eog, use_emg_encoder=args.use_emg,
                         eog_only=args.eog_only,
                         fusion=args.fusion, eog_dim=args.eog_dim, emg_dim=args.emg_dim,
                         eog_kernels=tuple(args.eog_kernels), emg_kernels=tuple(args.emg_kernels),
                         fusion_level=args.fusion_level,
                         pool_type=args.pool_type,
                         deep_branch=args.deep_branch,
                         use_gnn=args.use_gnn, use_vit=args.use_vit, readout=args.readout,
                         context_size=args.context_size).to(device)
    print(f'ablation: use_gnn={args.use_gnn}  use_vit={args.use_vit}  readout={args.readout}')

    # --- class weights ---
    class_weight = None
    if args.class_weights is not None:
        class_weight = np.array(args.class_weights, dtype=np.float32)
        print(f'class weights (manual): W={class_weight[0]:.3f} N1={class_weight[1]:.3f} '
              f'N2={class_weight[2]:.3f} N3={class_weight[3]:.3f} R={class_weight[4]:.3f}')
    elif args.use_class_weights:
        train_labels = np.array(train_ds.labels)
        w = compute_class_weight('balanced', classes=np.arange(5), y=train_labels)
        class_weight = w.astype(np.float32)
        print(f'class weights (auto): W={w[0]:.3f} N1={w[1]:.3f} N2={w[2]:.3f} '
              f'N3={w[3]:.3f} R={w[4]:.3f}')

    criterion = ClassificationLoss(
        label_smoothing=args.label_smoothing,
        class_weight=class_weight,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
    ).to(device)

    # --- resume ---
    start_epoch = 0
    best_val_kappa = -1.0
    if args.resume_ckpt is not None:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_kappa = ckpt.get('best_val_kappa', -1.0)
        print(f'resumed from {args.resume_ckpt}  '
              f'(epoch={start_epoch}, best_val_kappa={best_val_kappa:.4f})')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, foreach=False)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    sched = cosine_warmup_scheduler(opt, total_steps, warmup_steps)
    if start_epoch > 0:
        for _ in range(start_epoch * len(train_loader)):
            sched.step()
    scaler = torch.amp.GradScaler('cuda') if args.amp and device.type == 'cuda' else None

    # --- training loop ---
    history = []
    no_improve = 0
    best_state = None
    for epoch in range(start_epoch, args.epochs):
        ep_start = time.time()
        tr = train_one_epoch(model, criterion, train_loader, opt, device,
                                scheduler=sched, scaler=scaler, log_every=100)
        val = evaluate(model, val_loader, device, progress=True)
        elapsed = time.time() - ep_start
        history.append({'epoch': epoch, 'train': tr, 'val': val})
        marker = ''
        if val['kappa'] > best_val_kappa:
            best_val_kappa = val['kappa']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = '  *best*'
            torch.save({'model': best_state, 'best_val_kappa': best_val_kappa, 'epoch': epoch},
                       args.ckpt_dir / 'best.pt')
        else:
            no_improve += 1
        print(f'ep {epoch:3d}: tr loss={tr["loss_mean"]:.4f}  '
              f'val kappa={val["kappa"]:.4f}  acc={val["accuracy"]:.4f}  '
              f'mF1={val["macro_f1"]:.4f}  time={elapsed:.0f}s{marker}')
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f'early stop @ ep {epoch}: no improvement for {args.early_stop_patience} '
                  f'epochs (best kappa={best_val_kappa:.4f})')
            break

    # --- save final history & confusion matrix ---
    (args.ckpt_dir / 'history.json').write_text(
        json.dumps({'best_val_kappa': best_val_kappa, 'history': history}, indent=2, default=str))
    if best_state is not None:
        model.load_state_dict(best_state)
        val_final = evaluate(model, val_loader, device, progress=True)
        save_confusion_matrix(val_final['confusion_matrix'], args.ckpt_dir, tag='best')
        print(f'\n--- Best Val Results ---')
        print(f'kappa    : {val_final["kappa"]:.4f}')
        print(f'accuracy : {val_final["accuracy"]:.4f}')
        print(f'macro_F1 : {val_final["macro_f1"]:.4f}')
        for name, f1 in zip(STAGE_NAMES, val_final['per_class_f1']):
            print(f'  {name}: {f1:.4f}')
    print(f'\ncheckpoint -> {args.ckpt_dir / "best.pt"}')


if __name__ == '__main__':
    main()
