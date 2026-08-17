"""Copy only the records referenced in a kfold splits JSON to a smaller cache dir.

Use this when the full cache (165 GB) won't fit in OS page cache and the
training is I/O-bound. After running, point --cache-dir at the new directory.

If --splits-file does not exist, this script will generate the splits first
(deterministic via --seed), so you can run it before run_kfold.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _make_kfold_splits(cache_dir: Path, n_subjects: int, n_folds: int,
                       n_val: int, seed: int) -> list[dict]:
    """Same logic as scripts.run_kfold.make_kfold_splits — duplicated to avoid
    importing torch via run_kfold's module-level imports."""
    idx = np.load(cache_dir / 'index.npz', allow_pickle=False)
    all_records = sorted(set(idx['record_ids'].tolist()))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_records, size=n_subjects, replace=False).tolist()
    rng.shuffle(chosen)
    fold_size = n_subjects // n_folds
    folds = []
    for i in range(n_folds):
        test = chosen[i * fold_size:(i + 1) * fold_size]
        rest = [r for r in chosen if r not in test]
        val = rest[:n_val]
        train = rest[n_val:]
        folds.append({
            'fold_id': i,
            'train': sorted(train),
            'val': sorted(val),
            'test': sorted(test),
        })
    return folds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src-cache', type=Path, default=Path('./cache_dataset'))
    p.add_argument('--dst-cache', type=Path, required=True)
    p.add_argument('--splits-file', type=Path, required=True,
                   help='Load/generate kfold splits to determine which records to copy.')
    p.add_argument('--n-subjects', type=int, default=200)
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--n-val', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dtype', choices=['float32', 'float16'], default='float32',
                   help='float16 halves cache size (~16 GB for 200 records); '
                        'precision is plenty for z-scored EEG.')
    args = p.parse_args()

    args.dst_cache.mkdir(parents=True, exist_ok=True)

    args.splits_file.parent.mkdir(parents=True, exist_ok=True)
    if args.splits_file.exists():
        folds = json.loads(args.splits_file.read_text())
        print(f'loaded existing splits: {args.splits_file}')
    else:
        folds = _make_kfold_splits(args.src_cache, args.n_subjects, args.n_folds,
                                   args.n_val, args.seed)
        args.splits_file.write_text(json.dumps(folds, indent=2))
        print(f'generated splits      -> {args.splits_file}')
    used = set()
    for f in folds:
        used |= set(f['train']) | set(f['val']) | set(f['test'])
    used = sorted(used)
    print(f'records to copy: {len(used)}')

    cast_signals = (args.dtype == 'float16')
    desc = 'converting->float16' if cast_signals else 'copying'
    for rid in tqdm(used, desc=desc):
        sig_dst = args.dst_cache / f'{rid}_signal.npy'
        lab_dst = args.dst_cache / f'{rid}_labels.npy'
        if not sig_dst.exists():
            if cast_signals:
                arr = np.load(args.src_cache / f'{rid}_signal.npy')
                np.save(sig_dst, arr.astype(np.float16))
            else:
                shutil.copy2(args.src_cache / f'{rid}_signal.npy', sig_dst)
        if not lab_dst.exists():
            shutil.copy2(args.src_cache / f'{rid}_labels.npy', lab_dst)

    src_idx = np.load(args.src_cache / 'index.npz', allow_pickle=False)
    keep = np.isin(src_idx['record_ids'], used)
    np.savez(args.dst_cache / 'index.npz',
             record_ids=src_idx['record_ids'][keep],
             epoch_idx=src_idx['epoch_idx'][keep],
             labels=src_idx['labels'][keep])
    n_epochs = int(keep.sum())
    print(f'wrote index.npz with {n_epochs:,} epochs from {len(used)} records '
          f'-> {args.dst_cache}')


if __name__ == '__main__':
    main()
