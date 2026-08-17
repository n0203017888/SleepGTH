"""CinC Challenge 2018 sleep-staging dataset for SleepGTH (supervised).

Cache is built from CinC2018/training/ with labels 0..4 (W/N1/N2/N3/R).
Records without a usable .arousal file or with unknown ('?') stages have
those epochs excluded from the index.

One-time cache build:
    python -m datasets.cinc2018 --data-root E:/CinC_Challenge_2018/training --out-dir ./cache_dataset

Use in code:
    from datasets.cinc2018 import CinC2018EpochDataset
    ds = CinC2018EpochDataset('./cache_dataset')
    item = ds[0]
    # item = {'eeg': (6, 6000), 'label': int}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import wfdb
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import Dataset
from tqdm import tqdm


EEG_CHANNELS = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']
TARGET_CHANNELS = EEG_CHANNELS

N_EEG = len(EEG_CHANNELS)

FS = 200
EPOCH_SEC = 30
EPOCH_LEN = FS * EPOCH_SEC

# Paper-compliant bandpass cutoffs for EEG.
EEG_BAND = (0.3, 35.0)
FILTER_ORDER = 4

STAGE_TO_LABEL = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}
NUM_CLASSES = 5


def _bandpass(sig: np.ndarray, lo: float, hi: float, fs: int = FS,
              order: int = FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth bandpass along the last axis."""
    sos = butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, sig, axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Cache build (run once)
# --------------------------------------------------------------------------- #

def _stage_per_epoch(record_path: Path, n_epochs: int) -> np.ndarray:
    """Read .arousal aux_note change-points, expand to per-epoch labels (-1 for unknown)."""
    ann = wfdb.rdann(str(record_path), 'arousal')
    events = [(s, STAGE_TO_LABEL[a.strip()])
              for s, a in zip(ann.sample, ann.aux_note)
              if a.strip() in STAGE_TO_LABEL]

    labels = np.full(n_epochs, -1, dtype=np.int8)
    current = -1
    it = iter(events)
    nxt = next(it, None)
    for ep in range(n_epochs):
        ep_start = ep * EPOCH_LEN
        while nxt is not None and nxt[0] <= ep_start:
            current = nxt[1]
            nxt = next(it, None)
        labels[ep] = current
    return labels


def _process_record(record_path: Path, out_dir: Path) -> tuple[np.ndarray, bool] | None:
    """Return (per-epoch labels, has_labels). Records without .arousal get all -1."""
    rec = wfdb.rdrecord(str(record_path))
    if rec.fs != FS:
        return None
    name_to_idx = {n: i for i, n in enumerate(rec.sig_name)}
    if any(c not in name_to_idx for c in TARGET_CHANNELS):
        return None
    col_idx = [name_to_idx[c] for c in TARGET_CHANNELS]
    sig = rec.p_signal[:, col_idx].astype(np.float32)

    # Bandpass filter along time axis (paper: EEG 0.3-35 Hz at fs=200).
    sig_t = sig.T  # (n_channels, T)
    sig_t[:] = _bandpass(sig_t, *EEG_BAND)
    sig = sig_t.T

    sig = (sig - sig.mean(axis=0, keepdims=True)) / (sig.std(axis=0, keepdims=True) + 1e-6)

    n_epochs = sig.shape[0] // EPOCH_LEN
    sig = sig[:n_epochs * EPOCH_LEN].reshape(n_epochs, EPOCH_LEN, len(TARGET_CHANNELS))
    sig = np.ascontiguousarray(sig.transpose(0, 2, 1))

    has_labels = record_path.with_suffix('.arousal').exists()
    if has_labels:
        labels = _stage_per_epoch(record_path, n_epochs)
    else:
        labels = np.full(n_epochs, -1, dtype=np.int8)

    np.save(out_dir / f'{record_path.name}_signal.npy', sig)
    np.save(out_dir / f'{record_path.name}_labels.npy', labels)
    return labels, has_labels


def build_cache(data_root: Path, out_dir: Path, limit: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    record_ids = [r.strip().rstrip('/')
                  for r in (data_root / 'RECORDS').read_text().splitlines()
                  if r.strip()]
    if limit:
        record_ids = record_ids[:limit]

    rec_arr: list[str] = []
    ep_arr: list[int] = []
    lab_arr: list[int] = []
    skipped = 0
    for rid in tqdm(record_ids, desc='caching'):
        result = _process_record(data_root / rid / rid, out_dir)
        if result is None:
            skipped += 1
            continue
        labels, has_labels = result
        # Labeled records: drop unknown ('?') gaps. Unlabeled: keep every epoch.
        keep = labels >= 0 if has_labels else np.ones_like(labels, dtype=bool)
        rec_arr.extend([rid] * int(keep.sum()))
        ep_arr.extend(np.flatnonzero(keep).tolist())
        lab_arr.extend(labels[keep].tolist())

    np.savez(
        out_dir / 'index.npz',
        record_ids=np.array(rec_arr),
        epoch_idx=np.array(ep_arr, dtype=np.int32),
        labels=np.array(lab_arr, dtype=np.int8),
    )
    print(f'wrote {len(lab_arr)} epochs from {len(record_ids) - skipped} records '
          f'({skipped} skipped) -> {out_dir / "index.npz"}')


# --------------------------------------------------------------------------- #
# PyTorch Dataset
# --------------------------------------------------------------------------- #


class CinC2018EpochDataset(Dataset):
    """One 30-s epoch per item.

    Pass ``record_ids`` to restrict the dataset to a specific split (e.g. train/val/test).
    Signals are fully preloaded into RAM at init time (no mmap), for stable sequential access
    with num_workers=0.
    """

    def __init__(self, cache_dir: str | Path, record_ids: list[str] | None = None,
                 transform=None, context_size: int = 1):
        if context_size < 1 or context_size % 2 == 0:
            raise ValueError(f'context_size must be a positive odd integer, got {context_size}')
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.context_size = context_size
        idx = np.load(self.cache_dir / 'index.npz', allow_pickle=False)
        rec, ep, lb = idx['record_ids'], idx['epoch_idx'], idx['labels']
        if record_ids is not None:
            keep = np.isin(rec, list(record_ids))
            rec, ep, lb = rec[keep], ep[keep], lb[keep]
        self.record_ids = rec
        self.epoch_idx = ep
        self.labels = lb

        unique = sorted(set(rec.tolist()))
        self.signals: dict[str, np.ndarray] = {}
        for rid in tqdm(unique, desc=f'preloading {len(unique)} subjects', leave=False):
            arr = np.load(self.cache_dir / f'{rid}_signal.npy')  # (n_epochs, n_ch, 6000)
            if arr.shape[1] < N_EEG:
                raise RuntimeError(
                    f'Cache for {rid!r} has only {arr.shape[1]} stored channels, '
                    f'but {N_EEG} EEG channels are required. '
                    f'Please rebuild the cache:\n'
                    f'  python -m datasets.cinc2018 --data-root <DATA_ROOT> '
                    f'--out-dir {self.cache_dir}'
                )
            if arr.shape[1] > N_EEG:
                # Caches built before the EEG-only refactor store extra trailing channels.
                # EEG comes first, so a prefix slice is all that is needed; copy so the
                # rest of the array is freed from RAM.
                arr = np.ascontiguousarray(arr[:, :N_EEG, :])    # (n_epochs, N_EEG, 6000)
            self.signals[rid] = arr  # (n_epochs, N_EEG, 6000)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        rid = str(self.record_ids[i])
        ep = int(self.epoch_idx[i])
        signals = self.signals[rid]                              # (n_epochs_total, N_EEG, 6000)

        if self.context_size == 1:
            sig = signals[ep]                                    # (N_EEG, 6000)
            item = {'eeg': torch.from_numpy(sig[:N_EEG].copy()).float(),
                    'label': int(self.labels[i])}
        else:
            half = self.context_size // 2
            start, end = ep - half, ep + half + 1                # half-open [start, end)
            n_total = signals.shape[0]
            ctx = np.zeros((self.context_size, signals.shape[1], signals.shape[2]),
                           dtype=signals.dtype)
            v_start = max(start, 0)
            v_end = min(end, n_total)
            ctx[v_start - start:v_end - start] = signals[v_start:v_end]
            # ctx: (L, N_EEG, 6000) — zero-padded on boundary epochs
            item = {'eeg': torch.from_numpy(ctx[:, :N_EEG, :].copy()).float(),
                    'label': int(self.labels[i])}                # center epoch's label
        if self.transform is not None:
            item = self.transform(item)
        return item


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('E:/CinC_Challenge_2018/training'))
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=None,
                   help='Process only the first N records (debugging).')
    args = p.parse_args()
    build_cache(args.data_root, args.out_dir, args.limit)
