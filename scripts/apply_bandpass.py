"""One-time utility: apply paper-compliant bandpass filter to an existing cache.

Run this on caches that were built before bandpass filtering was added to
``datasets/cinc2018.py``. New cache builds already include filtering and do
NOT need this script.

Filtering is done on the full per-record continuous signal (not per-epoch),
so there are no edge effects within epochs. After filtering each record we
re-z-score per channel since the bandpass shifts the mean/std slightly.

Cutoffs (matches the cache builder):
    EEG / EOG: 0.3-35 Hz Butterworth, order 4
    EMG:       10-95 Hz Butterworth, order 4
              (paper used 10-100 Hz at fs=256; we cap at 95 since fs=200 → Nyquist=100)

The original signal dtype (float32 or float16) is preserved. Files are
overwritten in place; back up the cache first if you want to keep the
unfiltered version.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm


FS = 200
N_EEG = 6
N_EOG = 1
N_EMG = 1
N_TOTAL = N_EEG + N_EOG + N_EMG
N_EEG_EOG = N_EEG + N_EOG

EEG_EOG_BAND = (0.3, 35.0)
EMG_BAND = (10.0, 95.0)
FILTER_ORDER = 4


def _bandpass(sig: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sos = butter(FILTER_ORDER, [lo, hi], btype='bandpass', fs=FS, output='sos')
    return sosfiltfilt(sos, sig, axis=-1)


def process_record(sig_path: Path) -> None:
    sig = np.load(sig_path)                                  # (n_epochs, 8, 6000)
    orig_dtype = sig.dtype
    n_ep = sig.shape[0]

    # (n_epochs, 8, 6000) -> (8, n_epochs * 6000) continuous per channel.
    sig_cont = (sig.astype(np.float32)
                   .transpose(1, 0, 2)
                   .reshape(N_TOTAL, -1))

    sig_cont[:N_EEG_EOG] = _bandpass(sig_cont[:N_EEG_EOG], *EEG_EOG_BAND)
    sig_cont[N_EEG_EOG:] = _bandpass(sig_cont[N_EEG_EOG:], *EMG_BAND)

    # Re-z-score per channel (filter shifts mean/std).
    mean = sig_cont.mean(axis=1, keepdims=True)
    std = sig_cont.std(axis=1, keepdims=True)
    sig_cont = (sig_cont - mean) / (std + 1e-6)

    sig_new = (sig_cont.reshape(N_TOTAL, n_ep, -1)
                       .transpose(1, 0, 2)
                       .astype(orig_dtype))
    np.save(sig_path, sig_new)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-dir', type=Path, required=True)
    args = p.parse_args()

    if not args.cache_dir.exists():
        raise SystemExit(f'cache dir not found: {args.cache_dir}')

    idx = np.load(args.cache_dir / 'index.npz', allow_pickle=False)
    record_ids = sorted(set(idx['record_ids'].tolist()))
    print(f'cache: {args.cache_dir}')
    print(f'records to filter: {len(record_ids)}')
    print(f'  EEG/EOG: {EEG_EOG_BAND[0]}-{EEG_EOG_BAND[1]} Hz')
    print(f'  EMG:     {EMG_BAND[0]}-{EMG_BAND[1]} Hz')

    for rid in tqdm(record_ids, desc='filtering'):
        sig_path = args.cache_dir / f'{rid}_signal.npy'
        if not sig_path.exists():
            print(f'  warn: missing {sig_path}, skipping')
            continue
        process_record(sig_path)

    print(f'done. {len(record_ids)} records updated in place.')


if __name__ == '__main__':
    main()
