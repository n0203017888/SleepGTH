"""Sleep-EEG augmentation transforms (training only — never apply to val/test).

Each transform takes the dict returned by CinC2018EpochDataset
    {'eeg': (C, 6000) or (L, C, 6000), 'label': int}
and returns a new dict with augmented signals (label unchanged).
"""
from __future__ import annotations

import random

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt


# --------------------------------------------------------------------------- #
# Primitives (each carries its own probability gate)
# --------------------------------------------------------------------------- #

class GaussianNoise:
    """Add per-sample Gaussian noise. Signals are already z-scored (~unit std),
    so noise std=0.05 = ~5% relative noise level."""

    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, item: dict) -> dict:
        out = dict(item)
        for k in ('eeg',):
            out[k] = item[k] + torch.randn_like(item[k]) * self.std
        return out


class TimeMask:
    """With probability p, zero out one random time window in all channels.

    Window length sampled uniformly in [min_len, max_len] samples. At 200 Hz,
    a window of 100..400 samples is 0.5..2.0 s of the 30-s epoch.
    """

    def __init__(self, p: float = 0.5, min_len: int = 100, max_len: int = 400):
        self.p = p
        self.min_len = min_len
        self.max_len = max_len

    def __call__(self, item: dict) -> dict:
        if random.random() > self.p:
            return item
        out = dict(item)
        T = item['eeg'].shape[-1]
        win = random.randint(self.min_len, self.max_len)
        start = random.randint(0, T - win)
        for k in ('eeg',):
            sig = item[k].clone()
            sig[..., start:start + win] = 0
            out[k] = sig
        return out


class ChannelDropout:
    """Independently zero out each channel with probability p."""

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, item: dict) -> dict:
        out = dict(item)
        for k in ('eeg',):
            sig = item[k].clone()
            for c in range(sig.shape[0]):
                if random.random() < self.p:
                    sig[c] = 0
            out[k] = sig
        return out


class AmplitudeScaling:
    """Multiply all channels by a random scalar in [low, high]."""

    def __init__(self, low: float = 0.8, high: float = 1.2):
        self.low = low
        self.high = high

    def __call__(self, item: dict) -> dict:
        out = dict(item)
        scale = random.uniform(self.low, self.high)
        for k in ('eeg',):
            out[k] = item[k] * scale
        return out


# --------------------------------------------------------------------------- #
# New augmentations (each independently gated by p=0.5)
# --------------------------------------------------------------------------- #

class RandomAmplitudeScale:
    """Multiply all channels by a random scalar in [low, high], with prob p."""

    def __init__(self, p: float = 0.5, low: float = 0.5, high: float = 2.0):
        self.p = p
        self.low = low
        self.high = high

    def __call__(self, item: dict) -> dict:
        if random.random() > self.p:
            return item
        out = dict(item)
        scale = random.uniform(self.low, self.high)
        for k in ('eeg',):
            out[k] = item[k] * scale
        return out


class RandomTimeShift:
    """Circularly shift the signal along the time axis by a random offset.

    shift_range: (min, max) in samples. At 200 Hz, ±600 = ±3 s.
    Circular shift avoids introducing discontinuities at the boundary.
    """

    def __init__(self, p: float = 0.5, shift_range: tuple[int, int] = (-600, 600)):
        self.p = p
        self.shift_range = shift_range

    def __call__(self, item: dict) -> dict:
        if random.random() > self.p:
            return item
        out = dict(item)
        shift = random.randint(self.shift_range[0], self.shift_range[1])
        for k in ('eeg',):
            out[k] = torch.roll(item[k], shift, dims=-1)
        return out


class RandomAmplitudeShift:
    """Add a random DC offset sampled from [low, high] to all channels.

    Signals are z-scored, so ±0.5 is a moderate shift (~half a std).
    """

    def __init__(self, p: float = 0.5, low: float = -0.5, high: float = 0.5):
        self.p = p
        self.low = low
        self.high = high

    def __call__(self, item: dict) -> dict:
        if random.random() > self.p:
            return item
        out = dict(item)
        offset = random.uniform(self.low, self.high)
        for k in ('eeg',):
            out[k] = item[k] + offset
        return out


class RandomZeroMasking:
    """Zero out a random contiguous window of length in [min_len, max_len] samples.

    At 200 Hz, max_len=600 = 3 s out of a 30-s epoch (10% of the epoch).
    """

    def __init__(self, p: float = 0.5, mask_range: tuple[int, int] = (0, 600)):
        self.p = p
        self.mask_range = mask_range

    def __call__(self, item: dict) -> dict:
        if random.random() > self.p:
            return item
        win = random.randint(self.mask_range[0], self.mask_range[1])
        if win == 0:
            return item
        out = dict(item)
        T = item['eeg'].shape[-1]
        start = random.randint(0, max(0, T - win))
        for k in ('eeg',):
            sig = item[k].clone()
            sig[..., start:start + win] = 0.0
            out[k] = sig
        return out


class RandomGaussianNoise:
    """Add zero-mean Gaussian noise with std sampled from [std_low, std_high], with prob p."""

    def __init__(self, p: float = 0.5, std: float = 0.05):
        self.p = p
        self.std = std

    def __call__(self, item: dict) -> dict:
        if random.random() > self.p:
            return item
        out = dict(item)
        for k in ('eeg',):
            out[k] = item[k] + torch.randn_like(item[k]) * self.std
        return out


class RandomBandStopFilter:
    """Apply a zero-phase Butterworth band-stop filter at a random centre frequency.

    Centre frequency is sampled uniformly from freq_range (Hz). The notch width
    is band_width Hz. Uses scipy sosfiltfilt (zero-phase, applied on CPU numpy array).
    """

    def __init__(self, p: float = 0.5,
                 freq_range: tuple[float, float] = (0.5, 30.0),
                 band_width: float = 2.0,
                 sampling_rate: float = 200.0):
        self.p = p
        self.freq_range = freq_range
        self.band_width = band_width
        self.fs = sampling_rate

    def __call__(self, item: dict) -> dict:
        if random.random() > self.p:
            return item
        center = random.uniform(self.freq_range[0], self.freq_range[1])
        half = self.band_width / 2.0
        nyq = self.fs / 2.0
        low  = max(center - half, 0.1)
        high = min(center + half, nyq - 0.1)
        sos = butter(4, [low, high], btype='bandstop', fs=self.fs, output='sos')

        out = dict(item)
        for k in ('eeg',):
            sig_np = item[k].numpy()
            filtered = sosfiltfilt(sos, sig_np, axis=-1).astype(np.float32)
            out[k] = torch.from_numpy(filtered)
        return out


# --------------------------------------------------------------------------- #
# Composition helpers
# --------------------------------------------------------------------------- #

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, item: dict) -> dict:
        for t in self.transforms:
            item = t(item)
        return item


class LabelConditionalAugment:
    """Apply different augmentation pipelines depending on the sample label.

    label_transforms: dict mapping label int -> transform callable.
    Labels not in the dict are passed through unchanged.
    """

    def __init__(self, label_transforms: dict):
        self.label_transforms = label_transforms

    def __call__(self, item: dict) -> dict:
        t = self.label_transforms.get(item['label'])
        return t(item) if t is not None else item


# --------------------------------------------------------------------------- #
# Factory functions
# --------------------------------------------------------------------------- #

def default_train_augment(noise_std: float = 0.05,
                          mask_p: float = 0.5,
                          mask_max: int = 400,
                          channel_drop_p: float = 0.1) -> Compose:
    """Standard sleep-EEG 3-pack: noise -> time mask -> channel dropout."""
    return Compose([
        GaussianNoise(std=noise_std),
        TimeMask(p=mask_p, min_len=100, max_len=mask_max),
        ChannelDropout(p=channel_drop_p),
    ])


def label_conditional_augment() -> LabelConditionalAugment:
    """N1: GaussianNoise + TimeMask + ChannelDropout.
       N3: GaussianNoise + AmplitudeScaling.
       Others: no augmentation.
    """
    n1_aug = Compose([GaussianNoise(), TimeMask(), ChannelDropout()])
    n3_aug = Compose([GaussianNoise(), AmplitudeScaling()])
    return LabelConditionalAugment({1: n1_aug, 3: n3_aug})


def six_pack_augment() -> Compose:
    """6 independent augmentations, each applied with p=0.5:
        1. RandomAmplitudeScale   (×0.5 ~ ×2.0)
        2. RandomTimeShift        (±600 samples = ±3 s, circular)
        3. RandomAmplitudeShift   (DC offset ±0.5 std)
        4. RandomZeroMasking      (0 ~ 600 samples = 0 ~ 3 s)
        5. RandomGaussianNoise    (std=0.05)
        6. RandomBandStopFilter   (0.5~30 Hz, width=2 Hz, fs=200)
    """
    return Compose([
        RandomAmplitudeScale(p=0.5, low=0.5, high=2.0),
        RandomTimeShift(p=0.5, shift_range=(-600, 600)),
        RandomAmplitudeShift(p=0.5, low=-0.5, high=0.5),
        RandomZeroMasking(p=0.5, mask_range=(0, 600)),
        RandomGaussianNoise(p=0.5, std=0.05),
        RandomBandStopFilter(p=0.5, freq_range=(0.5, 30.0),
                             band_width=2.0, sampling_rate=200.0),
    ])
