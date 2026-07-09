"""
Sub-pixel registration + parity statistics against a reference dataset.

Extracted from the three per-instrument integration scripts (which had
triplicated it); behaviour matches the phase-3 version — bounds-guarded
parabolic peak refinement, bright-pixel data ratios, masked-pixel-excluded
noise ratios.
"""

from typing import Dict, Tuple

import numpy as np

# Pixels at/above this noise value are masked-by-noise products (see
# noise.rms.MASKED_NOISE_VALUE) or their shift-interpolation bleed.
_MASKED_EXCLUSION_THRESHOLD = 1.0e6


def subpixel_offset(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """(dy, dx) shift of `b` relative to `a`: FFT cross-correlation + parabolic peak."""
    a0 = np.nan_to_num(a - np.nanmedian(a))
    b0 = np.nan_to_num(b - np.nanmedian(b))
    corr = np.fft.fftshift(
        np.fft.irfft2(np.fft.rfft2(a0) * np.conj(np.fft.rfft2(b0)), s=a0.shape)
    )
    peak = np.unravel_index(np.argmax(corr), corr.shape)

    def parabolic(axis):
        prev = list(peak); prev[axis] -= 1
        nxt = list(peak); nxt[axis] += 1
        if min(prev[axis], 0) < 0 or nxt[axis] >= corr.shape[axis]:
            return 0.0
        cm, c0, cp = corr[tuple(prev)], corr[peak], corr[tuple(nxt)]
        denom = cm - 2 * c0 + cp
        return 0.0 if denom == 0 else 0.5 * (cm - cp) / denom

    return (
        peak[0] - a.shape[0] // 2 + parabolic(0),
        peak[1] - a.shape[1] // 2 + parabolic(1),
    )


def registered_ratios(
    new_data: np.ndarray,
    new_noise: np.ndarray,
    ref_data: np.ndarray,
    ref_noise: np.ndarray,
    bright_sigma: float = 10.0,
) -> Dict:
    """Register `new` onto `ref` (sub-pixel) and report parity statistics."""
    from scipy.ndimage import shift as nd_shift

    if new_data.shape != ref_data.shape:
        raise ValueError(
            f"shape mismatch: new {new_data.shape} vs reference {ref_data.shape}"
        )

    dy, dx = subpixel_offset(ref_data, new_data)
    new_data_r = nd_shift(np.nan_to_num(new_data), (dy, dx), order=3)
    new_noise_r = nd_shift(np.nan_to_num(new_noise), (dy, dx), order=1)

    bright = ref_data > bright_sigma * np.nanmedian(ref_noise)
    data_ratio = new_data_r[bright] / ref_data[bright]
    # Exclude masked-by-noise pixels and their shift-interpolation bleed.
    valid = new_noise_r < _MASKED_EXCLUSION_THRESHOLD
    noise_ratio = np.where(valid, new_noise_r / ref_noise, np.nan)
    return {
        "offset": [float(dy), float(dx)],
        "n_bright": int(bright.sum()),
        "data_ratio_median": float(np.nanmedian(data_ratio)),
        "data_ratio_16_84": [
            float(np.nanpercentile(data_ratio, 16)),
            float(np.nanpercentile(data_ratio, 84)),
        ],
        "noise_ratio_median": float(np.nanmedian(noise_ratio)),
        "noise_ratio_16_84": [
            float(np.nanpercentile(noise_ratio, 16)),
            float(np.nanpercentile(noise_ratio, 84)),
        ],
    }
