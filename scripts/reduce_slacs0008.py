"""
Integration + acceptance (issue #2): reduce slacs0008-0004 through the
*production* pipeline — proposal-filtered exposure set (10886 only, the
spike's neighbouring-pointing contamination excluded) — then compare the
products against the legacy modeling dataset with sub-pixel registration.

Run:  ~/venv/PyAuto/bin/python scripts/reduce_slacs0008.py
Network + drizzlepac required; unit tests never import this.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from autoreduce import TargetSpec, reduce_target  # noqa: E402

LEGACY_DIR = Path("/mnt/c/Users/Jammy/Science/subhalo/dataset/slacs/slacs0008-0004")
CACHE_ROOT = REPO / "scripts" / "cache"
OUTPUT_ROOT = REPO / "scripts" / "output"

SPEC = TargetSpec(
    name="slacs0008-0004",
    ra=2.012333,
    dec=-0.068944,
    proposal_ids=("10886",),
)


def subpixel_offset(a: np.ndarray, b: np.ndarray):
    """(dy, dx) shift of b relative to a: FFT cross-correlation + parabolic peak."""
    a0 = np.nan_to_num(a - np.nanmedian(a))
    b0 = np.nan_to_num(b - np.nanmedian(b))
    corr = np.fft.irfft2(np.fft.rfft2(a0) * np.conj(np.fft.rfft2(b0)), s=a0.shape)
    corr = np.fft.fftshift(corr)
    peak = np.unravel_index(np.argmax(corr), corr.shape)

    def parabolic(idx, axis):
        c0 = corr[peak]
        prev = list(peak); prev[axis] -= 1
        nxt = list(peak); nxt[axis] += 1
        cm, cp = corr[tuple(prev)], corr[tuple(nxt)]
        denom = cm - 2 * c0 + cp
        return 0.0 if denom == 0 else 0.5 * (cm - cp) / denom

    dy = peak[0] - a.shape[0] // 2 + parabolic(peak, 0)
    dx = peak[1] - a.shape[1] // 2 + parabolic(peak, 1)
    return dy, dx


def main():
    record = reduce_target(SPEC, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)
    print(json.dumps(record["drizzle"]["weight_uniformity"], indent=2))

    from astropy.io import fits
    from scipy.ndimage import shift as nd_shift

    out_dir = OUTPUT_ROOT / SPEC.name
    new_data = fits.getdata(out_dir / "data.fits").astype(float)
    new_noise = fits.getdata(out_dir / "noise_map.fits").astype(float)
    legacy_data = fits.getdata(LEGACY_DIR / "data.fits").astype(float)
    legacy_noise = fits.getdata(LEGACY_DIR / "noise_map.fits").astype(float)

    dy, dx = subpixel_offset(legacy_data, new_data)
    print(f"[parity] sub-pixel offset (legacy vs new): dy={dy:.3f}, dx={dx:.3f}")
    new_data_r = nd_shift(np.nan_to_num(new_data), (dy, dx), order=3)
    new_noise_r = nd_shift(np.nan_to_num(new_noise), (dy, dx), order=1)

    bright = legacy_data > 10 * np.nanmedian(legacy_noise)
    data_ratio = new_data_r[bright] / legacy_data[bright]
    noise_ratio = new_noise_r / legacy_noise

    summary = {
        "n_exposures": record["acquire"]["n_exposures"],
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
        "correlated_noise_factor_applied": record["noise"]["correlated_noise_factor"],
        "psf_diagnostics": record["psf"],
    }
    print("[parity] ---- production parity ----")
    print(json.dumps(summary, indent=2))
    (out_dir / "parity_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
