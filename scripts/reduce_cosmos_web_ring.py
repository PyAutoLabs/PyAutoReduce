"""
JWST integration + acceptance (issue #6): the COSMOS-Web ring, four bands.

Reduces the ring (RA 150.10048, +1.89301; Mercier et al. 2024) from MAST
level-2 ``_cal`` exposures through calwebb_image3, then compares data/noise
against the autolens_assistant demo dataset for that band (sub-pixel
registered ratios — the SLACS parity method). SW bands (F115W/F150W) output
0.03"/pix; LW (F277W/F444W) 0.06"/pix, matching the demo convention.

Run:  ~/venv/PyAuto/bin/python scripts/reduce_cosmos_web_ring.py --band F444W
Network + jwst pipeline required; unit tests never import this.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from autoreduce import TargetSpec, reduce_target  # noqa: E402
from autoreduce.instruments import nircam_adapter_for_filter  # noqa: E402

RA, DEC = 150.10048, 1.89301
CACHE_ROOT = REPO / "scripts" / "cache"
OUTPUT_ROOT = REPO / "scripts" / "output"
DEMO_ROOT = Path(
    "/home/jammy/Code/PyAutoLabs/autolens_assistant/dataset/imaging/"
    "cosmos_web_ring/wavebands"
)

BANDS = ("F115W", "F150W", "F277W", "F444W")


def spec_for(band: str) -> TargetSpec:
    adapter = nircam_adapter_for_filter(band)
    # Demo cutouts: SW 419x419 @0.03 (12.57"), LW 209x209 @0.06 (12.54") —
    # match the demo shapes exactly so parity is pixel-to-pixel.
    shape = (419, 419) if adapter.key == "nircam_sw" else (209, 209)
    return TargetSpec(
        name=f"cosmos_web_ring_{band.lower()}",
        ra=RA,
        dec=DEC,
        instrument=adapter.key,
        filter_name=band,
        # COSMOS-Web only: the demo parity products are built from program
        # 1727 mosaics; other programs at these coords (e.g. 5893) would
        # change the depth and skew the noise parity.
        proposal_ids=("1727",),
        final_scale=adapter.recommended_final_scale,
        final_pixfrac=1.0,  # COSMOS-Web mosaics use the full drop
        cutout_shape=shape,
    )


def subpixel_offset(a, b):
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


def compare(band: str, out_dir: Path) -> dict:
    from astropy.io import fits
    from scipy.ndimage import shift as nd_shift

    new_data = fits.getdata(out_dir / "data.fits").astype(float)
    new_noise = fits.getdata(out_dir / "noise_map.fits").astype(float)
    demo_dir = DEMO_ROOT / band
    demo_data = fits.getdata(demo_dir / "data.fits").astype(float)
    demo_noise = fits.getdata(demo_dir / "noise_map.fits").astype(float)

    if new_data.shape != demo_data.shape:
        raise ValueError(
            f"{band}: shape mismatch new {new_data.shape} vs demo {demo_data.shape}"
        )

    dy, dx = subpixel_offset(demo_data, new_data)
    new_data_r = nd_shift(np.nan_to_num(new_data), (dy, dx), order=3)
    new_noise_r = nd_shift(np.nan_to_num(new_noise), (dy, dx), order=1)

    bright = demo_data > 10 * np.nanmedian(demo_noise)
    data_ratio = new_data_r[bright] / demo_data[bright]
    noise_ratio = new_noise_r / demo_noise
    return {
        "band": band,
        "offset": [float(dy), float(dx)],
        "n_bright": int(bright.sum()),
        "data_ratio_median": float(np.nanmedian(data_ratio)),
        "noise_ratio_median": float(np.nanmedian(noise_ratio)),
        "noise_ratio_16_84": [
            float(np.nanpercentile(noise_ratio, 16)),
            float(np.nanpercentile(noise_ratio, 84)),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", required=True, choices=[*BANDS, "all"])
    args = parser.parse_args()
    bands = BANDS if args.band == "all" else (args.band,)

    for band in bands:
        spec = spec_for(band)
        record = reduce_target(spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)
        out_dir = OUTPUT_ROOT / spec.name
        summary = {
            "n_exposures": record["acquire"]["n_exposures"],
            "weight_uniformity": record["drizzle"]["weight_uniformity"],
            "weight_uniformity_cutout": record["drizzle"].get("weight_uniformity_cutout"),
            "correlated_noise_factor": record["noise"]["correlated_noise_factor"],
            "sky_over_err_floor": record["noise"].get("sky_over_err_floor"),
            "psf": record["psf"],
            "parity": compare(band, out_dir),
        }
        print(f"[{band}] ---- validation ----")
        print(json.dumps(summary, indent=2))
        (out_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
