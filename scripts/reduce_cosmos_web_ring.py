"""
JWST integration + acceptance (issue #6): the COSMOS-Web ring, four bands.

Reduces the ring (RA 150.10048, +1.89301; Mercier et al. 2024) from MAST
level-2 ``_cal`` exposures through calwebb_image3, then compares data/noise
against the autolens_assistant demo dataset for that band (sub-pixel
registered ratios — the SLACS parity method). SW bands (F115W/F150W) output
0.03"/pix; LW (F277W/F444W) 0.06"/pix, matching the demo convention.

Run:  ~/venv/PyAuto/bin/python scripts/reduce_cosmos_web_ring.py --band F444W
Network + jwst pipeline required; unit tests never import this.

PSF back-end (issue #35): ``--psf-backend starred`` selects the optional
Tier-1b STARRED super-sampled ePSF instead of the photutils Tier-1 default.
STARRED is GPL/JAX and shipped as an extra — install ``pyautoreduce[starred]``
(it coexists with the reduction stack; autoreduce has no astropy/scipy caps).
"""

import argparse
import json
import sys
from pathlib import Path

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


def spec_for(band: str, psf_backend: str = "epsf") -> TargetSpec:
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
        psf_backend=psf_backend,  # "epsf" (default) | "starred" (Tier-1b, #35)
    )


def compare(band: str, out_dir: Path) -> dict:
    from astropy.io import fits

    from autoreduce.validation import registered_ratios

    new_data = fits.getdata(out_dir / "data.fits").astype(float)
    new_noise = fits.getdata(out_dir / "noise_map.fits").astype(float)
    demo_dir = DEMO_ROOT / band
    demo_data = fits.getdata(demo_dir / "data.fits").astype(float)
    demo_noise = fits.getdata(demo_dir / "noise_map.fits").astype(float)
    return {
        "band": band,
        **registered_ratios(new_data, new_noise, demo_data, demo_noise),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", required=True, choices=[*BANDS, "all"])
    parser.add_argument(
        "--psf-backend", default="epsf", choices=["epsf", "starred"],
        help="PSF back-end: photutils Tier-1 (default) or STARRED Tier-1b (#35; "
        "needs the pyautoreduce[starred] extra)",
    )
    args = parser.parse_args()
    bands = BANDS if args.band == "all" else (args.band,)

    for band in bands:
        spec = spec_for(band, psf_backend=args.psf_backend)
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
