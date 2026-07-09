"""
WFC3 integration + acceptance (issue #4): SDSS J0252+0039.

--channel uvis : F390W through wfc3_uvis at the published Bayer et al.
                 (arXiv:1803.05952) dials — 0.0396"/pix, pixfrac 1.0 — then
                 check our products against their published numbers (units
                 e-/s; sigma_sky ~ 0.002 e-/s; noise closure; R accounting).
--channel ir   : discover WFC3/IR imaging at the same coordinates via MAST
                 and reduce it at the adapter-recommended 0.065"/pix;
                 internal validation only (no published anchor).

Run:  ~/venv/PyAuto/bin/python scripts/reduce_j0252_wfc3.py --channel uvis
Network + drizzlepac required; unit tests never import this.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from autoreduce import TargetSpec, reduce_target  # noqa: E402
from autoreduce import instruments  # noqa: E402

# SDSS J0252+0039: 02h52m45.21s +00d39m58.4s (legacy dataset info.json)
RA, DEC = 43.188375, 0.666222
CACHE_ROOT = REPO / "scripts" / "cache"
OUTPUT_ROOT = REPO / "scripts" / "output"

BAYER_SIGMA_SKY = 0.002  # e-/s, published for the F390W reduction


def spec_for(channel: str) -> TargetSpec:
    if channel == "uvis":
        return TargetSpec(
            name="j0252+0039_f390w",
            ra=RA,
            dec=DEC,
            instrument="wfc3_uvis",
            filter_name="F390W",
            final_scale=0.0396,   # Bayer dial
            final_pixfrac=1.0,    # Bayer dial
        )
    if channel == "ir":
        filter_name = discover_ir_filter()
        return TargetSpec(
            name=f"j0252+0039_{filter_name.lower()}",
            ra=RA,
            dec=DEC,
            instrument="wfc3_ir",
            filter_name=filter_name,
            final_scale=instruments.get("wfc3_ir").recommended_final_scale,
            # Few-dither snapshot on a half-native grid: pixfrac 0.8 leaves
            # zero-weight speckle (the finite-noise guard caught it); the
            # full drop closes coverage at the cost of a larger R — the
            # dial trade-off working as documented.
            final_pixfrac=1.0,
            # 281 px at 0.065" spans 18.3" — far more sky than the ACS-era
            # 14" footprint, and it clips a zero-coverage detector-defect
            # blob 8.5" from the lens (the guard refused it). Match the ACS
            # sky footprint instead: 14" / 0.065 -> 215 px.
            cutout_shape=(215, 215),
        )
    raise ValueError(channel)


def discover_ir_filter() -> str:
    """Find which WFC3/IR filter (if any) covers the target."""
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Observations

    from autoreduce.acquire.mast import select_observations

    obs = Observations.query_criteria(
        coordinates=SkyCoord(RA, DEC, unit="deg"),
        radius="0.5 arcmin",
        obs_collection="HST",
        instrument_name="WFC3/IR",
        dataproduct_type="image",
    )
    direct = select_observations(obs)
    if not direct:
        sys.exit(
            "[ir] no direct WFC3/IR observations at J0252+0039 — the IR leg "
            "needs a different target (parked question on issue #4)"
        )
    # HAP composite rows carry the pseudo-filter 'detection'; never reduce it.
    filters = sorted(
        {str(row["filters"]) for row in direct} - {"detection"}
    )
    if not filters:
        sys.exit("[ir] only HAP composite products found — no real filter")
    print(f"[ir] direct WFC3/IR observations found; filters: {filters}")
    preferred = [f for f in filters if f == "F160W"] or filters
    return preferred[0]


def validate(channel: str, record: dict, out_dir: Path):
    from astropy.io import fits

    noise = fits.getdata(out_dir / "noise_map.fits").astype(float)
    data = fits.getdata(out_dir / "data.fits").astype(float)
    with fits.open(out_dir / "data.fits") as hdul:
        bunit = hdul[0].header.get("BUNIT", "unknown")

    r_factor = record["noise"]["correlated_noise_factor"]
    sky_rms = record["noise"]["empirical_background_rms"]
    summary = {
        "channel": channel,
        "n_exposures": record["acquire"]["n_exposures"],
        "bunit": bunit,
        "weight_uniformity": record["drizzle"]["weight_uniformity"],
        "correlated_noise_factor": r_factor,
        "empirical_sky_rms_cps": sky_rms,
        "noise_map_min_median": [float(np.nanmin(noise)), float(np.nanmedian(noise))],
        "psf": record["psf"],
    }
    if channel == "uvis":
        # Published anchor: Bayer et al. sigma_sky ~ 0.002 e-/s at these dials.
        # Our empirical sky RMS is pre-R; theirs enters sigma_n pre-realization.
        summary["bayer_sigma_sky_cps"] = BAYER_SIGMA_SKY
        summary["sky_rms_over_bayer"] = sky_rms / BAYER_SIGMA_SKY
        # Noise-floor closure: in blank sky the map should approach R * sky_rms.
        summary["noise_floor_over_R_times_sky"] = float(
            np.nanpercentile(noise, 5) / (r_factor * sky_rms)
        )
    print(f"[{channel}] ---- validation ----")
    print(json.dumps(summary, indent=2))
    (out_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True, choices=["uvis", "ir"])
    args = parser.parse_args()

    spec = spec_for(args.channel)
    record = reduce_target(spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)
    validate(args.channel, record, OUTPUT_ROOT / spec.name)


if __name__ == "__main__":
    main()
