"""
Validation (issue #17): reduce slacs1430+4105 (SDSS J1430+4105) through the
*production* pipeline — program 10886 (SLACS ACS snapshot, obs j9op36010,
F814W, 2128s) confirmed via MAST at (217.517083, +41.099194) — then compare
the products against the legacy modeling dataset with sub-pixel registration.

Second lens of the design doc's "Validation — SLACS parity study"
(docs/design/hst_acs_pipeline.md); slacs0008-0004 (issue #2) was the first.

Run:  ~/venv/PyAuto/bin/python scripts/reduce_slacs1430.py
Network + drizzlepac required; unit tests never import this.
"""

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from autoreduce import TargetSpec, reduce_target  # noqa: E402

LEGACY_DIR = Path("/mnt/c/Users/Jammy/Science/subhalo/dataset/slacs/slacs1430+4105")
CACHE_ROOT = REPO / "scripts" / "cache"
OUTPUT_ROOT = REPO / "scripts" / "output"

SPEC = TargetSpec(
    name="slacs1430+4105",
    ra=217.517083,
    dec=41.099194,
    proposal_ids=("10886",),
)


def main():
    record = reduce_target(SPEC, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)
    print(json.dumps(record["drizzle"]["weight_uniformity"], indent=2))

    from astropy.io import fits

    from autoreduce.validation import registered_ratios

    out_dir = OUTPUT_ROOT / SPEC.name
    new_data = fits.getdata(out_dir / "data.fits").astype(float)
    new_noise = fits.getdata(out_dir / "noise_map.fits").astype(float)
    legacy_data = fits.getdata(LEGACY_DIR / "data.fits").astype(float)
    legacy_noise = fits.getdata(LEGACY_DIR / "noise_map.fits").astype(float)

    summary = {
        "n_exposures": record["acquire"]["n_exposures"],
        **registered_ratios(new_data, new_noise, legacy_data, legacy_noise),
        "correlated_noise_factor_applied": record["noise"]["correlated_noise_factor"],
        "psf_diagnostics": record["psf"],
    }
    print("[parity] ---- production parity ----")
    print(json.dumps(summary, indent=2))
    (out_dir / "parity_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
