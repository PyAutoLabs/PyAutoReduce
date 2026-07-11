"""
WFC3/UVIS stellar-field reduction for the STARRED Tier-1b science comparison
(PyAutoReduce#37, WFC3 leg).

Omega Centauri (NGC 5139) via WFC3/UVIS F606W, proposal 15733 (6x60s at one
pointing, ~10" off the core) — a genuinely STELLAR, well-sampled field, unlike
the extragalactic lens targets whose "point sources" are galaxies (the #35/#37
finding). Reduces a cutout so the STARRED vs photutils Tier-1 ePSF comparison
can run on REAL stars.

Run:  ~/venv/PyAuto/bin/python scripts/reduce_omegacen_wfc3.py
Network + drizzlepac + CRDS required; unit tests never import this.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from autoreduce import TargetSpec, reduce_target  # noqa: E402

# Mean pointing of the proposal-15733 F606W exposures (MAST s_ra/s_dec).
RA, DEC = 201.69283, -47.47906
CACHE_ROOT = REPO / "scripts" / "cache"
OUTPUT_ROOT = REPO / "scripts" / "output"


def spec() -> TargetSpec:
    return TargetSpec(
        name="omegacen_f606w",
        ra=RA,
        dec=DEC,
        instrument="wfc3_uvis",
        filter_name="F606W",
        proposal_ids=("15733",),  # one clean 6x60s dithered visit
        final_scale=0.0396,  # UVIS adapter recommended
        final_pixfrac=1.0,  # 6-dither snapshot -> full drop for guaranteed coverage
        cutout_shape=(401, 401),  # ~16" -> a rich isolated-star sample
    )


def main():
    s = spec()
    record = reduce_target(s, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)
    summary = {
        "n_exposures": record["acquire"]["n_exposures"],
        "correlated_noise_factor": record["noise"].get("correlated_noise_factor"),
        "psf": record["psf"],
    }
    print(json.dumps(summary, indent=2))
    (OUTPUT_ROOT / s.name / "reduce_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
