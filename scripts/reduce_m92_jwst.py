"""
JWST/NIRCam stellar-field reduction for the STARRED Tier-1b validation
(JWST leg; sibling of reduce_omegacen_wfc3.py / #37).

M92 (NGC 6341) via NIRCam SW F150W, program 1334 (the JWST NIRCam
astrometric/flux calibration field) — a genuinely STELLAR, deep (1245 s) field.
F150W SW is *undersampled* at 0.03"/pix (PSF FWHM ~1.7 px), the regime the #35
adversarial test flagged for STARRED broadening — so this stresses STARRED
where JWST is hardest, and parallels the extragalactic COSMOS-Web F115W (SW)
that could not be tested (galaxies, not stars).

Run:  ~/venv/PyAuto/bin/python scripts/reduce_m92_jwst.py
Network + jwst pipeline (calwebb_image3) + CRDS required; unit tests never import this.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from autoreduce import TargetSpec, reduce_target  # noqa: E402
from autoreduce.instruments import nircam_adapter_for_filter  # noqa: E402

# M92 (NGC 6341) cluster centre.
RA, DEC = 259.28079, 43.13594
CACHE_ROOT = REPO / "scripts" / "cache"
OUTPUT_ROOT = REPO / "scripts" / "output"


def spec(band: str) -> TargetSpec:
    adapter = nircam_adapter_for_filter(band)
    return TargetSpec(
        name=f"m92_{band.lower()}",
        ra=RA,
        dec=DEC,
        instrument=adapter.key,
        filter_name=band,
        proposal_ids=("1334",),  # JWST NIRCam calibration field
        final_scale=adapter.recommended_final_scale,  # SW 0.03" / LW 0.06"
        final_pixfrac=1.0,
        cutout_shape=(
            501,
            501,
        ),  # ~15" -> rich stellar sample across the density gradient
    )


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--band",
        default="F150W",
        help="F150W (SW, undersampled) | F277W (LW, well-sampled)",
    )
    s = spec(p.parse_args().band)
    record = reduce_target(s, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)
    summary = {
        "n_exposures": record["acquire"]["n_exposures"],
        "correlated_noise_factor": record["noise"].get("correlated_noise_factor"),
        "psf": record.get("psf"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
