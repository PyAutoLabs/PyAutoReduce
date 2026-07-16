"""
Survey-cutout spike (issue #50): the cutout domain's real-network
demonstration — fetch slacs0008-0004 colour context from all three
phase-1 services and print a per-band coverage summary.

Run:  ~/venv/PyAuto/bin/python prototypes/survey_cutouts_spike.py
Network required (Legacy viewer, PS1 fitscut, SDSS SAS via astroquery);
writes prototypes/output/survey_cutouts/; unit tests never import this.
"""

import json
from pathlib import Path

import numpy as np

from autoreduce import instruments
from autoreduce.pipeline import reduce_target
from autoreduce.target import TargetSpec

NAME = "slacs0008-0004"
RA, DEC = 2.012333, -0.068944


def main() -> int:
    out_root = Path("prototypes/output/survey_cutouts").resolve()
    summary = {}
    for key in ("legacy_surveys", "sdss", "panstarrs"):
        print(f"== {key} ==")
        try:
            record = reduce_target(
                TargetSpec(name=NAME, ra=RA, dec=DEC, instrument=key,
                           cutout_shape=(101, 101)),
                out_root / "cache",
                out_root / key,
            )
        except Exception as err:  # per-service verdicts, not one crash
            summary[key] = {"error": str(err)}
            print(f"   FAILED: {err}")
            continue
        per_band = {}
        for product in record["package"]["products"]:
            path = out_root / key / NAME / product
            data = None
            if path.name == "data.fits":
                from astropy.io import fits

                data = fits.getdata(path)
                band = path.parent.name
                per_band[band] = {
                    "shape": list(data.shape),
                    "finite_fraction": float(np.isfinite(data).mean()),
                    "max": float(np.nanmax(data)),
                }
        summary[key] = {
            "bands": record["acquire"]["bands_delivered"],
            "products": record["package"]["products"],
            "per_band": per_band,
        }
        print(json.dumps(summary[key], indent=2))
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    failed = [k for k, v in summary.items() if "error" in v]
    print("SURVEYS", "OK" if not failed else f"FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
