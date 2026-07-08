"""
JWST combination backend (roadmap phase 3): calwebb_image3 — the drizzle
analogue (tweakreg / skymatch / outlier_detection / resample). Defaults-first:
the pipeline runs with its own defaults; only the lensing dials map through
(output pixel scale, pixfrac, kernel, north-up rotation, IVM weighting).

The ``_i2d`` product is multi-extension (SCI/ERR/CON/WHT/VAR_*); this module
normalizes it to the package's internal contract — standalone ``sci``/``wht``/
``err`` FITS files with the WCS and an EXPTIME key — so every downstream
stage (noise, psf, package) is backend-agnostic.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..instruments import InstrumentAdapter
from ..target import TargetSpec
from .diagnostics import check_weight_uniformity
from ..noise.rms import casertano_r


def combine(
    exposures: List[Path],
    spec: TargetSpec,
    adapter: InstrumentAdapter,
    output_dir: Path,
) -> Tuple[Path, Path, Dict]:
    """Run calwebb_image3; return (sci_path, wht_path, provenance_fragment)."""
    from astropy.io import fits
    from jwst.associations.asn_from_list import asn_from_list
    from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
    from jwst.pipeline import Image3Pipeline

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    product_name = f"{spec.name}_{spec.filter_name}".lower()

    asn = asn_from_list(
        [str(p) for p in exposures], rule=DMS_Level3_Base, product_name=product_name
    )
    asn_path = output_dir / f"{product_name}_asn.json"
    _, serialized = asn.dump(format="json")
    asn_path.write_text(serialized)

    import os

    cwd = os.getcwd()
    os.chdir(output_dir)
    try:
        Image3Pipeline.call(
            str(asn_path),
            output_dir=str(output_dir),
            steps={
                "resample": {
                    "pixel_scale": spec.final_scale,
                    "pixfrac": spec.final_pixfrac,
                    "kernel": spec.final_kernel,
                    "rotation": 0.0,
                    "weight_type": adapter.default_drizzle_kwargs.get(
                        "weight_type", "ivm"
                    ),
                },
            },
        )
    finally:
        os.chdir(cwd)

    i2d = output_dir / f"{product_name}_i2d.fits"
    if not i2d.exists():
        raise FileNotFoundError(f"calwebb_image3 did not produce {i2d}")

    # Normalize to the internal contract: standalone sci/wht/err files.
    with fits.open(i2d) as hdul:
        sci = hdul["SCI"].data.astype(np.float32)
        err = hdul["ERR"].data.astype(np.float32)
        wht = hdul["WHT"].data.astype(np.float32)
        header = hdul["SCI"].header.copy()
        header["EXPTIME"] = hdul[0].header.get(
            "XPOSURE", hdul["SCI"].header.get("XPOSURE", 0.0)
        )
        header["BUNIT"] = hdul["SCI"].header.get("BUNIT", "MJy/sr")

    paths = {}
    for name, data in (("sci", sci), ("wht", wht), ("err", err)):
        path = output_dir / f"{product_name}_{name}.fits"
        fits.PrimaryHDU(data, header=header).writeto(path, overwrite=True)
        paths[name] = path

    provenance = {
        "backend": "jwst_image3",
        "n_exposures": len(exposures),
        "exposures": [Path(p).name for p in exposures],
        "single_exposure_branch": len(exposures) == 1,
        "resample_kwargs": {
            "pixel_scale": spec.final_scale,
            "pixfrac": spec.final_pixfrac,
            "kernel": spec.final_kernel,
            "rotation": 0.0,
        },
        "correlated_noise_factor": casertano_r(
            spec.final_pixfrac, adapter.scale_ratio(spec.final_scale)
        ),
        "weight_uniformity": check_weight_uniformity(wht),
        "err_path": str(paths["err"]),
        "i2d_path": str(i2d),
    }
    return paths["sci"], paths["wht"], provenance
