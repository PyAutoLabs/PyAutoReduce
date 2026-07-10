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

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..instruments import InstrumentAdapter
from ..target import TargetSpec


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

    from ._common import chdir_scratch, combine_provenance

    product_name = f"{spec.name}_{spec.filter_name}".lower()
    with chdir_scratch(output_dir) as output_dir:
        asn = asn_from_list(
            [str(p) for p in exposures], rule=DMS_Level3_Base,
            product_name=product_name,
        )
        asn_path = output_dir / f"{product_name}_asn.json"
        _, serialized = asn.dump(format="json")
        asn_path.write_text(serialized)

        Image3Pipeline.call(
            str(asn_path),
            output_dir=str(output_dir),
            save_results=True,  # stpipe .call() discards results otherwise
            in_memory=False,  # on-disk models: image3 OOMs this machine otherwise
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
                # The _crf products (outlier-flagged, tweakreg-updated cal
                # frames) are the JWST analogue of driz_cr-flagged _flc files
                # — the frame-products mode packages them (issue #27).
                "outlier_detection": {"save_results": True},
            },
        )

    i2d = output_dir / f"{product_name}_i2d.fits"
    if not i2d.exists():
        raise FileNotFoundError(f"calwebb_image3 did not produce {i2d}")

    crf_paths = sorted(str(p) for p in output_dir.glob("*_crf.fits"))

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

    provenance = combine_provenance(
        spec,
        adapter,
        exposures,
        wht,
        kwargs_key="resample_kwargs",
        kwargs={
            "pixel_scale": spec.final_scale,
            "pixfrac": spec.final_pixfrac,
            "kernel": spec.final_kernel,
            "rotation": 0.0,
        },
        head={"backend": "jwst_image3"},
        tail={
            "err_path": str(paths["err"]),
            "i2d_path": str(i2d),
            "crf_paths": crf_paths,
        },
    )
    return paths["sci"], paths["wht"], provenance
