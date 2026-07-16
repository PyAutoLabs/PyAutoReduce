"""
The cutout-domain branch: fetch -> package (docs/design/surveys.md).

Products are colour context for lenses whose modeling data carries no
optical counterpart (especially ALMA targets), not modeling inputs:
``data.fits`` per band, ``noise_map.fits`` only where the service ships
variance, no PSF in this phase. The provenance record states explicitly
what was NOT produced and why, so a survey cutout can never masquerade
as a modeling-ready dataset.
"""

from pathlib import Path
from typing import Dict

import numpy as np

from ..instruments import SurveyCutoutAdapter
from ..target import TargetSpec
from . import fetch as fetch_mod


def rms_from_invvar(ivar: np.ndarray) -> np.ndarray:
    """RMS map from inverse variance; non-positive/non-finite stay NaN."""
    good = np.isfinite(ivar) & (ivar > 0.0)
    if not good.any():
        raise ValueError("inverse-variance map has no positive pixels")
    out = np.full(ivar.shape, np.nan)
    out[good] = 1.0 / np.sqrt(ivar[good])
    return out


def reduce_survey_target(
    spec: TargetSpec, adapter: SurveyCutoutAdapter, out_dir: Path
) -> Dict:
    """Fetch the target's cutouts for every band; write per-band products."""
    from astropy.io import fits

    bands = spec.survey_bands or adapter.bands
    size = max(spec.cutout_shape)
    fetcher = fetch_mod.FETCHERS[adapter.observatory]
    fetched = fetcher(spec.ra, spec.dec, size, tuple(bands))

    products = []
    for band, payload in fetched.items():
        band_dir = out_dir / adapter.key / band
        band_dir.mkdir(parents=True, exist_ok=True)
        header = payload["header"]
        header["SURVEY"] = (adapter.key, "survey cutout (colour context)")
        header["BAND"] = band
        fits.PrimaryHDU(
            payload["data"].astype(np.float32), header=header
        ).writeto(band_dir / "data.fits", overwrite=True)
        products.append(f"{adapter.key}/{band}/data.fits")
        if adapter.noise_available:
            noise = rms_from_invvar(payload["ivar"])
            fits.PrimaryHDU(
                noise.astype(np.float32), header=header
            ).writeto(band_dir / "noise_map.fits", overwrite=True)
            products.append(f"{adapter.key}/{band}/noise_map.fits")

    record = {
        "target": spec.as_dict(),
        "instrument": adapter.key,
        "acquire": {
            "source": adapter.observatory,
            "bands_requested": list(bands),
            "bands_delivered": sorted(fetched),
        },
        "package": {
            "products": products,
            "pixel_scale": adapter.native_scale,
            "products_optional": {
                "noise_map": (
                    "from service inverse variance"
                    if adapter.noise_available
                    else "not produced — service ships no variance product"
                ),
                "psf": (
                    "not produced — colour-context products carry no PSF "
                    "(docs/design/surveys.md phase 1)"
                ),
            },
        },
    }
    return record
