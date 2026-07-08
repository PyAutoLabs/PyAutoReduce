"""
Packaging: WCS-correct cutouts (design doc stage 6).

Deviation from the legacy datasets, deliberately: cutout headers keep the
WCS, units and exposure metadata the legacy stripped-header files lost.
"""

from pathlib import Path
from typing import Tuple

import numpy as np


def cutout_to_fits(
    data: np.ndarray,
    header,
    ra: float,
    dec: float,
    shape: Tuple[int, int],
    out_path: Path,
    extra_header: dict = None,
) -> np.ndarray:
    """Cut `shape` around (ra, dec) and write with an intact cutout WCS."""
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS

    coord = SkyCoord(ra, dec, unit="deg")
    cut = Cutout2D(data, coord, shape, wcs=WCS(header), mode="strict")

    out_header = cut.wcs.to_header()
    for key in ("BUNIT", "EXPTIME", "TEXPTIME", "FILTER", "INSTRUME", "TELESCOP"):
        if key in header:
            out_header[key] = header[key]
    if extra_header:
        for key, value in extra_header.items():
            out_header[key] = value

    fits.PrimaryHDU(cut.data.astype(np.float32), header=out_header).writeto(
        out_path, overwrite=True
    )
    return cut.data
