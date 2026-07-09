"""
Packaging: WCS-correct cutouts (design doc stage 6).

Deviation from the legacy datasets, deliberately: cutout headers keep the
WCS, units and exposure metadata the legacy stripped-header files lost.
"""

from pathlib import Path
from typing import Tuple

import numpy as np


def make_cutout(
    data: np.ndarray,
    header,
    ra: float,
    dec: float,
    shape: Tuple[int, int],
):
    """Cut `shape` around (ra, dec); returns (cut_data, out_header, center_xy)."""
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS

    coord = SkyCoord(ra, dec, unit="deg")
    cut = Cutout2D(data, coord, shape, wcs=WCS(header), mode="strict")

    out_header = cut.wcs.to_header()
    for key in ("BUNIT", "EXPTIME", "TEXPTIME", "FILTER", "INSTRUME", "TELESCOP"):
        if key in header:
            out_header[key] = header[key]
    center = cut.wcs.world_to_pixel_values(ra, dec)
    return cut.data, out_header, (float(center[0]), float(center[1]))


def write_fits(data: np.ndarray, header, out_path: Path) -> None:
    from astropy.io import fits

    fits.PrimaryHDU(data.astype(np.float32), header=header).writeto(
        out_path, overwrite=True
    )


def cutout_to_fits(
    data: np.ndarray,
    header,
    ra: float,
    dec: float,
    shape: Tuple[int, int],
    out_path: Path,
    extra_header: dict = None,
) -> np.ndarray:
    """Cut and write in one step (kept for callers with no post-processing)."""
    cut_data, out_header, _ = make_cutout(data, header, ra, dec, shape)
    if extra_header:
        for key, value in extra_header.items():
            out_header[key] = value
    write_fits(cut_data, out_header, out_path)
    return cut_data
