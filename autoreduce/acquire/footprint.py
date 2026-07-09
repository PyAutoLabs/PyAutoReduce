"""
Detector-footprint filtering (design docs stage 1; added for JWST).

A survey visit's exposures span many detectors, most of which never touch the
target; combining them wastes memory and time (and on this machine, OOMs the
jwst pipeline). Keep only calibrated exposures whose detector footprint
contains the target, with a margin for the cutout and dither pattern.

Uses the approximate FITS WCS every calibrated product carries in its SCI
extension (JWST cal files carry both gwcs and FITS-approx; HST flt/flc carry
FITS WCS) — footprint containment at arcsecond precision, which is all this
filter needs.
"""

from pathlib import Path
from typing import List, Tuple


def covers_target(path: Path, ra: float, dec: float, margin_arcsec: float) -> bool:
    """True if any SCI extension's footprint contains (ra, dec) ± margin.

    Containment is tested in *pixel* space (project the target through the
    extension's WCS, allow a margin in pixels) — immune to the RA-wraparound
    and cos(dec) pitfalls of sky-coordinate bounding boxes.
    """
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.name != "SCI" or hdu.data is None:
                continue
            wcs = WCS(hdu.header, naxis=2)
            ny, nx = hdu.data.shape[-2], hdu.data.shape[-1]
            x, y = wcs.world_to_pixel_values(ra, dec)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            scale_arcsec = float(
                np.mean(proj_plane_pixel_scales(wcs)) * 3600.0
            )
            m = margin_arcsec / scale_arcsec
            if -m <= float(x) <= nx - 1 + m and -m <= float(y) <= ny - 1 + m:
                return True
    return False


def filter_to_target(
    exposures: List[Path], ra: float, dec: float, margin_arcsec: float = 30.0
) -> Tuple[List[Path], List[Path]]:
    """Split exposures into (covering, skipped); loud if nothing covers."""
    covering, skipped = [], []
    for path in exposures:
        (covering if covers_target(path, ra, dec, margin_arcsec) else skipped).append(
            path
        )
    if not covering:
        raise LookupError(
            f"none of {len(exposures)} exposures cover ({ra}, {dec}) — "
            f"wrong coordinates or a query/footprint bug"
        )
    return covering, skipped
