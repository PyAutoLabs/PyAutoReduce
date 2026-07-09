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
    """True if any SCI extension's footprint contains (ra, dec) ± margin."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    margin_deg = margin_arcsec / 3600.0
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.name != "SCI" or hdu.data is None:
                continue
            wcs = WCS(hdu.header, naxis=2)
            ny, nx = hdu.data.shape[-2], hdu.data.shape[-1]
            corners = wcs.pixel_to_world_values(
                [0, nx - 1, nx - 1, 0], [0, 0, ny - 1, ny - 1]
            )
            ras = np.asarray(corners[0], dtype=float)
            decs = np.asarray(corners[1], dtype=float)
            if (
                ras.min() - margin_deg <= ra <= ras.max() + margin_deg
                and decs.min() - margin_deg <= dec <= decs.max() + margin_deg
            ):
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
