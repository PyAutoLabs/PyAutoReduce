"""
Exposure usability screening (design docs stage 1).

MAST serves failed exposures (e.g. ``EXPFLAG = 'EXCESSIVE DOWNTIME'`` with
``EXPTIME = 0``) alongside the good ones in the same visit; a zero-second
frame carries no science and must not enter combination or per-frame
packaging. Screened at acquire so cached lists that predate this filter are
cleaned on re-use too.
"""

from pathlib import Path
from typing import List, Tuple


def filter_usable_exposures(
    exposures: List[Path],
) -> Tuple[List[Path], List[Path]]:
    """Split exposures into (usable, unusable) by primary-header EXPTIME.

    An explicit non-positive EXPTIME marks a failed exposure. A *missing*
    keyword is left for downstream stages to judge — JWST products carry
    their own exposure accounting and never hit the HST noise recipe that
    needs EXPTIME. Loud if nothing usable survives.
    """
    from astropy.io import fits

    usable, unusable = [], []
    for path in exposures:
        exptime = fits.getheader(path).get("EXPTIME")
        if exptime is not None and float(exptime) <= 0.0:
            unusable.append(path)
        else:
            usable.append(path)
    if not usable:
        raise LookupError(
            f"all {len(exposures)} exposures are unusable (EXPTIME <= 0) — "
            "the visit has no science content"
        )
    return usable, unusable
