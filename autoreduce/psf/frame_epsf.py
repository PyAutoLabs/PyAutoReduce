"""
Per-frame native-pixel ePSFs for the frame-products mode (issue #21).

The frames the mode packages are undrizzled — their PSF lives in native,
geometrically distorted chip pixels, so the mosaic's drizzled ePSF is the
wrong kernel for fitting them. Tier 1 builds an ePSF from the frame's own
full chip. DQ-flagged pixels (~3% of a real ACS chip: hot/warm/dead pixels
plus driz_cr cosmic rays) are patched with a local median **in the
estimator's working image only** — the shipped data products are never
touched. Rejection-on-contact is not an option at that flag density (no
stamp window is clean), and the patch also erases flagged cosmic rays from
the candidate pool in multi-exposure visits (single-exposure visits carry
no CR flags; the sharpness/roundness cuts are then the only CR screen —
both cases recorded as `cr_screen` in the diagnostics).

Insufficient stars is a *recorded outcome, not a hard stop* — a deliberate
deviation from the mosaic path's tier-2 escalation: one ~500 s exposure
legitimately may lack the minimum usable stars, and the frame's data
products remain useful without a PSF. The manifest and a loud runtime
notice say so; the TinyTim / model-PSF tier is the roadmap upgrade.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from ..instruments import InstrumentAdapter
from ..target import TargetSpec
from . import epsf as epsf_mod
from . import stars as stars_mod


def _native_peak_max(bunit: str, exptime: float, adapter, selection) -> float:
    """The saturation cap in the frame's own units.

    A star saturates when its rate fills the well within the exposure; the
    frame is a single exposure, so the cap is the fractional full well in
    ELECTRONS, or that divided by EXPTIME for e-/s frames (WFC3/IR).
    """
    cap = selection.saturation_fraction * adapter.saturation_dn
    unit = bunit.strip().upper()
    if unit in ("ELECTRONS", "ELECTRON"):
        return cap
    if unit in ("ELECTRONS/S", "ELECTRON/S", "ELECTRONS/SEC"):
        if not np.isfinite(exptime) or exptime <= 0.0:
            raise ValueError(
                f"cannot form an e-/s saturation cap without a positive "
                f"EXPTIME: {exptime}"
            )
        return cap / exptime
    raise ValueError(
        f"unrecognised frame BUNIT {bunit!r} — expected ELECTRONS or "
        f"ELECTRONS/S for HST calibrated products"
    )


def build_frame_epsf(
    hdul,
    extver: int,
    spec: TargetSpec,
    adapter: InstrumentAdapter,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    Tier-1 ePSF from one calibrated frame's full chip.

    Returns ``(psf, psf_full, diagnostics)``; ``(None, None, diagnostics)``
    when fewer than the minimum usable stars survive — the caller records
    the outcome and ships the frame without PSF products.
    """
    from astropy.wcs import WCS

    sci_hdu = hdul["SCI", extver]
    dq = hdul["DQ", extver].data
    hdr = sci_hdu.header
    primary = hdul[0].header

    # Sky-subtracted working image with DQ-flagged pixels local-median
    # patched — estimator input only (see module docstring); the patch is
    # smooth, so a patched hot pixel or cosmic ray cannot pass the star
    # sharpness cuts, let alone bias a stamp.
    work = sci_hdu.data.astype(float) - float(hdr.get("MDRIZSKY", 0.0))
    bad = dq != 0
    n_patched = int(bad.sum())
    if n_patched:
        from scipy.ndimage import median_filter

        work = np.where(bad, median_filter(work, size=5), work)

    # Exclude stars near the target itself, exactly as the mosaic path does;
    # the full-distortion projection is the same one the cutout uses.
    wcs_full = WCS(hdr, fobj=hdul, naxis=2)
    x, y = wcs_full.world_to_pixel_values(spec.ra, spec.dec)

    selection = stars_mod.StarSelection()
    peak_max = _native_peak_max(
        str(hdr.get("BUNIT", "")),
        float(primary.get("EXPTIME", 0.0)),
        adapter,
        selection,
    )
    found = stars_mod.find_stars(
        work,
        selection,
        target_xy=(float(x), float(y)),
        peak_max=peak_max,
    )

    cr_screen = "DQ-patched" if n_patched else "shape-cuts-only"
    try:
        psf, psf_full, diag = epsf_mod.build_epsf(
            work, found, spec.psf_shape, spec.psf_full_shape
        )
    except epsf_mod.InsufficientStarsError as err:
        return (
            None,
            None,
            {
                "method": "none",
                "reason": str(err),
                "n_candidates": 0 if found is None else int(len(found)),
                "n_patched_pixels": n_patched,
                "cr_screen": cr_screen,
            },
        )
    diag = {
        "method": "epsf-frame-tier1",
        **{k: v for k, v in diag.items() if k != "method"},
        "n_patched_pixels": n_patched,
        "cr_screen": cr_screen,
    }
    return psf, psf_full, diag
