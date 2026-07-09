"""
Tier-1 empirical ePSF (design doc stage 5): Anderson & King-style effective
PSF built from selected stars on the drizzled mosaic via photutils.

Emits the two modeling kernels — compact (`psf.fits`) and extended
(`psf_full.fits`) — odd-shaped, centred, unit-normalised, plus diagnostics
for the provenance record. If too few stars survive selection the build
fails loudly; tier 2 (model-PSF fallback) is a deliberate choice recorded in
provenance, never a silent degradation.
"""

from typing import Dict, Tuple

import numpy as np

MIN_STARS = 8


class InsufficientStarsError(RuntimeError):
    """Tier 1 is not viable for this field; choose tier 2 explicitly."""


def normalise_kernel(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Centre-crop to the requested odd shape and normalise to unit sum."""
    if any(s % 2 == 0 for s in shape):
        raise ValueError(f"kernel shape must be odd: {shape}")
    ny, nx = psf.shape
    cy, cx = ny // 2, nx // 2
    hy, hx = shape[0] // 2, shape[1] // 2
    if hy > cy or hx > cx:
        raise ValueError(f"requested shape {shape} exceeds built PSF {psf.shape}")
    cut = psf[cy - hy : cy + hy + 1, cx - hx : cx + hx + 1].astype(np.float64)
    total = cut.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("PSF kernel has non-positive total flux")
    return cut / total


def build_epsf(
    sci: np.ndarray,
    stars_table,
    psf_shape: Tuple[int, int],
    psf_full_shape: Tuple[int, int],
    oversampling: int = 2,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Build the ePSF and return (psf, psf_full, diagnostics)."""
    from astropy.nddata import NDData
    from astropy.table import Table
    from photutils.psf import EPSFBuilder, extract_stars

    if stars_table is None or len(stars_table) < MIN_STARS:
        n = 0 if stars_table is None else len(stars_table)
        raise InsufficientStarsError(
            f"{n} usable stars (< {MIN_STARS}); tier 1 ePSF is not viable — "
            f"select tier 2 (model PSF) explicitly"
        )

    positions = Table(
        {"x": stars_table["xcentroid"], "y": stars_table["ycentroid"]}
    )
    # Extraction window comfortably larger than the extended kernel.
    size = max(psf_full_shape) + 10
    if size % 2 == 0:
        size += 1
    stars = extract_stars(NDData(sci), positions, size=size)

    # Reject stars whose window contains non-finite pixels (coverage edges,
    # DQ holes — routine in JWST mosaics): EPSFBuilder's fitter refuses NaN.
    from photutils.psf import EPSFStars

    stars = EPSFStars([s for s in stars.all_stars if np.isfinite(s.data).all()])

    # extract_stars drops stars whose window overruns the mosaic edge and the
    # finite cut above drops more, so the minimum-star contract must be
    # re-checked on what actually survived.
    if len(stars) < MIN_STARS:
        raise InsufficientStarsError(
            f"{len(stars)} stars survived cutout extraction + finite-window "
            f"cut (< {MIN_STARS}); tier 1 ePSF is not viable — select tier 2 "
            f"(model PSF) explicitly"
        )

    builder = EPSFBuilder(oversampling=oversampling, maxiters=10, progress_bar=False)
    epsf_model, fitted = builder(stars)

    # Evaluate the oversampled model back onto the native mosaic pixel grid.
    full = _evaluate_native(epsf_model, psf_full_shape)
    psf_full = normalise_kernel(full, psf_full_shape)
    psf = normalise_kernel(full, psf_shape)

    diagnostics = {
        "method": "epsf-tier1",
        "n_stars_used": int(len(fitted)),
        "oversampling": oversampling,
        "fwhm_pix": _fwhm_of(psf),
    }
    return psf, psf_full, diagnostics


def _evaluate_native(epsf_model, shape: Tuple[int, int]) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return epsf_model.evaluate(
        xx, yy, flux=1.0, x_0=shape[1] // 2, y_0=shape[0] // 2
    )


def _fwhm_of(psf: np.ndarray) -> float:
    """Crude FWHM estimate from the radial profile — a diagnostic, not science."""
    ny, nx = psf.shape
    y, x = np.mgrid[0:ny, 0:nx]
    r = np.hypot(y - ny // 2, x - nx // 2)
    half = psf.max() / 2.0
    above = r[psf >= half]
    return float(2.0 * above.max()) if above.size else float("nan")
