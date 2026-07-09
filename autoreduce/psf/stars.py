"""
Star selection for empirical ePSF construction (design doc stage 5, tier 1).

Selection cuts, applied to detections on the *drizzled* mosaic so the
resulting PSF is drizzle-consistent by construction: point-like (DAOFind
sharpness/roundness), unsaturated, uncrowded, away from the mosaic edge and
from the lens itself.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class StarSelection:
    """Cuts for ePSF star candidates; defaults tuned for ACS-like mosaics."""

    detection_sigma: float = 10.0
    fwhm_pix: float = 2.0
    sharp_range: Tuple[float, float] = (0.4, 1.0)
    round_limit: float = 0.3
    saturation_fraction: float = 0.7  # of adapter.saturation_dn, in counts
    min_separation_pix: float = 25.0
    # Must exceed half the ePSF extraction window (psf_full 61 + 20 pad -> 41)
    # or edge stars pass selection only to be dropped at extraction.
    edge_margin_pix: int = 46
    exclusion_radius_pix: float = 50.0  # around the target itself


def reject_crowded(x: np.ndarray, y: np.ndarray, min_separation: float) -> np.ndarray:
    """Boolean mask keeping sources with no neighbour within min_separation."""
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        d2 = (x - x[i]) ** 2 + (y - y[i]) ** 2
        d2[i] = np.inf
        if (d2 < min_separation**2).any():
            keep[i] = False
    return keep


def reject_edges(
    x: np.ndarray, y: np.ndarray, shape: Tuple[int, int], margin: int
) -> np.ndarray:
    ny, nx = shape
    return (
        (x >= margin) & (x < nx - margin) & (y >= margin) & (y < ny - margin)
    )


def reject_near(
    x: np.ndarray, y: np.ndarray, x0: float, y0: float, radius: float
) -> np.ndarray:
    return (x - x0) ** 2 + (y - y0) ** 2 > radius**2


def find_stars(
    sci: np.ndarray,
    selection: StarSelection,
    target_xy: Tuple[float, float],
    peak_max: Optional[float],
):
    """DAOStarFinder detections filtered through every selection cut.

    ``peak_max=None`` disables the saturation cut (surface-brightness-unit
    mosaics where a full-well cap is meaningless; saturated cores arrive
    blanked from the upstream pipeline).
    """
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    # Mosaics carry NaN outside the coverage footprint (JWST especially).
    finite = np.isfinite(sci)
    _, median, std = sigma_clipped_stats(sci[finite], sigma=3.0)
    finder = DAOStarFinder(
        fwhm=selection.fwhm_pix,
        threshold=selection.detection_sigma * std,
        sharplo=selection.sharp_range[0],
        sharphi=selection.sharp_range[1],
        roundlo=-selection.round_limit,
        roundhi=selection.round_limit,
        peakmax=peak_max,
    )
    sources = finder(np.nan_to_num(sci - median), mask=~finite)
    if sources is None or len(sources) == 0:
        return None

    x = np.asarray(sources["xcentroid"], dtype=float)
    y = np.asarray(sources["ycentroid"], dtype=float)
    keep = (
        reject_crowded(x, y, selection.min_separation_pix)
        & reject_edges(x, y, sci.shape, selection.edge_margin_pix)
        & reject_near(x, y, *target_xy, selection.exclusion_radius_pix)
    )
    return sources[keep]
