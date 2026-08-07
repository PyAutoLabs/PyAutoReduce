"""
Drizzle-quality diagnostics reported with every reduction, so the
user-facing ``pixfrac``/``kernel`` dials are auditable per dataset
(design doc stage 3).
"""

import numpy as np


def weight_uniformity(wht: np.ndarray) -> float:
    """
    STScI rule-of-thumb statistic: RMS/median of the (positive) weight map
    over the science region. Values above ~0.2 mean the pixfrac is too small
    for the dither pattern (coverage speckle/holes).
    """
    good = wht[np.isfinite(wht) & (wht > 0.0)]
    if good.size == 0:
        raise ValueError("weight map has no positive pixels — empty coverage")
    return float(good.std() / np.median(good))


WEIGHT_UNIFORMITY_LIMIT = 0.2


def check_weight_uniformity(wht: np.ndarray) -> dict:
    """Compute the diagnostic and its verdict for the provenance record."""
    value = weight_uniformity(wht)
    return {
        "wht_rms_over_median": value,
        "limit": WEIGHT_UNIFORMITY_LIMIT,
        "acceptable": value <= WEIGHT_UNIFORMITY_LIMIT,
    }


# The science region the local diagnostic below interrogates — deliberately
# the same 1.5" as `noise.rms.mask_isolated_bad_pixels`'s
# `protect_radius_arcsec`, because the two guards answer the same question
# ("is the lens itself clean?") about different failure modes.
SCIENCE_RADIUS_ARCSEC = 1.5

# A line whose weight falls below this fraction of the cutout median is
# flagged. Derivation: losing one exposure of N along a column leaves IVM
# weight (N-1)/N, so 0.9 detects a single lost exposure for any N <= 9 —
# exactly the few-exposure regime where the sqrt(N/(N-1)) noise inflation
# is visible (x1.41 at N=2 ... x1.06 at N=9). PROVISIONAL: uncalibrated
# against real reductions, which is why the verdict is recorded and never
# raised (issue #65 leg 2).
LOCAL_WEIGHT_DEFICIT_LIMIT = 0.9


def local_weight_deficit(
    wht: np.ndarray,
    center_xy,
    pixel_scale: float,
    radius_arcsec: float = SCIENCE_RADIUS_ARCSEC,
) -> dict:
    """
    Local coverage deficit inside the science region (issue #65 leg 2).

    `weight_uniformity` is a *global* RMS/median over the whole cutout, so a
    handful of degraded columns cannot move it (the slacs0008 spike measured
    0.066 against a 0.2 limit); `mask_isolated_bad_pixels` only sees weight
    that has gone non-finite or non-positive. Neither can see the failure
    mode this reports: coverage that is finite and positive but materially
    *reduced* along a line running through the lens — the ACS/WFC stripe
    class, where rejecting hot/warm pixels on column-organised CCD defects
    drops one exposure's contribution along a column rather than zeroing it.

    Reports, as fractions of the cutout's median positive weight:

    - ``science_median_ratio`` — median weight inside `radius_arcsec`. Catches
      a uniformly-degraded science region.
    - ``min_line_ratio`` — the worst row/column median, over lines crossing
      the science region. Catches the stripe: a single degraded column barely
      shifts the region median but drives this down to (N-1)/N.

    Both axes are tested because a detector-column defect lands on an image
    column only for a north-up frame at the detector's own orientation; after
    drizzling to a common sky frame it can run either way.
    """
    wht = np.asarray(wht, dtype=float)
    if wht.ndim != 2:
        raise ValueError(f"weight map must be 2D, got shape {wht.shape}")
    good = np.isfinite(wht) & (wht > 0.0)
    if not good.any():
        raise ValueError("weight map has no positive pixels — empty coverage")
    median_all = float(np.median(wht[good]))

    cx, cy = center_xy
    ys, xs = np.indices(wht.shape)
    r_arcsec = np.hypot(ys - cy, xs - cx) * pixel_scale
    science = r_arcsec < radius_arcsec
    if not science.any():
        raise ValueError(
            f"science region (r < {radius_arcsec}\") falls outside the weight "
            f"map — centre {center_xy} is off the cutout"
        )

    # Zero/non-finite weight inside the science region is a *total* loss, not
    # a deficit; report it as ratio 0 rather than dropping it from the median,
    # so a hole cannot masquerade as clean coverage.
    science_weights = np.where(good, wht, 0.0)[science]
    science_median_ratio = float(np.median(science_weights) / median_all)

    min_line_ratio = np.inf
    min_line_axis, min_line_index = None, None
    for axis, name in ((0, "column"), (1, "row")):
        # Lines are indexed along the *other* axis: axis=0 collapses rows, so
        # each entry is one image column.
        crossed = np.flatnonzero(science.any(axis=axis))
        for index in crossed:
            line = science.take(index, axis=1 - axis)
            values = np.where(good, wht, 0.0).take(index, axis=1 - axis)[line]
            ratio = float(np.median(values) / median_all)
            if ratio < min_line_ratio:
                min_line_ratio, min_line_axis, min_line_index = ratio, name, int(index)

    return {
        "science_median_ratio": science_median_ratio,
        "min_line_ratio": min_line_ratio,
        "min_line_axis": min_line_axis,
        "min_line_index": min_line_index,
        "radius_arcsec": radius_arcsec,
        "n_science_pixels": int(science.sum()),
    }


def check_local_weight_deficit(
    wht: np.ndarray,
    center_xy,
    pixel_scale: float,
    radius_arcsec: float = SCIENCE_RADIUS_ARCSEC,
) -> dict:
    """
    Compute the local deficit and its verdict for the provenance record.

    **Records, never raises.** The limit is provisional (see
    `LOCAL_WEIGHT_DEFICIT_LIMIT`) and has not been calibrated against real
    reductions, so a fatal guard here would refuse datasets that are very
    likely fine. `acceptable=False` in `reduction.json` is the signal; the
    control test for issue #65 leg 1 is what calibrates the limit.
    """
    stats = local_weight_deficit(wht, center_xy, pixel_scale, radius_arcsec)
    worst = min(stats["science_median_ratio"], stats["min_line_ratio"])
    return {
        **stats,
        "limit": LOCAL_WEIGHT_DEFICIT_LIMIT,
        "acceptable": worst >= LOCAL_WEIGHT_DEFICIT_LIMIT,
    }
