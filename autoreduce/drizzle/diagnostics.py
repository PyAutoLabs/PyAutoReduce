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
