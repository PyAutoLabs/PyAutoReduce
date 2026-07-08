"""
JWST noise stage (roadmap phase 3): *read, don't construct*.

The resampled ``_i2d`` product already carries a per-pixel total-error map
(ERR: Poisson + read noise + flat, propagated and resampled by the jwst
pipeline), so stage 4 reads it rather than rebuilding it from weights — and
then applies the same correlated-noise factor R the HST path uses, since
resample correlates neighbouring pixels exactly as drizzle does and the
propagated ERR is a per-pixel quantity that underestimates the effective
noise a lens-model chi^2 sees.

A consistency check against the empirical blank-sky RMS of the mosaic is
recorded in provenance; large disagreement means the upstream error model
and the sky disagree and must be investigated, not absorbed.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .rms import empirical_background_rms


def noise_map_from_error(
    err: np.ndarray,
    sci: np.ndarray,
    correlated_noise_factor: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    """RMS map from a propagated ERR array; NaN/zero stay NaN (loud later)."""
    if err.shape != sci.shape:
        raise ValueError(f"shape mismatch: err {err.shape} vs sci {sci.shape}")
    if correlated_noise_factor < 1.0:
        raise ValueError(
            f"correlated-noise factor must be >= 1: {correlated_noise_factor}"
        )
    noise = np.where(
        np.isfinite(err) & (err > 0.0), correlated_noise_factor * err, np.nan
    )

    sky_rms = empirical_background_rms(sci[np.isfinite(noise)])
    err_floor = float(np.nanpercentile(noise, 5)) / correlated_noise_factor
    consistency = {
        "empirical_sky_rms": sky_rms,
        "err_5th_percentile_pre_R": err_floor,
        "sky_over_err_floor": sky_rms / err_floor if err_floor > 0 else float("inf"),
    }
    return noise, consistency
