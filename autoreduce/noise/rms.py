"""
RMS noise-map construction (design doc stage 4).

    sigma_i = R * sqrt( N_i / t_exp + 1 / W_i )

with N_i the (sky-subtracted, floored-at-zero) source counts/s in pixel i,
t_exp the total exposure time, W_i the IVM drizzle weight (inverse background
variance), and R the Casertano et al. (2000) / DrizzlePac-handbook correction
for the noise correlation the drizzle kernel introduces. The spike validated
applying R: the legacy SLACS noise maps are consistent with it
(parity appendix, docs/design/hst_acs_pipeline.md).
"""

import numpy as np


def casertano_r(pixfrac: float, scale_ratio: float) -> float:
    """
    Correlated-noise correction factor R = 1/r.

    r is the variance-reduction factor of Casertano et al. (2000) for drizzle
    drop size ``pixfrac`` (p) onto an output grid ``scale_ratio`` (s) times
    the native pixel. Smaller p reduces correlation (R -> 1 in the
    interlacing limit); p = 1 at s = 1 is shift-and-add (R = 1.5).
    """
    if not 0.0 < pixfrac <= 1.0:
        raise ValueError(f"pixfrac must be in (0, 1]: {pixfrac}")
    if scale_ratio <= 0.0:
        raise ValueError(f"scale_ratio must be positive: {scale_ratio}")
    p, s = pixfrac, scale_ratio
    if s < p:
        # Finer output grid than the drop: correlation grows without bound
        # as s -> 0 (r -> 0, R -> inf); continuous with the s >= p branch
        # at s = p (r = 2/3).
        r = (s / p) * (1.0 - s / (3.0 * p))
    else:
        r = 1.0 - p / (3.0 * s)
    return 1.0 / r


def noise_map_from(
    sci: np.ndarray,
    wht: np.ndarray,
    exptime: float,
    correlated_noise_factor: float = 1.0,
) -> np.ndarray:
    """
    Per-pixel RMS from a cps science mosaic and its IVM weight map.

    Zero/negative/NaN weights are propagated as NaN; callers packaging a
    cutout must fail loudly if any land inside it (`assert_finite_within`),
    never patch them silently.
    """
    if sci.shape != wht.shape:
        raise ValueError(f"shape mismatch: sci {sci.shape} vs wht {wht.shape}")
    if not np.isfinite(exptime) or exptime <= 0.0:
        raise ValueError(f"exptime must be positive and finite: {exptime}")
    if correlated_noise_factor < 1.0:
        raise ValueError(
            f"correlated-noise factor must be >= 1: {correlated_noise_factor}"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        var_bkg = np.where(wht > 0.0, 1.0 / wht, np.nan)
    var_src = np.clip(sci, 0.0, None) / exptime
    return correlated_noise_factor * np.sqrt(var_src + var_bkg)


def assert_finite_within(noise_map: np.ndarray, region_name: str) -> None:
    """Loud failure if the noise map carries NaN/inf/zero inside a cutout."""
    bad = ~np.isfinite(noise_map) | (noise_map <= 0.0)
    if bad.any():
        raise ValueError(
            f"noise map has {int(bad.sum())} non-finite or non-positive pixels "
            f"inside {region_name}; refusing to package — fix the reduction "
            f"(coverage, weights) rather than patching the noise map"
        )


def empirical_background_rms(sci: np.ndarray, n_sigma: float = 3.0) -> float:
    """Sigma-clipped RMS of the mosaic — the blank-sky validation check."""
    from astropy.stats import sigma_clipped_stats

    _, _, std = sigma_clipped_stats(sci[np.isfinite(sci)], sigma=n_sigma)
    return float(std)
