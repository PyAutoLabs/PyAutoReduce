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


# Masked-by-noise convention: bad pixels carry effectively infinite noise so
# any chi^2 ignores them — the same treatment the legacy noise-scaled
# datasets use for artifacts and contaminants.
MASKED_NOISE_VALUE = 1.0e8


def mask_isolated_bad_pixels(
    data_cut: np.ndarray,
    noise_cut: np.ndarray,
    center_xy,
    pixel_scale: float,
    max_bad_fraction: float = 0.005,
    protect_radius_arcsec: float = 1.5,
    region_name: str = "cutout",
):
    """
    Apply the bad-pixel policy to a cutout pair.

    Isolated non-finite/non-positive noise pixels (fully-rejected or dead
    pixels — routine in deep resampled stacks) are set to `MASKED_NOISE_VALUE`
    with the data zeroed, and the count/positions are returned for provenance.
    The failure stays loud where it matters: more than `max_bad_fraction` of
    the cutout bad, or any bad pixel within `protect_radius_arcsec` of the
    target centre (the lens itself must be clean).
    """
    bad = ~np.isfinite(noise_cut) | (noise_cut <= 0.0)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return data_cut, noise_cut, {"n_masked_pixels": 0}

    # "Isolated" is enforced: a bad pixel with two or more bad 4-neighbours
    # marks a structured defect (blob/column), which must fail loudly — only
    # scattered singletons and pairs are maskable.
    neighbours = sum(
        np.roll(bad, shift, axis) for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1))
    )
    if (bad & (neighbours >= 2)).any():
        raise ValueError(
            f"structured bad-pixel region in {region_name} ({n_bad} bad px with "
            f"contiguous clustering) — fix the reduction, don't mask a defect"
        )

    fraction = n_bad / bad.size
    if fraction > max_bad_fraction:
        raise ValueError(
            f"{n_bad} bad noise pixels ({fraction:.2%}) in {region_name} exceed "
            f"the {max_bad_fraction:.2%} policy limit — fix the reduction "
            f"(coverage, weights), don't mask wholesale"
        )
    ys, xs = np.where(bad)
    cx, cy = center_xy
    r_arcsec = np.hypot(ys - cy, xs - cx) * pixel_scale
    if (r_arcsec < protect_radius_arcsec).any():
        raise ValueError(
            f"bad noise pixel within {protect_radius_arcsec}\" of the target "
            f"centre in {region_name} — the lens region must reduce cleanly"
        )

    data_out = np.where(bad, 0.0, data_cut)
    noise_out = np.where(bad, MASKED_NOISE_VALUE, noise_cut)
    diagnostics = {
        "n_masked_pixels": n_bad,
        "masked_fraction": fraction,
        "masked_noise_value": MASKED_NOISE_VALUE,
        "min_masked_radius_arcsec": float(r_arcsec.min()),
    }
    return data_out, noise_out, diagnostics


def empirical_background_rms(sci: np.ndarray, n_sigma: float = 3.0) -> float:
    """Sigma-clipped RMS of the mosaic — the blank-sky validation check."""
    from astropy.stats import sigma_clipped_stats

    _, _, std = sigma_clipped_stats(sci[np.isfinite(sci)], sigma=n_sigma)
    return float(std)
