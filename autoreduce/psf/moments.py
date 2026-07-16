"""
Shared kernel-moment diagnostics.

The two remaining FWHM estimators live with their single callers on
purpose: `epsf._fwhm_of` (radial half-max, a crude build diagnostic) and
`nirc2_star._fwhm_arcsec` (equivalent-area of the above-half-max core,
the AO vetting statistic) measure different things.
"""

import numpy as np


def moment_fwhm(kernel: np.ndarray) -> float:
    """
    Second-moment FWHM proxy (px): 2.3548 x sigma, isotropic average.

    Detection-free and continuous, so it works on undersampled kernels
    where a DAOFind-based estimate finds nothing and does not quantise to
    the pixel grid like a half-max-radius FWHM. Negative pixels (noise,
    convolution ringing) are clipped — they are not kernel flux.
    """
    pc = np.clip(kernel, 0.0, None)
    ny, nx = pc.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    total = pc.sum()
    cy = (pc * yy).sum() / total
    cx = (pc * xx).sum() / total
    var = (pc * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / total / 2.0
    return float(2.3548 * np.sqrt(max(var, 0.0)))
