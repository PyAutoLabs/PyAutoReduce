"""
Tier-2 PSF fallback interface (design doc stage 5).

For star-poor fields (the SLACS-snapshot regime) the PSF comes from a model —
STScI focus-diverse ePSF grids or TinyTim raytraces — evaluated per exposure
and made drizzle-consistent by resampling through the same footprint as the
science mosaic. Phase 1 ships the interface and provenance contract; the
concrete grid/TinyTim back-ends land when the first star-poor target needs
them (tracked on the roadmap).
"""

from typing import Dict, Tuple

import numpy as np


class ModelPSFUnavailableError(NotImplementedError):
    """No tier-2 back-end is wired up yet; the caller must not degrade silently."""


def model_psf(
    spec_name: str,
    filter_name: str,
    psf_shape: Tuple[int, int],
    psf_full_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    raise ModelPSFUnavailableError(
        f"tier-2 model PSF requested for {spec_name} ({filter_name}) but no "
        f"back-end (focus-diverse ePSF grid / TinyTim) is implemented yet; "
        f"tier 1 failed or was declined — this is a hard stop, not a warning"
    )
