"""
Tier-1b optional PSF back-end: STARRED super-sampled ePSF (design doc stage 5,
Tier 1b; PyAutoReduce#35).

STARRED (STARlet REgularized Deconvolution; Michalewicz, Millon et al.,
COSMOGRAIL; arXiv:2402.08725) reconstructs the PSF from the *same field stars*
Tier 1 selects, as an analytic Moffat core plus a starlet-l1-regularised
super-sampled residual grid (Nyquist-sampled). It is a higher-fidelity,
reduction-stage alternative to the photutils Tier-1 ePSF — **not** a target
reconstruction (reconstructing the PSF from the lensed images themselves is a
modelling-stage concern; see `hst_acs_pipeline.md` Stage 5 and `keck_ao.md`
Tier C).

Following the `fallback.py` pattern, this module ships the **interface and
provenance contract**; the concrete STARRED reconstruction and the
drizzle-consistency resample land from the prototype spike (PyAutoReduce#35).
Two isolation rules are load-bearing and enforced here:

* **License.** STARRED is GPL-3.0-or-later while PyAutoReduce is permissive, so
  it is never a hard dependency: it is imported lazily, only when this back-end
  is explicitly selected, and shipped only as the optional extra
  ``pyautoreduce[starred]``. A missing import is a loud
  :class:`StarredUnavailableError`, never a silent degradation to Tier 1.
* **JAX.** STARRED depends on JAX, which the numpy/astropy-only unit-test
  boundary forbids; STARRED-touching code therefore runs in ``prototypes/`` /
  integration only (guard such tests with ``pytest.importorskip("starred")``).

The open technical problem — the spike's job — is **drizzle-consistency**:
STARRED's Nyquist super-sampled PSF lives in a "deconvolved frame" and must be
brought onto the drizzled mosaic grid (same kernel, pixfrac, scale and
orientation — the ``psf/__init__.py`` invariant) before it can be shipped as
``psf.fits`` / ``psf_full.fits``. Two candidate routes to evaluate in the
spike: (a) block-rebin the super-sampled grid by ``subsampling`` to the mosaic
scale (simplest; ignores the drizzle drop shape); or (b) the
``psf/frame_combine.py`` route — treat the reconstruction like a native-frame
PSF and drop-convolve (the ``final_pixfrac`` box) then resample via the local
frame->mosaic WCS Jacobian. Wiring emits a wrong PSF if this is guessed, so the
builder hard-stops until the spike resolves it.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from .epsf import normalise_kernel  # shared odd-crop + unit-normalise contract

__all__ = ["StarredUnavailableError", "build_starred_epsf"]


class StarredUnavailableError(NotImplementedError):
    """Tier-1b STARRED back-end requested but not installed/wired — a hard stop."""


def _require_starred():
    """Lazily import STARRED, isolating its GPL/JAX dependency from the core.

    Returns the imported top-level module; raises :class:`StarredUnavailableError`
    (a loud hard stop, per the no-silent-degradation contract) when the optional
    ``pyautoreduce[starred]`` extra is not installed.
    """
    try:
        import starred  # GPL-3.0-or-later; optional extra `pyautoreduce[starred]`
    except ImportError as exc:
        raise StarredUnavailableError(
            "Tier-1b STARRED ePSF back-end requested but the optional 'starred' "
            "package is not installed. Install the GPL-isolated extra "
            "(`pip install pyautoreduce[starred]`) to use it — this is a hard "
            "stop, not a fall-through to the photutils Tier-1 ePSF."
        ) from exc
    return starred


def build_starred_epsf(
    sci: np.ndarray,
    stars_table,
    psf_shape: Tuple[int, int],
    psf_full_shape: Tuple[int, int],
    subsampling: int = 2,
    moffat_fwhm_guess: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Build the STARRED super-sampled ePSF; return ``(psf, psf_full, diagnostics)``.

    The signature mirrors :func:`epsf.build_epsf` so the two Tier-1 back-ends are
    interchangeable behind the PSF-stage dispatch: same field-star input
    (``stars_table`` with ``xcentroid`` / ``ycentroid`` on ``sci``), same odd,
    centred, unit-normalised outputs, and the same diagnostics shape — a
    ``method`` key plus the provenance the ``reduction.json`` record carries
    (STARRED version, star count, ``subsampling``, regularisation, and the
    drizzle-consistency method once wired).

    NOT YET WIRED — the concrete reconstruction + drizzle-consistency resample
    land from the PyAutoReduce#35 spike; the dependency is checked first so an
    accidental selection without the extra gives the install message rather than
    a confusing partial run.
    """
    _require_starred()  # loud if the GPL/JAX optional extra is absent

    n_stars = 0 if stars_table is None else len(stars_table)
    raise StarredUnavailableError(
        "Tier-1b STARRED ePSF interface is defined but the reconstruction is "
        "not wired yet (PyAutoReduce#35). The prototype spike must resolve: "
        f"(1) the STARRED PSF-reconstruction call on the {n_stars} field-star "
        f"cutouts (Moffat core + starlet residuals, subsampling={subsampling}, "
        f"moffat_fwhm_guess={moffat_fwhm_guess}); and (2) the drizzle-consistency "
        "resample of the super-sampled PSF onto the mosaic grid (see this "
        "module's docstring and `psf/frame_combine.py`). Outputs must pass "
        f"through `normalise_kernel` to {psf_shape} / {psf_full_shape}."
    )
