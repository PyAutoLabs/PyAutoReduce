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

Two isolation rules are load-bearing:

* **License.** STARRED is GPL-3.0-or-later while PyAutoReduce is permissive, so
  it is never a hard dependency: imported lazily, only when this back-end is
  selected, shipped only as the optional extra ``pyautoreduce[starred]``. A
  missing import is a loud :class:`StarredUnavailableError`, never a silent
  degradation to Tier 1. (autoreduce itself has no astropy/scipy caps, so the
  extra coexists with STARRED's astropy-8 / jax stack — unlike the full
  autoarray/autofit stack, which does cap them.)
* **JAX.** STARRED depends on JAX; ``build_starred_epsf`` runs only where the
  extra is installed. Unit tests here are numpy/scipy-only: the star guard and
  the centroid-preserving *delivery* are exercised without importing STARRED
  (``pytest.importorskip("starred")`` guards any test that reconstructs).

**Drizzle-consistency (settled by the #35 adversarial ground-truth test).**
STARRED emits a Nyquist super-sampled PSF; delivery = block-downsample to the
mosaic grid (route a) followed by a **centroid-preserving crop** — crop around
the measured centroid and sub-pixel-recentre so the centre of light lands on
the odd kernel's central pixel. The naive even-super -> odd-kernel crop injects
a ~0.5 px offset; centroid-preserving delivery removes it (0.63 px -> 0.009 px).
For well-sampled PSFs this is drizzle-consistent (centroid ~0.002 px, flux
exact, size within ~3%); the drizzle drop shape (pixfrac<1) is a second-order
refinement. Undersampled PSFs (FWHM<~1.6 px) broaden and are flagged.
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np

from .epsf import InsufficientStarsError

__all__ = [
    "StarredUnavailableError",
    "build_starred_epsf",
    "build_starred_frame_epsf",
]

MIN_STARS = 8  # match the Tier-1 floor; STARRED works from ~6 but be conservative
STAMP_PAD = 8  # star-cutout pad beyond the extended kernel (centroid-crop margin)
UNDERSAMPLED_FWHM_PX = (
    1.6  # below this STARRED broadens (~24% at 1.3px) -> flag, don't ship silently
)


def _stamp_for(psf_full_shape):
    """Even per-star cutout size. The cutout sets the reconstruction extent, and
    the downsampled grid it yields (== stamp px) must contain the *extended*
    kernel plus centroid-crop margin — a fixed small stamp cannot deliver
    psf_full (the end-to-end #35 bug)."""
    stamp = max(psf_full_shape) + STAMP_PAD
    return stamp + (stamp % 2)


class StarredUnavailableError(NotImplementedError):
    """Tier-1b STARRED back-end requested but the optional extra is not installed."""


def _require_starred():
    """Lazily import STARRED, isolating its GPL/JAX dependency from the core."""
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


def _extract_cutouts(sci, noise, stars_table, stamp):
    """Square (N, stamp, stamp) science + noise cutouts around each star.

    Numpy-only (no STARRED). Skips stars whose window overruns the mosaic edge
    or contains non-finite / non-positive-noise pixels — STARRED's fitter
    refuses NaNs and needs a positive noise map.
    """
    if stars_table is None or len(stars_table) == 0:
        return np.empty((0, stamp, stamp)), np.empty((0, stamp, stamp))
    ny, nx = sci.shape
    h = stamp // 2
    sci_cuts, noise_cuts = [], []
    for x, y in zip(
        np.asarray(stars_table["xcentroid"], dtype=float),
        np.asarray(stars_table["ycentroid"], dtype=float),
    ):
        ix, iy = int(round(x)), int(round(y))
        if ix - h < 0 or iy - h < 0 or ix + h > nx or iy + h > ny:
            continue
        s = sci[iy - h : iy + h, ix - h : ix + h]
        n = noise[iy - h : iy + h, ix - h : ix + h]
        if np.isfinite(s).all() and np.isfinite(n).all() and (n > 0).all():
            sci_cuts.append(s)
            noise_cuts.append(n)
    if not sci_cuts:
        return np.empty((0, stamp, stamp)), np.empty((0, stamp, stamp))
    return np.array(sci_cuts), np.array(noise_cuts)


def _core_centroid(p, window=6):
    """Centroid of the PSF *core*: centre of mass in a window around the peak.
    Robust to the asymmetric wings / low-level noise that bias a global centre
    of mass — the mis-centring the end-to-end #35 run exposed (0.69 px) on a
    real STARRED PSF where a symmetric-Gaussian unit test did not."""
    pc = np.clip(p, 0, None)
    iy, ix = np.unravel_index(int(np.argmax(pc)), pc.shape)
    y0, y1 = max(iy - window, 0), min(iy + window + 1, pc.shape[0])
    x0, x1 = max(ix - window, 0), min(ix + window + 1, pc.shape[1])
    sub = pc[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    t = sub.sum()
    return (yy * sub).sum() / t, (xx * sub).sum() / t


def _downsample_box(img, factor):
    """Block-average rebin by an integer factor (== starred Downsample,
    conserve_flux=False, up to the unit normalisation applied downstream)."""
    ny, nx = img.shape
    ny -= ny % factor
    nx -= nx % factor
    v = img[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor)
    return v.mean(axis=(1, 3))


def _size_fwhm(p):
    """Second-moment FWHM proxy (px): 2.355 * sigma. Continuous, unlike a
    half-max-radius FWHM which quantises to the pixel grid."""
    pc = np.clip(p, 0, None)
    ny, nx = pc.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    t = pc.sum()
    cy, cx = (yy * pc).sum() / t, (xx * pc).sum() / t
    ixx = ((xx - cx) ** 2 * pc).sum() / t
    iyy = ((yy - cy) ** 2 * pc).sum() / t
    return float(2.3548 * np.sqrt(0.5 * (ixx + iyy)))


def _deliver(psf_supersampled, subsampling, shape):
    """Route-a drizzle-consistent delivery: block-downsample to the mosaic grid,
    then crop around the measured centroid and sub-pixel-recentre onto the odd
    kernel's central pixel. Returns an odd, centred, unit-normalised kernel.

    The recentre is the fix the #35 ground-truth test mandated: a naive
    array-centre crop of an even super-grid injects a ~0.5 px offset.
    """
    from scipy.ndimage import shift as nd_shift

    if any(s % 2 == 0 for s in shape):
        raise ValueError(f"kernel shape must be odd: {shape}")
    grid = _downsample_box(np.asarray(psf_supersampled, dtype=float), subsampling)
    cy, cx = _core_centroid(grid)
    icy, icx = int(round(cy)), int(round(cx))
    hy, hx = shape[0] // 2, shape[1] // 2
    if (
        icy - hy < 0
        or icx - hx < 0
        or icy + hy + 1 > grid.shape[0]
        or icx + hx + 1 > grid.shape[1]
    ):
        raise ValueError(
            f"delivered kernel {shape} does not fit around the PSF centroid in the "
            f"downsampled grid {grid.shape}; increase the STARRED stamp/subsampling"
        )
    cut = grid[icy - hy : icy + hy + 1, icx - hx : icx + hx + 1].astype(float)
    cut = nd_shift(cut, (-(cy - icy), -(cx - icx)), order=3, mode="constant")
    cut = np.clip(cut, 0, None)
    total = cut.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("delivered STARRED kernel has non-positive total flux")
    return cut / total


def build_starred_epsf(
    sci: np.ndarray,
    noise: np.ndarray,
    stars_table,
    psf_shape: Tuple[int, int],
    psf_full_shape: Tuple[int, int],
    subsampling: int = 2,
    stamp: Optional[int] = None,
    moffat_fwhm_guess: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Build the STARRED super-sampled ePSF; return ``(psf, psf_full, diagnostics)``.

    Mirrors :func:`epsf.build_epsf` so the two Tier-1 back-ends are
    interchangeable behind the PSF-stage dispatch, but also takes the ``noise``
    map (STARRED weights each star by its per-pixel noise). ``stars_table`` is
    the same ``find_stars`` output (``xcentroid`` / ``ycentroid``). Outputs are
    odd, centred, unit-normalised kernels; diagnostics carry the provenance the
    ``reduction.json`` record needs.
    """
    # Star selection + extraction are numpy-only and run first, so a too-poor
    # field fails the same loud way as Tier-1 regardless of the optional extra.
    if stamp is None:
        stamp = _stamp_for(psf_full_shape)
    cutouts, noisemaps = _extract_cutouts(sci, noise, stars_table, stamp)
    if len(cutouts) < MIN_STARS:
        raise InsufficientStarsError(
            f"{len(cutouts)} usable star cutouts (< {MIN_STARS}); STARRED Tier-1b "
            f"is not viable for this field — select Tier 1 or Tier 2 explicitly"
        )

    starred = _require_starred()  # loud if the GPL/JAX optional extra is absent
    from starred.procedures.psf_routines import build_psf

    result = build_psf(
        image=np.asarray(cutouts, dtype=float),
        noisemap=np.asarray(noisemaps, dtype=float),
        subsampling_factor=subsampling,
        guess_fwhm_pixels=moffat_fwhm_guess if moffat_fwhm_guess is not None else 3.0,
        adjust_sky=True,
    )
    full_ss = np.asarray(result["full_psf"], dtype=float)

    psf_full = _deliver(full_ss, subsampling, psf_full_shape)
    psf = _deliver(full_ss, subsampling, psf_shape)

    fwhm = _size_fwhm(psf)
    undersampled = fwhm < UNDERSAMPLED_FWHM_PX
    if undersampled:
        warnings.warn(
            f"STARRED Tier-1b PSF is undersampled (FWHM {fwhm:.2f} px < "
            f"{UNDERSAMPLED_FWHM_PX} px); the reconstruction broadens in this "
            f"regime (~24% at 1.3 px in the #35 test) — recorded in provenance",
            RuntimeWarning,
        )
    dcy, dcx = _core_centroid(psf)
    diagnostics = {
        "method": "starred-epsf-tier1b",
        "starred_version": str(getattr(starred, "__version__", None) or "unknown"),
        "n_stars_used": int(len(cutouts)),
        "subsampling": int(subsampling),
        "delivery": "downsample+centroid-recentre",
        "sampling_fwhm_px": fwhm,
        "undersampled": bool(undersampled),
        "centroid_residual_px": float(
            np.hypot(dcy - (psf.shape[0] - 1) / 2, dcx - (psf.shape[1] - 1) / 2)
        ),
    }
    return psf, psf_full, diagnostics


def build_starred_frame_epsf(
    hdul,
    extver: int,
    spec,
    adapter,
    subsampling: int = 2,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """Tier-1b native-pixel STARRED ePSF for one calibrated frame (frame-products
    mode). Mirrors :func:`frame_epsf.build_frame_epsf`'s contract: returns
    ``(None, None, diagnostics)`` when too few usable stars survive — a single
    exposure legitimately may lack stars, so that is a recorded outcome, not a
    hard stop (only a missing optional extra is).

    Unlike the mosaic back-end there is **no drizzle-consistency resample** — the
    frame PSF stays on the frame's own native, distorted pixel grid, delivered
    by the same super-sample downsample + centroid-preserving crop. Per-star
    noise weights come from the frame's ERR extension.

    NB the #37/#39 regime finding: STARRED broadens on undersampled PSFs, and
    native frames are less well-sampled than the drizzled mosaic. Prefer this for
    well-sampled frames; the `undersampled` diagnostic flags the risky regime.
    """
    from .frame_epsf import _prepare_frame

    starred = _require_starred()  # loud if the optional GPL/JAX extra is absent
    work, err, found, n_patched, _primary, _target_xy, _det_shape = _prepare_frame(
        hdul, extver, spec, adapter
    )
    if err is None:
        raise ValueError(
            "STARRED frame back-end needs a per-pixel ERR extension for star "
            "noise weighting; this frame carries none"
        )

    n_cand = 0 if found is None else len(found)
    if n_cand < MIN_STARS:
        return (
            None,
            None,
            {
                "method": "none",
                "reason": f"{n_cand} star candidates (< {MIN_STARS}); STARRED frame ePSF not viable",
                "n_patched_pixels": n_patched,
            },
        )

    from starred.procedures.psf_routines import build_psf

    stamp = _stamp_for(spec.psf_full_shape)
    cutouts, noisemaps = _extract_cutouts(work, err, found, stamp)
    if len(cutouts) < MIN_STARS:
        return (
            None,
            None,
            {
                "method": "none",
                "reason": f"{len(cutouts)} usable star cutouts (< {MIN_STARS}) after edge/finite cut",
                "n_patched_pixels": n_patched,
            },
        )

    result = build_psf(
        image=np.asarray(cutouts, dtype=float),
        noisemap=np.asarray(noisemaps, dtype=float),
        subsampling_factor=subsampling,
        adjust_sky=True,
    )
    full_ss = np.asarray(result["full_psf"], dtype=float)
    psf_full = _deliver(full_ss, subsampling, spec.psf_full_shape)
    psf = _deliver(full_ss, subsampling, spec.psf_shape)

    fwhm = _size_fwhm(psf)
    dcy, dcx = _core_centroid(psf)
    return (
        psf,
        psf_full,
        {
            "method": "starred-frame-tier1b",
            "starred_version": str(getattr(starred, "__version__", None) or "unknown"),
            "n_stars_used": int(len(cutouts)),
            "subsampling": int(subsampling),
            "delivery": "native downsample+centroid-recentre",
            "sampling_fwhm_px": fwhm,
            "undersampled": bool(fwhm < UNDERSAMPLED_FWHM_PX),
            "n_patched_pixels": n_patched,
            "cr_screen": "DQ-patched" if n_patched else "shape-cuts-only",
            "centroid_residual_px": float(
                np.hypot(dcy - (psf.shape[0] - 1) / 2, dcx - (psf.shape[1] - 1) / 2)
            ),
        },
    )
