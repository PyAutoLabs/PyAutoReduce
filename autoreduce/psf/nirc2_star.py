"""
Tier-A AO PSF: PSF-star exposures reduced pipeline-identically
(design doc keck_ao.md, stage 6).

The AO PSF is time-variable and there is rarely a star inside the narrow
camera's 10" field, so SHARP observes dedicated PSF stars interleaved with
the science and selects between epochs *during lens modelling* (Bayesian
evidence — Lagattuta et al. 2012). The reduction therefore ships **every
epoch as a candidate** (`psf_candidate_<i>.fits`) plus the sharpest one as
`psf.fits`/`psf_full.fits`, and marks the products provisional in
provenance — an AO PSF is never final at reduction time.

Candidate = one contiguous PSF-star visit (frames separated by more than
`EPOCH_GAP_S` start a new epoch), combined through the same nirc2_native
backend as the science mosaic — the drizzled-PSF invariant, exactly as HST
tier 2 drizzles TinyTim models through the science footprint.
"""

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..instruments import InstrumentAdapter
from ..target import TargetSpec
from .epsf import normalise_kernel

EPOCH_GAP_S = 600.0


def group_epochs(mjds: List[float], gap_s: float = EPOCH_GAP_S) -> List[List[int]]:
    """Frame indices grouped into contiguous visits by MJD gaps."""
    if not mjds:
        raise ValueError("no PSF-star frames to group")
    order = sorted(range(len(mjds)), key=lambda i: mjds[i])
    groups, current = [], [order[0]]
    for prev, this in zip(order, order[1:]):
        if (mjds[this] - mjds[prev]) * 86400.0 > gap_s:
            groups.append(current)
            current = []
        current.append(this)
    groups.append(current)
    return groups


def _centre_on_peak(sci: np.ndarray, half: int) -> np.ndarray:
    """Odd-sized window centred on the brightest pixel; loud near edges."""
    finite = np.nan_to_num(sci, nan=-np.inf)
    py, px = np.unravel_index(np.argmax(finite), sci.shape)
    if (
        py - half < 0
        or px - half < 0
        or py + half + 1 > sci.shape[0]
        or px + half + 1 > sci.shape[1]
    ):
        raise ValueError(
            f"PSF star peak at ({py}, {px}) is within {half} px of the mosaic "
            f"edge — the star combine footprint is too tight"
        )
    return sci[py - half : py + half + 1, px - half : px + half + 1]


def _fwhm_arcsec(kernel: np.ndarray, pixel_scale: float) -> float:
    """Equivalent-area FWHM of the above-half-maximum core."""
    peak = float(kernel.max())
    if peak <= 0.0:
        raise ValueError("PSF kernel has a non-positive peak")
    n_above = int((kernel > 0.5 * peak).sum())
    return 2.0 * np.sqrt(n_above / np.pi) * pixel_scale


def build_candidates(
    star_frame_paths: List[Path],
    star_mjds: List[float],
    spec: TargetSpec,
    adapter: InstrumentAdapter,
    work_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Dict]:
    """
    Combine each PSF-star epoch; return (psf, psf_full, candidates,
    diagnostics). `psf`/`psf_full` are cut from the sharpest candidate
    (highest peak fraction — a Strehl proxy); all epochs ride along as
    `candidates` for evidence-based selection during modelling.
    """
    from astropy.io import fits

    from ..drizzle import nirc2_combine

    if len(star_frame_paths) != len(star_mjds):
        raise ValueError(
            f"{len(star_frame_paths)} star frames vs {len(star_mjds)} MJDs"
        )

    half_full = max(spec.psf_full_shape) // 2
    candidates, stats, rejected = [], [], []
    for i, group in enumerate(group_epochs(star_mjds)):
        epoch_spec = replace(spec, name=f"{spec.name}_psfstar{i}")
        epoch_dir = Path(work_dir) / f"psf_epoch_{i}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        paths = [star_frame_paths[j] for j in group]
        if len(paths) == 1:
            # A single-frame epoch cannot register/reject; combine still
            # resamples it through the identical mapping (drizzled-PSF
            # invariant), it just carries no outlier protection.
            pass
        sci_path, _, _ = nirc2_combine.combine(paths, epoch_spec, adapter, epoch_dir)
        sci = fits.getdata(sci_path).astype(np.float64)
        window = _centre_on_peak(sci, half_full)

        # Local background off the window's border ring, then vet: a real AO
        # PSF is spatially coherent (bright 3x3 core) with positive total
        # flux; a hot-pixel peak on empty sky is neither. Starless epochs
        # (telescope offsets, failed acquisitions) are rejected with a
        # recorded reason — only an all-epochs-starless result is fatal.
        ring = np.concatenate(
            [window[0], window[-1], window[1:-1, 0], window[1:-1, -1]]
        )
        window = window - np.nanmedian(ring)
        ring_mad = 1.4826 * np.nanmedian(np.abs(ring - np.nanmedian(ring)))
        h = window.shape[0] // 2
        core = window[h - 1 : h + 2, h - 1 : h + 2].sum()
        total = np.nansum(window)
        if total <= 0.0 or (ring_mad > 0 and core < 10.0 * ring_mad):
            rejected.append(
                {
                    "epoch": i,
                    "n_frames": len(paths),
                    "reason": f"no coherent star: window total {total:.1f}, "
                    f"3x3 core {core:.1f} vs ring MAD {ring_mad:.2f}",
                }
            )
            continue
        # Physical sharpness floor: a real AO PSF cannot be narrower than
        # the diffraction core (~45 mas at K' on Keck); a "PSF" of one or
        # two pixels is a cosmic-ray hit or hot pixel, which single-frame
        # epochs cannot reject at combine. Validated on SHARP B1938: CR
        # epochs measured 11-16 mas, the real star 72 mas.
        fwhm = _fwhm_arcsec(
            window / total if total > 0 else window, spec.final_scale
        )
        if fwhm < max(2.0 * spec.final_scale, 0.025):
            rejected.append(
                {
                    "epoch": i,
                    "n_frames": len(paths),
                    "reason": f"unphysically sharp peak (FWHM {fwhm*1000:.0f} "
                    f"mas < diffraction floor) — cosmic ray, not a star",
                }
            )
            continue

        kernel_full = normalise_kernel(window, spec.psf_full_shape)
        candidates.append(kernel_full)
        stats.append(
            {
                "epoch": i,
                "n_frames": len(paths),
                "mjd_start": float(min(star_mjds[j] for j in group)),
                "peak_fraction": float(kernel_full.max()),
                "peak_cps": float(np.nanmax(sci)),
                "fwhm_arcsec": _fwhm_arcsec(kernel_full, spec.final_scale),
            }
        )

    if not candidates:
        raise ValueError(
            f"no PSF-star epoch contains a coherent star "
            f"(all {len(rejected)} rejected: {rejected}) — check the "
            f"koa_psf_star_ids frame selection"
        )

    best = int(np.argmax([s["peak_fraction"] for s in stats]))
    psf_full = candidates[best]
    psf = normalise_kernel(psf_full, spec.psf_shape)

    diagnostics = {
        "method": "tier A: PSF-star epochs reduced pipeline-identically",
        "psf_provisional": True,
        "n_candidates": len(candidates),
        "selected_epoch": stats[best]["epoch"],
        "selection": "peak-fraction (Strehl proxy); final selection belongs "
        "to lens modelling (evidence over candidates)",
        "candidates": stats,
        "rejected_epochs": rejected,
    }
    return psf, psf_full, candidates, diagnostics
