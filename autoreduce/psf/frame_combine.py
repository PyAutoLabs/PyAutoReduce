"""
Mosaic PSF from the per-frame ePSFs (issue #21, `psf_from_frames`).

An alternative to estimating the PSF on the drizzled mosaic: build each
frame's tier-1 native ePSF (`frame_epsf`), push it through the same
geometry the science pixels took — convolve with the drizzle drop (the
``final_pixfrac`` box in native pixels) and resample onto the mosaic pixel
grid via the local frame→mosaic WCS Jacobian at the target position — then
average, weighted by exposure time. Star-measured mosaic ePSFs inherit
resampling artifacts and the mosaic's star scarcity; the combination
sidesteps both and uses every frame's full star field.

Approximations, recorded in the diagnostics: the drop convolution and the
local-affine resample capture drizzle's broadening and geometry to first
order; the sub-pixel dither phases of the output sampling are not modelled
(they average toward a half-pixel box already present in the ePSFs).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..instruments import InstrumentAdapter
from ..target import TargetSpec
from . import frame_epsf as frame_epsf_mod
from .epsf import normalise_kernel


def _moment_fwhm(kernel: np.ndarray) -> float:
    """Second-moment FWHM (2.3548·sigma) — detection-free, so it works on
    undersampled kernels where a DAOFind-based estimate finds nothing."""
    yy, xx = np.mgrid[0 : kernel.shape[0], 0 : kernel.shape[1]]
    total = kernel.sum()
    cy = (kernel * yy).sum() / total
    cx = (kernel * xx).sum() / total
    var = (kernel * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / total / 2.0
    return float(2.3548 * np.sqrt(max(var, 0.0)))


def _drop_convolve(kernel: np.ndarray, pixfrac: float) -> np.ndarray:
    """Convolve with drizzle's drop — a ``pixfrac``-wide box in native px.

    Done in Fourier space so the fractional box width is exact: the box's
    transform is ``sinc(pixfrac * f)`` per axis.
    """
    ny, nx = kernel.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.rfftfreq(nx)[None, :]
    transfer = np.sinc(pixfrac * fy) * np.sinc(pixfrac * fx)
    return np.fft.irfft2(np.fft.rfft2(kernel) * transfer, s=kernel.shape)


def _local_jacobian(frame_wcs, frame_xy, mosaic_wcs) -> np.ndarray:
    """d(mosaic pixel)/d(frame pixel) at ``frame_xy``, by finite differences
    through the composed frame→world→mosaic transform."""

    def to_mosaic(x, y):
        ra, dec = frame_wcs.pixel_to_world_values(x, y)
        mx, my = mosaic_wcs.world_to_pixel_values(ra, dec)
        return np.array([float(mx), float(my)])

    x0, y0 = frame_xy
    base = to_mosaic(x0, y0)
    jac = np.empty((2, 2))
    jac[:, 0] = to_mosaic(x0 + 1.0, y0) - base  # d(mx,my)/d(frame x)
    jac[:, 1] = to_mosaic(x0, y0 + 1.0) - base  # d(mx,my)/d(frame y)
    return jac


def _resample_to_mosaic(
    kernel: np.ndarray, jac: np.ndarray, shape: Tuple[int, int]
) -> np.ndarray:
    """Resample a centred native-pixel kernel onto the mosaic grid.

    Output pixel offsets (from the kernel centre, in mosaic pixels) are
    mapped back to frame-pixel offsets with the inverse Jacobian and the
    kernel is sampled there; the Jacobian determinant preserves flux.
    """
    from scipy.ndimage import map_coordinates

    ny, nx = shape
    cy_out, cx_out = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    offsets = np.stack([xx - cx_out, yy - cy_out])  # mosaic-px (dx, dy)

    inv = np.linalg.inv(jac)
    fx = inv[0, 0] * offsets[0] + inv[0, 1] * offsets[1]
    fy = inv[1, 0] * offsets[0] + inv[1, 1] * offsets[1]

    ky, kx = kernel.shape
    cy_in, cx_in = (ky - 1) / 2.0, (kx - 1) / 2.0
    out = map_coordinates(
        kernel, [fy + cy_in, fx + cx_in], order=1, mode="constant", cval=0.0
    )
    return out * abs(np.linalg.det(inv))


def combined_mosaic_psf(
    exposures: List,
    spec: TargetSpec,
    adapter: InstrumentAdapter,
    mosaic_header,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    The mosaic PSF as the exposure-time-weighted combination of the
    per-frame tier-1 ePSFs, resampled through each frame's geometry.

    Loud when no frame can contribute — the caller asked for this method
    explicitly; silently falling back to the mosaic-star ePSF would hide
    that the request was not honoured.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    mosaic_wcs = WCS(mosaic_header)
    contributions = []
    per_frame: List[Dict] = []
    for path in exposures:
        with fits.open(path) as hdul:
            primary = hdul[0].header
            rootname = str(primary.get("ROOTNAME", "unknown")).strip().lower()
            extvers = [
                int(hdu.header.get("EXTVER", 1))
                for hdu in hdul
                if hdu.name == "SCI"
            ]
            for extver in extvers:
                hdr = hdul["SCI", extver].header
                wcs_full = WCS(hdr, fobj=hdul, naxis=2)
                x, y = wcs_full.world_to_pixel_values(spec.ra, spec.dec)
                ny, nx = hdul["SCI", extver].data.shape
                if not (
                    np.isfinite(x)
                    and np.isfinite(y)
                    and 0 <= float(x) <= nx - 1
                    and 0 <= float(y) <= ny - 1
                ):
                    continue  # PSF at the target position needs the target on chip
                psf, psf_full, diag = frame_epsf_mod.build_frame_epsf(
                    hdul, extver, spec, adapter
                )
                entry = {"rootname": rootname, "chip": extver, **diag}
                if psf_full is None:
                    per_frame.append(entry)
                    continue
                dropped = _drop_convolve(psf_full, spec.final_pixfrac)
                jac = _local_jacobian(wcs_full, (float(x), float(y)), mosaic_wcs)
                resampled = _resample_to_mosaic(
                    dropped, jac, spec.psf_full_shape
                )
                from ..package.frames import _exposure_time

                weight = _exposure_time(primary)
                if weight <= 0.0:
                    raise ValueError(
                        f"{rootname} chip {extver}: no positive exposure time "
                        "(EXPTIME/XPOSURE/EFFEXPTM) to weight the PSF "
                        "combination"
                    )
                contributions.append((resampled, weight))
                per_frame.append({**entry, "weight_exptime": weight})

    if not contributions:
        raise ValueError(
            "psf_from_frames requested but no frame yields a tier-1 ePSF "
            f"({len(per_frame)} chips examined) — drop the option or wait "
            "for the tier-2 model PSF (frame outcomes above in this record)"
        )

    total = sum(w for _, w in contributions)
    stack = sum(w * k for k, w in contributions) / total
    psf_full = normalise_kernel(stack, spec.psf_full_shape)
    psf = normalise_kernel(stack, spec.psf_shape)
    diagnostics = {
        "method": "epsf-frames-combined",
        "n_frames_combined": len(contributions),
        "weighting": "exptime",
        "pixfrac_drop_convolved": spec.final_pixfrac,
        "resample": "local-affine frame->mosaic Jacobian at the target",
        "moment_fwhm_pix": _moment_fwhm(psf),
        "frames": per_frame,
    }
    return psf, psf_full, diagnostics
