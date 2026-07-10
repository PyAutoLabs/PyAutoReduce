"""
NIRC2 native combine backend (design doc keck_ao.md, stage 4): dewarp +
registration + coaddition through the ``drizzle`` package — the same
resampling engine inside drizzlepac and the jwst pipeline, so the Casertano
correlated-noise factor and the drizzled-PSF invariant carry over unchanged.

Inputs are the pipeline-prepared frames the ground stages wrote to the work
directory: calibrated (total e-), sky-subtracted, with the per-frame facts
in the header (ITIME, COADDS, SKYLEV, DISTX/DISTY pointing at the synced
distortion tables). The geometric distortion enters as the drizzle pixel
mapping — exactly how drizzlepac treats ACS distortion — so rectification
and coaddition are one resampling, not two.

Per-frame weights are inverse background variance (sky + dark + read noise,
in cps^2), making the accumulated weight map the IVM the shared noise recipe
expects; the mosaic is written in e-/s with the total EXPTIME, so
``noise.rms.noise_map_from`` applies verbatim.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..align.registration import offsets_to_reference
from ..instruments import InstrumentAdapter
from ..target import TargetSpec

# Margin (output pixels) around the union footprint of all aligned frames,
# on top of the loaded distortion solution's own maximum shift.
_GRID_MARGIN = 4


def load_distortion(dist_x_path: Path, dist_y_path: Path, shape) -> np.ndarray:
    """
    Load the lookup tables into an additive (dy, dx) correction stack:
    rectified = observed + correction, native pixels.
    """
    from astropy.io import fits

    dx = fits.getdata(dist_x_path).astype(np.float64)
    dy = fits.getdata(dist_y_path).astype(np.float64)
    if dx.shape != tuple(shape) or dy.shape != tuple(shape):
        raise ValueError(
            f"distortion tables {dx.shape}/{dy.shape} do not match the frame "
            f"shape {tuple(shape)} — subarray reductions are not supported; "
            f"reduce full frames (design doc, open items)"
        )
    return np.stack([dy, dx])


def build_pixmap(
    shape: Tuple[int, int],
    distortion: np.ndarray,
    offset: Tuple[float, float],
    origin: Tuple[float, float],
    scale_ratio: float,
    grids: Tuple[np.ndarray, np.ndarray] = None,
) -> np.ndarray:
    """
    The (ny, nx, 2) input->output pixel mapping drizzle consumes:
    rectify (distortion), align (frame offset), re-origin, and resample
    (native -> final scale). Axis order in the map is (x, y) per the
    drizzle convention. `grids` lets callers hoist the (yy, xx) mgrid out
    of a per-frame loop.
    """
    ny, nx = shape
    yy, xx = grids if grids is not None else np.mgrid[0.0:ny, 0.0:nx]
    y_rect = yy + distortion[0] - offset[0] - origin[0]
    x_rect = xx + distortion[1] - offset[1] - origin[1]
    pixmap = np.empty((ny, nx, 2), dtype=np.float64)
    pixmap[..., 0] = x_rect * scale_ratio
    pixmap[..., 1] = y_rect * scale_ratio
    return pixmap


def _frame_background_variance_e(header, detector) -> float:
    """Background variance (e-^2/pixel) from the frame's recorded facts."""
    coadds = int(header["COADDS"])
    itime = float(header["ITIME"])
    sky_e = float(header["SKYLEV"])
    dark_e = detector.dark_e_per_s * itime * coadds
    read_noise = detector.read_noise_e(
        int(header.get("SAMPMODE", 2)), int(header.get("MULTISAM", 1))
    )
    read_e2 = (read_noise**2) * coadds
    if sky_e < 0.0:
        # A negative median sky means the sky model failed upstream.
        raise ValueError(f"negative sky level in frame header: {sky_e}")
    return sky_e + dark_e + read_e2


def combine(
    exposures: List[Path],
    spec: TargetSpec,
    adapter: InstrumentAdapter,
    output_dir: Path,
) -> Tuple[Path, Path, Dict]:
    """
    Combine prepared NIRC2 frames; return (sci_path, wht_path, provenance).
    Matches the backend seam of `drizzle.combine.combine`.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    from drizzle.resample import Drizzle

    from ._common import combine_provenance

    if adapter.key != "nirc2_narrow":
        raise NotImplementedError(
            f"{adapter.key}: the published NIRC2 distortion solutions cover "
            f"the narrow camera only; wide-camera combination is a design "
            f"open item (docs/design/keck_ao.md)"
        )

    detector = adapter.ground_detector()
    frames, headers = [], []
    for path in exposures:
        with fits.open(path) as hdul:
            frames.append(hdul[0].data.astype(np.float64))
            headers.append(hdul[0].header.copy())

    # One distortion solution per combine: prepared frames must agree (a
    # set spanning the 2015-04-13 epoch boundary is rejected upstream at
    # acquire, and again here in case of manually assembled stacks).
    dist_keys = {(str(h["DISTX"]), str(h["DISTY"])) for h in headers}
    if len(dist_keys) != 1:
        raise ValueError(
            f"prepared frames carry {len(dist_keys)} different distortion "
            f"solutions ({sorted(dist_keys)}); frames must share one epoch"
        )
    distortion = load_distortion(
        Path(headers[0]["DISTX"]), Path(headers[0]["DISTY"]), frames[0].shape
    )
    offsets = offsets_to_reference(frames)
    scale_ratio = adapter.native_scale / spec.final_scale

    # Output grid: union footprint of the rectified, aligned frame corners,
    # padded by the distortion solution's own maximum shift (edge pixels can
    # move by more than a fixed margin) plus a fixed safety margin.
    ny, nx = frames[0].shape
    dist_margin_y = float(np.ceil(np.abs(distortion[0]).max()))
    dist_margin_x = float(np.ceil(np.abs(distortion[1]).max()))
    corners_y, corners_x = [], []
    for dy, dx in offsets:
        corners_y += [0.0 - dy - dist_margin_y, (ny - 1.0) - dy + dist_margin_y]
        corners_x += [0.0 - dx - dist_margin_x, (nx - 1.0) - dx + dist_margin_x]
    origin = (min(corners_y), min(corners_x))
    out_ny = int(np.ceil((max(corners_y) - origin[0]) * scale_ratio)) + 2 * _GRID_MARGIN
    out_nx = int(np.ceil((max(corners_x) - origin[1]) * scale_ratio)) + 2 * _GRID_MARGIN
    origin = (origin[0] - _GRID_MARGIN / scale_ratio, origin[1] - _GRID_MARGIN / scale_ratio)

    driz = Drizzle(kernel=spec.final_kernel, out_shape=(out_ny, out_nx), fillval=0.0)
    total_exptime = 0.0
    grids = np.mgrid[0.0:ny, 0.0:nx]
    for frame, header, offset in zip(frames, headers, offsets):
        t_frame = float(header["ITIME"]) * int(header["COADDS"])
        if t_frame <= 0.0:
            raise ValueError(f"non-positive frame exposure time: {t_frame}")
        var_cps2 = _frame_background_variance_e(header, detector) / t_frame**2
        weight = np.where(np.isfinite(frame), 1.0 / var_cps2, 0.0)
        data_cps = np.nan_to_num(frame, nan=0.0) / t_frame
        pixmap = build_pixmap(
            frame.shape, distortion, offset, origin, scale_ratio,
            grids=(grids[0], grids[1]),
        )
        driz.add_image(
            data_cps,
            exptime=t_frame,
            pixmap=pixmap,
            weight_map=weight,
            pixfrac=spec.final_pixfrac,
            in_units="cps",
        )
        total_exptime += t_frame

    sci = driz.out_img.astype(np.float64)
    wht = driz.out_wht.astype(np.float64)

    # Output WCS: TAN at the target, detector-frame orientation scaled to the
    # final grid (absolute orientation is validated, not assumed — the
    # astrometric-parity check quantifies it; design doc, open items).
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [spec.ra, spec.dec]
    wcs.wcs.crpix = [out_nx / 2.0 + 0.5, out_ny / 2.0 + 0.5]
    scale_deg = spec.final_scale / 3600.0
    wcs.wcs.cd = [[-scale_deg, 0.0], [0.0, scale_deg]]

    header_out = wcs.to_header()
    header_out["EXPTIME"] = total_exptime
    header_out["BUNIT"] = "ELECTRONS/S"
    header_out["NCOMBINE"] = len(frames)

    output_root = Path(output_dir) / f"{spec.name}_{spec.filter_name}".lower()
    sci_path = Path(f"{output_root}_sci.fits")
    wht_path = Path(f"{output_root}_wht.fits")
    fits.PrimaryHDU(sci.astype(np.float32), header=header_out).writeto(
        sci_path, overwrite=True
    )
    fits.PrimaryHDU(wht.astype(np.float32), header=header_out).writeto(
        wht_path, overwrite=True
    )

    kwargs = {
        "kernel": spec.final_kernel,
        "pixfrac": spec.final_pixfrac,
        "final_scale": spec.final_scale,
        "backend": "drizzle.resample.Drizzle",
    }
    provenance = combine_provenance(
        spec,
        adapter,
        exposures,
        wht,
        kwargs_key="nirc2_kwargs",
        kwargs=kwargs,
        tail={
            "registration_offsets_native_pix": [
                [round(dy, 3), round(dx, 3)] for dy, dx in offsets
            ],
            "distortion_files": [
                Path(headers[0]["DISTX"]).name,
                Path(headers[0]["DISTY"]).name,
            ],
            "total_exptime": total_exptime,
            "out_shape": [out_ny, out_nx],
            # The frame<->mosaic mapping constants (issue #33): with the
            # offsets and the distortion tables these fully define the
            # transform set — frame products invert it per frame, and the
            # outlier pass resamples the mosaic through it.
            "origin": [float(origin[0]), float(origin[1])],
            "scale_ratio": float(scale_ratio),
            "sci_path": str(sci_path),
            "wht_path": str(wht_path),
        },
    )
    return sci_path, wht_path, provenance
