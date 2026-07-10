"""
Keck NIRC2 per-exposure frame products (issue #33; feasibility in
keck_ao.md §"Per-exposure frame products").

The ground path has no archive-calibrated per-frame files — the packager
consumes the pipeline's own *prepared* frames (calibrated to electrons,
running-sky-subtracted, bad pixels as NaN, headers carrying the per-frame
facts: ITIME/COADDS/MJD-OBS/SAMPMODE/MULTISAM/SKYLEV/DISTX/DISTY). Three
things differ structurally from the HST/JWST branches:

- **Registration is offset-based.** NIRC2 header WCS is arcsecond-grade;
  the combine's measured phase-correlation offsets ARE the registration,
  and together with the distortion tables and the recorded mapping
  constants (origin, scale_ratio) they fully define the frame↔mosaic
  transform. No per-frame WCS is shipped — the manifest carries the
  offsets and the transform constants instead.
- **Noise is constructed, not read** — from the same recorded facts the
  combine's IVM weights use (sky + dark + MCDS-aware read noise) plus
  source Poisson.
- **Cosmic rays come from the frame-vs-stack outlier pass** — the mosaic
  resampled back onto each frame's native grid through the same pixmap
  arithmetic, robustly rescaled, with positive deviants above ``k·σ``
  flagged. This is the mask-generation half of the stack-level "CR
  rejection at combine" open item.

Per-frame PSFs are native-pixel stamps of the temporally nearest
*accepted* tier-A PSF-star frame (the drizzled epoch candidates live on
the final grid — the wrong basis for native frames). The tier-A vetting
verdicts are inherited; ``psf_provisional`` stays true, as everywhere on
the AO path. Products convert to e-/s so frames and mosaic share the cps
flux scale (per-frame ITIME×COADDS recorded).

Known caveat riding every product: the adapter's plate scale is under
revision (epoch-aware fix tracked by the acceptance task); recorded
``native_scale`` values inherit it.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..instruments import InstrumentAdapter
from ..noise import rms as rms_mod
from ..target import TargetSpec
from .frames import MANIFEST_VERSION, frame_cutout_shape

OUTLIER_SIGMA = 5.0


def _frame_noise_map_e(frame_e: np.ndarray, header, detector) -> np.ndarray:
    """Constructed per-frame sigma (e-): background variance from the same
    recorded facts the combine weights use, plus source Poisson."""
    from ..drizzle.nirc2_combine import _frame_background_variance_e

    var_bkg = _frame_background_variance_e(header, detector)
    return np.sqrt(np.clip(frame_e, 0.0, None) + var_bkg)


def _mosaic_coords(yy, xx, distortion, offset, origin, scale_ratio):
    """The combine's forward pixmap arithmetic for arbitrary frame pixels."""
    my = (yy + distortion[0][yy, xx] - offset[0] - origin[0]) * scale_ratio
    mx = (xx + distortion[1][yy, xx] - offset[1] - origin[1]) * scale_ratio
    return my, mx


def _target_position_in_frame(
    target_mosaic_yx, offset, origin, scale_ratio, distortion, shape
) -> Tuple[float, float]:
    """Invert the pixmap arithmetic at one point — fixed-point iteration
    over the smooth distortion tables."""
    ty, tx = target_mosaic_yx
    base_y = ty / scale_ratio + origin[0] + offset[0]
    base_x = tx / scale_ratio + origin[1] + offset[1]
    y, x = base_y, base_x
    ny, nx = shape
    for _ in range(3):
        iy = int(np.clip(round(y), 0, ny - 1))
        ix = int(np.clip(round(x), 0, nx - 1))
        y = base_y - distortion[0][iy, ix]
        x = base_x - distortion[1][iy, ix]
    return float(y), float(x)


def _stamp(frame: np.ndarray, centre_yx, shape) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Target-centred stamp with NaN padding where it hangs off the frame
    (no WCS machinery on this path — plain index arithmetic)."""
    cy, cx = int(round(centre_yx[0])), int(round(centre_yx[1]))
    hy, hx = shape[0] // 2, shape[1] // 2
    out = np.full(shape, np.nan)
    y0, y1 = cy - hy, cy + hy + 1
    x0, x1 = cx - hx, cx + hx + 1
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y1, frame.shape[0]), min(x1, frame.shape[1])
    if sy1 <= sy0 or sx1 <= sx0:
        raise ValueError("stamp does not overlap the frame")
    out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return out, (y0, x0)


def _outlier_mask(
    stamp_e: np.ndarray,
    model_e: np.ndarray,
    sigma_e: np.ndarray,
    k: float = OUTLIER_SIGMA,
) -> np.ndarray:
    """Positive deviants vs the (robustly rescaled) stack prediction.

    The robust scale factor self-calibrates flux conventions (cps
    normalisation, plate-scale area terms) out of the comparison; cosmic
    rays are positive by nature, so only positive residuals flag.
    """
    good = (
        np.isfinite(stamp_e)
        & np.isfinite(model_e)
        & np.isfinite(sigma_e)
        & (sigma_e > 0)
    )
    if not good.any():
        return np.zeros(stamp_e.shape, dtype=bool)
    bright = good & (model_e > np.nanmedian(np.abs(model_e[good])) * 3)
    if bright.sum() >= 25:
        scale = float(np.median(stamp_e[bright] / model_e[bright]))
    else:
        scale = 1.0
    resid = stamp_e - model_e * scale
    return good & (resid > k * sigma_e)


def _match_psf_star(
    mjd: float,
    star_frames: Sequence[np.ndarray],
    star_mjds: Sequence[float],
    accepted_epochs: Sequence[Dict],
) -> Optional[Tuple[np.ndarray, int, float]]:
    """The temporally nearest star frame belonging to an ACCEPTED tier-A
    epoch; (frame, epoch, gap_days) or None when no accepted epoch exists."""
    from ..psf.nirc2_star import group_epochs

    if not star_frames or not accepted_epochs:
        return None
    groups = group_epochs(list(star_mjds))
    accepted = {int(c["epoch"]) for c in accepted_epochs}
    best = None
    for epoch, indices in enumerate(groups):
        if epoch not in accepted:
            continue
        for idx in indices:
            gap = abs(float(star_mjds[idx]) - mjd)
            if best is None or gap < best[2]:
                best = (star_frames[idx], epoch, gap)
    return best


def package_keck_frame_products(
    exposures: List[Path],
    spec: TargetSpec,
    adapter: InstrumentAdapter,
    out_dir: Path,
    drizzle_prov: Dict,
    psf_star_frames: Sequence[np.ndarray],
    psf_star_mjds: Sequence[float],
    psf_record: Dict,
) -> Dict:
    """Package every prepared NIRC2 frame; returns the provenance fragment."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy.ndimage import map_coordinates

    from ..drizzle.nirc2_combine import load_distortion
    from ..psf.epsf import normalise_kernel
    from ..psf.nirc2_star import _centre_on_peak

    frames_dir = Path(out_dir) / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    detector = adapter.ground_detector()
    offsets = drizzle_prov["registration_offsets_native_pix"]
    origin = tuple(drizzle_prov["origin"])
    scale_ratio = float(drizzle_prov["scale_ratio"])
    mosaic = fits.getdata(drizzle_prov["sci_path"]).astype(float)  # e-/s
    mosaic_wcs = WCS(fits.getheader(drizzle_prov["sci_path"]))
    tx, ty = mosaic_wcs.world_to_pixel_values(spec.ra, spec.dec)
    target_mosaic = (float(ty), float(tx))

    shape = frame_cutout_shape(
        spec.cutout_shape, spec.final_scale, adapter.native_scale
    )
    accepted = psf_record.get("candidates", [])

    entries, distortion = [], None
    for path, offset in zip(exposures, offsets):
        with fits.open(path) as hdul:
            frame = hdul[0].data.astype(float)
            header = hdul[0].header.copy()
        if distortion is None:
            distortion = load_distortion(
                Path(header["DISTX"]), Path(header["DISTY"]), frame.shape
            )
        t_frame = float(header["ITIME"]) * int(header["COADDS"])
        centre = _target_position_in_frame(
            target_mosaic, offset, origin, scale_ratio, distortion, frame.shape
        )
        stamp_e, stamp_origin = _stamp(frame, centre, shape)
        sigma_e = _frame_noise_map_e(stamp_e, header, detector)

        # Frame-vs-stack outlier pass on the stamp region.
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        fy = np.clip(yy + stamp_origin[0], 0, frame.shape[0] - 1)
        fx = np.clip(xx + stamp_origin[1], 0, frame.shape[1] - 1)
        my, mx = _mosaic_coords(fy, fx, distortion, offset, origin, scale_ratio)
        model_e = (
            map_coordinates(mosaic, [my, mx], order=1, mode="constant", cval=np.nan)
            * t_frame
        )
        outliers = _outlier_mask(stamp_e, model_e, sigma_e)

        bad = ~np.isfinite(stamp_e) | ~np.isfinite(sigma_e) | (sigma_e <= 0)
        masked = bad | outliers
        data_cps = np.where(masked, 0.0, stamp_e) / t_frame
        noise_cps = np.where(masked, rms_mod.MASKED_NOISE_VALUE * t_frame, sigma_e) / t_frame

        stem = Path(path).stem
        chip_dir = frames_dir / stem
        chip_dir.mkdir(parents=True)
        fits.PrimaryHDU(data_cps.astype(np.float32)).writeto(
            chip_dir / "data.fits", overwrite=True
        )
        fits.PrimaryHDU(noise_cps.astype(np.float32)).writeto(
            chip_dir / "noise_map.fits", overwrite=True
        )
        fits.PrimaryHDU(outliers.astype(np.uint8)).writeto(
            chip_dir / "outlier_mask.fits", overwrite=True
        )

        match = _match_psf_star(
            float(header["MJD-OBS"]), psf_star_frames, psf_star_mjds, accepted
        )
        if match is not None:
            star, epoch, gap = match
            half = max(spec.psf_full_shape) // 2 + 10
            cut = _centre_on_peak(np.nan_to_num(star, nan=0.0), half)
            psf_full = normalise_kernel(cut, spec.psf_full_shape)
            psf = normalise_kernel(cut, spec.psf_shape)
            fits.PrimaryHDU(psf.astype(np.float32)).writeto(
                chip_dir / "psf.fits", overwrite=True
            )
            fits.PrimaryHDU(psf_full.astype(np.float32)).writeto(
                chip_dir / "psf_full.fits", overwrite=True
            )
            psf_diag = {
                "method": "tier-A star frame stamp (native pixels)",
                "psf_provisional": True,
                "epoch": int(epoch),
                "mjd_gap_minutes": float(gap) * 1440.0,
            }
        else:
            psf_diag = {
                "method": "none",
                "reason": "no accepted tier-A PSF epoch to match",
                "psf_provisional": True,
            }

        entries.append(
            {
                "frame": stem,
                "dir": chip_dir.name,
                "itime_x_coadds_s": t_frame,
                "mjd": float(header["MJD-OBS"]),
                "sky_subtracted_e": float(header["SKYLEV"]),
                "unit_conversion": "e- / (ITIME x COADDS)",
                "data_units": "ELECTRONS/S",
                "target_pixel": [
                    centre[0] - stamp_origin[0],
                    centre[1] - stamp_origin[1],
                ],
                "registration": {
                    "method": (
                        "phase cross-correlation offsets measured at combine "
                        "(header WCS is arcsecond-grade and untrusted)"
                    ),
                    "offset_dy_px": float(offset[0]),
                    "offset_dx_px": float(offset[1]),
                },
                "n_masked_pixels": int(masked.sum()),
                "n_outlier_pixels": int(outliers.sum()),
                "psf": psf_diag,
            }
        )

    if not entries:
        raise ValueError(
            f"keck frame_products wrote no frames for {spec.name} — empty "
            "prepared-frame list is an upstream bug"
        )

    manifest = {
        "version": MANIFEST_VERSION,
        "target": {"name": spec.name, "ra": spec.ra, "dec": spec.dec},
        "data_units": "ELECTRONS/S",
        "source": "pipeline-prepared frames (calibrated, running-sky-subtracted)",
        "frame_cutout_shape": list(shape),
        "native_scale": adapter.native_scale,
        "native_scale_note": (
            "plate scale under revision (epoch-aware fix tracked by the "
            "acceptance task); recorded values inherit the current adapter"
        ),
        "cr_method": {
            "method": (
                f"frame-vs-stack outlier pass (> {OUTLIER_SIGMA} sigma "
                "positive residual vs the resampled mosaic)"
            )
        },
        "registration_note": (
            "no per-frame WCS on the AO path: the measured offsets plus the "
            "distortion tables and mapping constants (origin, scale_ratio in "
            "the drizzle provenance) fully define the frame<->mosaic "
            "transform; offsets are exact by construction at the combine's "
            "sub-pixel correlation accuracy"
        ),
        "distortion_files": drizzle_prov.get("distortion_files"),
        "frames": entries,
    }
    with open(frames_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    n_psf = sum(1 for e in entries if e["psf"]["method"] != "none")
    print(
        f"[frames] keck frame products ({spec.name}): {len(entries)} frames, "
        f"{n_psf} with epoch-matched native PSF stamps, "
        f"{sum(e['n_outlier_pixels'] for e in entries)} outlier px flagged "
        "(frame-vs-stack pass). Registration = measured offsets (header WCS "
        "untrusted); plate-scale caveat in the manifest."
    )

    return {
        "n_frames": len(entries),
        "n_frames_with_psf": n_psf,
        "data_units": "ELECTRONS/S",
        "cr_method": manifest["cr_method"],
        "manifest": "frames/manifest.json",
    }
