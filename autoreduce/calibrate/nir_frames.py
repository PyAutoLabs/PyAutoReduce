"""
NIR frame calibration: DN -> e-, dark, flat, bad-pixel mask.

The recipe is the SHARP one (flat + sky are the essential steps; sky frames
carry the dark signal, so a master dark is used when the night has matched
darks and skipped — recorded, never silently — when it does not). A NIRC2
raw frame is ITIME seconds x COADDS coadds, stored as the per-coadd average
in DN; calibration converts to total electrons so Poisson statistics stay
computable downstream.

Bad pixels are found from the calibration frames themselves (hot in the
dark, dead in the flat) and carried as a mask; they enter combination with
zero weight rather than being interpolated over — the drizzle coverage
handles them exactly as it handles CR-rejected pixels on HST.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CalibrationFrames:
    """Master calibrations for one (camera, filter, itime, coadds) setup."""

    master_flat: np.ndarray
    master_dark: Optional[np.ndarray]  # total e- for the science ITIMExCOADDS
    bad_pixel_mask: np.ndarray  # True = bad
    provenance: Dict


def _median_stack(frames: List[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("cannot stack an empty frame list")
    return np.median(np.stack(frames), axis=0)


def build_calibrations(
    dark_frames: List[np.ndarray],
    flat_on_frames: List[np.ndarray],
    flat_off_frames: Optional[List[np.ndarray]] = None,
    hot_sigma: float = 5.0,
    dead_flat_threshold: float = 0.5,
) -> CalibrationFrames:
    """
    Build the master calibrations from raw calibration frames (DN).

    Flats: median of lamp-on frames, minus median of lamp-off frames when
    the night has them (removes the thermal pedestal K-band dome flats
    carry), normalised to unit median. Darks: median stack, kept in DN here
    and scaled to electrons by the caller (gain lives with the detector
    constants). Bad pixels: hot in the dark (> hot_sigma above the median)
    or unresponsive in the flat (< dead_flat_threshold of unit response).
    """
    flat = _median_stack(flat_on_frames)
    if flat_off_frames:
        flat = flat - _median_stack(flat_off_frames)
    flat_median = np.median(flat)
    if not np.isfinite(flat_median) or flat_median <= 0.0:
        raise ValueError(
            f"master flat has non-positive median ({flat_median}); the flat "
            f"set is unusable — fix acquisition, don't normalise garbage"
        )
    flat = flat / flat_median

    dead = flat < dead_flat_threshold

    master_dark = None
    hot = np.zeros_like(dead)
    if dark_frames:
        from ..noise.rms import mad_sigma

        master_dark = _median_stack(dark_frames)
        centre = np.median(master_dark)
        spread = mad_sigma(master_dark)
        if spread > 0:
            hot = master_dark > centre + hot_sigma * spread

    bad = dead | hot
    provenance = {
        "n_dark_frames": len(dark_frames),
        "n_flat_on_frames": len(flat_on_frames),
        "n_flat_off_frames": len(flat_off_frames or []),
        "n_bad_pixels": int(bad.sum()),
        "n_hot_pixels": int(hot.sum()),
        "n_dead_pixels": int(dead.sum()),
        "dark_subtraction": master_dark is not None,
    }
    return CalibrationFrames(
        master_flat=flat,
        master_dark=master_dark,
        bad_pixel_mask=bad,
        provenance=provenance,
    )


def calibrate_frame(
    raw_dn: np.ndarray,
    calib: CalibrationFrames,
    gain_e_per_dn: float,
    coadds: int,
) -> np.ndarray:
    """
    One raw frame (per-coadd-average DN) -> total electrons, dark-subtracted
    (when available) and flat-fielded. Bad pixels are NaN'd so no downstream
    stage can use them by accident; combination gives them zero weight.
    """
    if gain_e_per_dn <= 0.0:
        raise ValueError(f"gain must be positive: {gain_e_per_dn}")
    if coadds < 1:
        raise ValueError(f"coadds must be >= 1: {coadds}")
    electrons = raw_dn.astype(np.float64) * gain_e_per_dn * coadds
    if calib.master_dark is not None:
        electrons = electrons - calib.master_dark * gain_e_per_dn * coadds
    with np.errstate(divide="ignore", invalid="ignore"):
        electrons = electrons / calib.master_flat
    electrons[calib.bad_pixel_mask] = np.nan
    return electrons


def load_calibration_sets(
    calib_paths: List[Path],
    science_itime: float,
    science_coadds: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Sort downloaded calibration frames into (darks, flat_on, flat_off) by
    header type, keeping only darks that match the science ITIME/COADDS.
    """
    from astropy.io import fits

    darks, flat_on, flat_off = [], [], []
    for path in calib_paths:
        with fits.open(path) as hdul:
            header = hdul[0].header
            data = hdul[0].data.astype(np.float64)
        imtype = str(header.get("KOAIMTYP", "")).lower()
        if imtype == "dark":
            if (
                abs(float(header.get("ITIME", -1)) - science_itime) < 0.005
                and int(header.get("COADDS", -1)) == science_coadds
            ):
                darks.append(data)
        elif imtype in ("flatlamp", "domeflat"):
            flat_on.append(data)
        elif imtype == "flatlampoff":
            flat_off.append(data)
    return darks, flat_on, flat_off
