"""
Keck II / NIRC2 behind LGS/NGS AO — adapters #6 and #7 (phase 4), the first
ground-based path.

There is no maintained community pipeline to wrap (KAI is Python 2.7 + IRAF),
so the ground stages are implemented natively and validated against the SHARP
programme's published practice (docs/design/keck_ao.md). Reduction runs from
KOA raw level-0 frames: calibrate (DN -> e-, dark, flat, bad pixels), running
sky subtraction, then dewarp+combine through the ``drizzle`` package with the
published geometric-distortion solution as the pixel mapping.

Distortion solutions are narrow-camera lookup tables selected by observation
date: Yelda et al. (2010) before the 2015-04-13 AO-bench servicing, Service
et al. (2016) after. They are fetched into the references cache at acquire
time (FITS is never committed to this repo) and recorded in provenance. The
wide camera is registered for spec completeness but fails loudly at combine
until a wide-camera solution is adopted (design doc, open items).

Detector constants are adapter-owned and validated by the blank-sky noise
closure rather than trusted blindly: gain 4.0 e-/DN, CDS read noise 38 e-
(effective per-frame read noise scales 1/sqrt(coadds) for coadded frames),
dark current ~0.1 e-/s.
"""

from dataclasses import dataclass
from typing import Tuple

from .adapter import InstrumentAdapter, register

# The 2015-04-13 NIRC2/AO-bench servicing boundary (Service et al. 2016).
DISTORTION_EPOCH_BOUNDARY_MJD = 57125.0

# Narrow-camera distortion lookup tables (x/y shifts in native pixels;
# rectified = observed + shift), by epoch, from the canonical distribution
# the Keck dewarp page points at (github.com/jluastro/nirc2_distortion).
# Synced into the references cache by acquire.koa.
_DIST_ROOT = "https://raw.githubusercontent.com/jluastro/nirc2_distortion/master"
DISTORTION_SOLUTIONS = {
    "yelda2010": (
        f"{_DIST_ROOT}/nirc2_distort_X_pre20150413_v1.fits",
        f"{_DIST_ROOT}/nirc2_distort_Y_pre20150413_v1.fits",
    ),
    "service2016": (
        f"{_DIST_ROOT}/nirc2_distort_X_post20150413_v1.fits",
        f"{_DIST_ROOT}/nirc2_distort_Y_post20150413_v1.fits",
    ),
}


def distortion_solution_for_mjd(mjd: float) -> str:
    """Route an observation date to its distortion-solution epoch."""
    if mjd <= 0.0:
        raise ValueError(f"MJD must be positive: {mjd}")
    return "yelda2010" if mjd < DISTORTION_EPOCH_BOUNDARY_MJD else "service2016"


@dataclass(frozen=True)
class Nirc2Detector:
    """NIRC2 detector constants shared by both cameras (one physical array)."""

    gain_e_per_dn: float = 4.0
    read_noise_e_cds: float = 38.0
    dark_e_per_s: float = 0.1
    shape: Tuple[int, int] = (1024, 1024)

    def read_noise_e(self, sampmode: int, multisam: int) -> float:
        """
        Effective read noise per frame for the header's sampling mode:
        CDS (SAMPMODE 2) reads at the CDS value; MCDS/Fowler-M (SAMPMODE 3)
        averages M read pairs, cutting the variance by ~1/M. Validated on
        SHARP B1938 K' frames (MCDS-32): budget 62.0 e- vs empirical
        62-64 e- per frame, where the CDS value would read 72.
        """
        if sampmode == 3 and multisam >= 1:
            return self.read_noise_e_cds / multisam**0.5
        return self.read_noise_e_cds


NIRC2_DETECTOR = Nirc2Detector()

_COMMON = dict(
    mast_instrument_name="N/A (KOA)",  # not a MAST instrument
    calibrated_suffix="RAW",  # KOA level-0; calibration is the pipeline's job
    reference_env_key="",  # no CRDS analogue; distortion synced by acquire.koa
    crds_reference_subpath="references/keck",
    supports_cte_correction=False,
    observatory="keck",
    crds_server_url="",
    combine_backend="nirc2_native",
    mast_obs_collection="",
    archive="koa",
    detector=NIRC2_DETECTOR,
    default_drizzle_kwargs={},  # the native backend reads TargetSpec dials
    saturation_dn=18_000.0 * 4.0,  # shallow NIR well: ~18 kDN linearity limit
)

NIRC2_NARROW = register(
    InstrumentAdapter(
        key="nirc2_narrow",
        native_scale=0.009942,
        recommended_final_scale=0.010,  # SHARP convention (Chen et al. 2019)
        **_COMMON,
    )
)

NIRC2_WIDE = register(
    InstrumentAdapter(
        key="nirc2_wide",
        native_scale=0.039686,
        recommended_final_scale=0.040,
        **_COMMON,
    )
)
