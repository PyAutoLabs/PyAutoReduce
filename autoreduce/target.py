"""
Target specification — the declarative input of a reduction.

A reduction is a pure function of a `TargetSpec` plus the archive (design doc
stage 0): re-running the pipeline on the same spec reproduces the dataset,
modulo upstream reference-file updates, which `reduction.json` records.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple

import yaml


@dataclass(frozen=True)
class TargetSpec:
    """Everything the pipeline needs to know about one target."""

    name: str
    ra: float  # degrees
    dec: float  # degrees
    instrument: str = "acs_wfc"
    filter_name: str = "F814W"

    # Restrict acquisition to these proposal IDs (None = all direct
    # calibration-level-2 observations at the coordinates).
    proposal_ids: Optional[Tuple[str, ...]] = None

    cutout_shape: Tuple[int, int] = (281, 281)

    # Drizzle dials (design doc stage 3). pixfrac and kernel are deliberately
    # user-facing: published practice spans no-drizzle -> 0.6 -> 1.0, so the
    # choice is configuration, never a buried default.
    final_scale: float = 0.05  # arcsec / pix
    final_pixfrac: float = 0.8
    final_kernel: str = "square"

    # PSF products (design doc stage 5).
    psf_shape: Tuple[int, int] = (21, 21)
    psf_full_shape: Tuple[int, int] = (61, 61)

    # Opt-in per-exposure frame products (roadmap "Per-exposure frame
    # products"): additionally package every calibrated _flc/_flt chip as a
    # modeling-ready native-pixel product set under <out>/frames/. HST-only;
    # off by default — deepCR per-frame cosmic-ray masking is slow.
    frame_products: bool = False

    # Build the mosaic psf.fits from the per-frame tier-1 ePSFs (drop-
    # convolved, resampled through each frame's geometry, exptime-weighted)
    # instead of estimating on the drizzled mosaic. HST-only; independent of
    # frame_products (the frame ePSFs are built either way).
    psf_from_frames: bool = False

    # PSF back-end for the mosaic path: "epsf" (default, photutils Tier-1) or
    # "starred" (Tier-1b super-sampled ePSF from the same field stars; requires
    # the optional `pyautoreduce[starred]` extra — GPL/JAX, isolated). Ignored
    # when psf_from_frames or the Keck tier-A path own the PSF stage.
    psf_backend: str = "epsf"

    # Alignment: residual (pixels) above which TweakReg refinement triggers.
    alignment_tolerance_pix: float = 0.1

    # Synthetic-source injection (docs/design/simulate.md, phase 1 —
    # HST astrodrizzle path only). Path to a plain FITS image whose pixel
    # values are e-/s per input pixel; None = no injection.
    inject_image: Optional[str] = None
    # arcsec/pix of the input image; required with inject_image.
    inject_pixel_scale: Optional[float] = None
    # (ra, dec) degrees the input image is centred on; None = the target.
    inject_position: Optional[Tuple[float, float]] = None
    # PSF FITS (native scale, odd shape) convolved with the rendered input
    # on every frame; None = each frame's tier-1 ePSF (loud when the star
    # field cannot support one).
    inject_psf: Optional[str] = None
    # Seed for the injected source's per-frame Poisson realisations.
    inject_seed: int = 0

    # Visibility-domain (ALMA) additions — ignored by imaging instruments,
    # required by the visibility branch (docs/design/alma.md). The imaging
    # dials above (cutout, drizzle, psf shapes) are ignored in return.
    # Execution-block uids pinning the measurement sets, e.g.
    # ("A002_Xb9b1b9_X3046",).
    alma_uids: Optional[Tuple[str, ...]] = None
    # The science field name inside the MS, e.g. "G09v1.40".
    alma_field: Optional[str] = None
    # Spectral windows to extract, e.g. ("1", "2") — line-bearing spws are
    # simply left out for continuum work.
    alma_spws: Optional[Tuple[str, ...]] = None
    # Channel-averaging width; 0 = collapse each spw fully (the continuum
    # default).
    alma_width: int = 0
    # Directory of calibrated per-uid MS (uid___<uid>.ms.split.cal) from an
    # ARC delivery / scriptForPI restore; None = acquire from the archive.
    alma_ms_dir: Optional[str] = None
    # ALMA project code for archive acquisition, e.g. "2016.1.00282.S".
    alma_project_code: Optional[str] = None

    # Survey-cutout additions (docs/design/surveys.md) — ignored by the
    # imaging and visibility branches. Bands to fetch; None = the
    # adapter's defaults.
    survey_bands: Optional[Tuple[str, ...]] = None

    # simobserve acquire-alternative (docs/design/simulate.md phase 3) —
    # active when inject_image is set on a visibility-domain instrument.
    # Array configuration file name (casatools data repo).
    alma_sim_antennalist: str = "alma.cycle8.3.cfg"
    alma_sim_totaltime_s: float = 1800.0
    alma_sim_integration_s: float = 10.0
    alma_sim_freq_ghz: float = 230.0
    # Precipitable water vapour for tsys-atm thermal noise; 0 = noiseless.
    alma_sim_pwv_mm: float = 0.5

    # Ground-based (KOA) additions — ignored by space-based instruments.
    # Explicit KOA identifiers pin the science frame set exactly (the raw
    # archive has no association tables); None = query by coords + program.
    koa_science_ids: Optional[Tuple[str, ...]] = None
    # PSF-star frames from the same program/night, reduced pipeline-identically
    # into candidate PSF products (tier A; docs/design/keck_ao.md stage 6).
    koa_psf_star_ids: Optional[Tuple[str, ...]] = None
    # Running-sky window: number of temporally adjacent frames the per-frame
    # sky is medianed over (K'-band sky varies on minutes timescales).
    sky_window: int = 9

    def __post_init__(self):
        if not -360.0 <= self.ra <= 360.0:
            raise ValueError(f"ra out of range: {self.ra}")
        if not -90.0 <= self.dec <= 90.0:
            raise ValueError(f"dec out of range: {self.dec}")
        if not 0.0 < self.final_pixfrac <= 1.0:
            raise ValueError(f"final_pixfrac must be in (0, 1]: {self.final_pixfrac}")
        if self.alma_width < 0:
            raise ValueError(
                f"alma_width must be >= 0 (0 = collapse the spw): {self.alma_width}"
            )
        for shape_name in ("cutout_shape", "psf_shape", "psf_full_shape"):
            shape = getattr(self, shape_name)
            if len(shape) != 2 or any(s < 1 for s in shape):
                raise ValueError(f"{shape_name} must be two positive ints: {shape}")
        for shape_name in ("psf_shape", "psf_full_shape"):
            shape = getattr(self, shape_name)
            if any(s % 2 == 0 for s in shape):
                raise ValueError(
                    f"{shape_name} must be odd so the PSF has a centre pixel: {shape}"
                )
        for name in (
            "alma_sim_totaltime_s", "alma_sim_integration_s", "alma_sim_freq_ghz"
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive: {getattr(self, name)}")
        if self.alma_sim_pwv_mm < 0.0:
            raise ValueError(f"alma_sim_pwv_mm must be >= 0: {self.alma_sim_pwv_mm}")
        if self.inject_image is not None:
            if self.inject_pixel_scale is None or self.inject_pixel_scale <= 0.0:
                raise ValueError(
                    "inject_image requires a positive inject_pixel_scale "
                    f"(arcsec/pix of the input image): {self.inject_pixel_scale}"
                )
            if self.inject_position is not None and len(self.inject_position) != 2:
                raise ValueError(
                    f"inject_position must be (ra, dec) degrees: {self.inject_position}"
                )
        elif any(
            getattr(self, k) is not None
            for k in ("inject_pixel_scale", "inject_position", "inject_psf")
        ):
            raise ValueError(
                "inject_pixel_scale/inject_position/inject_psf are set but "
                "inject_image is None — injection dials require an input image"
            )

    @classmethod
    def from_yaml(cls, path) -> "TargetSpec":
        with open(path) as f:
            raw = yaml.safe_load(f)
        for key in ("cutout_shape", "psf_shape", "psf_full_shape"):
            if key in raw:
                raw[key] = tuple(raw[key])
        if raw.get("inject_position") is not None:
            raw["inject_position"] = tuple(float(v) for v in raw["inject_position"])
        for key in (
            "proposal_ids",
            "koa_science_ids",
            "koa_psf_star_ids",
            "alma_uids",
            "alma_spws",
            "survey_bands",
        ):
            if raw.get(key) is not None:
                raw[key] = tuple(str(p) for p in raw[key])
        return cls(**raw)

    def as_dict(self) -> dict:
        return asdict(self)
