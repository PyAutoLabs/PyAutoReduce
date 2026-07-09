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

    # Alignment: residual (pixels) above which TweakReg refinement triggers.
    alignment_tolerance_pix: float = 0.1

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

    @classmethod
    def from_yaml(cls, path) -> "TargetSpec":
        with open(path) as f:
            raw = yaml.safe_load(f)
        for key in ("cutout_shape", "psf_shape", "psf_full_shape"):
            if key in raw:
                raw[key] = tuple(raw[key])
        for key in (
            "proposal_ids",
            "koa_science_ids",
            "koa_psf_star_ids",
            "alma_uids",
            "alma_spws",
        ):
            if raw.get(key) is not None:
                raw[key] = tuple(str(p) for p in raw[key])
        return cls(**raw)

    def as_dict(self) -> dict:
        return asdict(self)
