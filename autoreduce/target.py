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

    def __post_init__(self):
        if not -360.0 <= self.ra <= 360.0:
            raise ValueError(f"ra out of range: {self.ra}")
        if not -90.0 <= self.dec <= 90.0:
            raise ValueError(f"dec out of range: {self.dec}")
        if not 0.0 < self.final_pixfrac <= 1.0:
            raise ValueError(f"final_pixfrac must be in (0, 1]: {self.final_pixfrac}")
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
        if raw.get("proposal_ids") is not None:
            raw["proposal_ids"] = tuple(str(p) for p in raw["proposal_ids"])
        return cls(**raw)

    def as_dict(self) -> dict:
        return asdict(self)
