"""
The instrument-adapter boundary (design doc + roadmap): everything
instrument-specific lives behind an `InstrumentAdapter`; no module outside
`autoreduce.instruments` may name a detector.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class InstrumentAdapter:
    """Static description of one instrument/detector reduction path."""

    # Product domain: imaging adapters feed the image pipeline
    # (acquire→align→drizzle→noise→psf→package); visibility adapters feed
    # the visibility branch (docs/design/alma.md). The orchestrator
    # dispatches on this and nothing else.
    domain = "imaging"

    key: str  # registry key, e.g. "acs_wfc"
    mast_instrument_name: str  # e.g. "ACS/WFC" as MAST spells it
    native_scale: float  # arcsec / pix
    calibrated_suffix: str  # exposure product to reduce, e.g. "FLC"
    reference_env_key: str  # CRDS reference-path variable, e.g. "jref"
    crds_reference_subpath: str  # where CRDS syncs this instrument's files
    supports_cte_correction: bool
    default_drizzle_kwargs: Dict[str, object]
    saturation_dn: float  # conservative full-well / saturation level, electrons
    # The adapter's recommendation for TargetSpec.final_scale (which remains
    # the user-facing dial); documents sensible sampling for this detector.
    recommended_final_scale: float = 0.05
    # Observatory-level routing (phase 3): which archive/CRDS ecosystem and
    # which combination backend this instrument reduces through.
    observatory: str = "hst"  # "hst" | "jwst" | "keck"
    crds_server_url: str = "https://hst-crds.stsci.edu"
    combine_backend: str = "astrodrizzle"  # "astrodrizzle" | "jwst_image3" | "nirc2_native"
    mast_obs_collection: str = "HST"
    # Archive routing (phase 4): which archive the acquire stage queries.
    # Ground-based instruments (KOA) reduce from raw level-0 frames plus the
    # night's own calibrations, so they also run the pre-combine ground
    # stages (calibrate, sky) that space-based level-2 products make moot.
    archive: str = "mast"  # "mast" | "koa"
    # Detector constants for ground-based calibration/noise (gain, read
    # noise, dark). Adapter-owned so stages outside `instruments/` never
    # name a detector; None for space-based instruments, whose level-2
    # products carry calibrated units already.
    detector: object = None

    def ground_detector(self):
        """The detector constants, loud when a ground stage needs them."""
        if self.detector is None:
            raise ValueError(
                f"instrument {self.key!r} has no detector constants — "
                f"ground-based stages require them on the adapter"
            )
        return self.detector

    def scale_ratio(self, final_scale: float) -> float:
        """s = output scale / native scale, as used by the Casertano factor."""
        return final_scale / self.native_scale


@dataclass(frozen=True)
class VisibilityInstrumentAdapter:
    """
    Static description of one visibility-domain (interferometer) reduction
    path (docs/design/alma.md). Deliberately not a subclass of
    `InstrumentAdapter`: the imaging fields (drizzle kwargs, saturation,
    CRDS routing) have no visibility meaning, and a shared registry plus the
    `domain` dispatch is the whole contract between the two families.
    """

    domain = "visibility"

    key: str  # registry key, e.g. "alma"
    observatory: str  # "alma"
    archive: str  # which archive the acquire stage queries, e.g. "alma"


_REGISTRY: Dict[str, InstrumentAdapter] = {}


def register(adapter: InstrumentAdapter) -> InstrumentAdapter:
    if adapter.key in _REGISTRY:
        raise ValueError(f"instrument adapter already registered: {adapter.key}")
    _REGISTRY[adapter.key] = adapter
    return adapter


def get(key: str) -> InstrumentAdapter:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown instrument {key!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def registered_keys() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
