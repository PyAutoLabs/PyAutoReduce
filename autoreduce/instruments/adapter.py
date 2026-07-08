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

    key: str  # registry key, e.g. "acs_wfc"
    mast_instrument_name: str  # e.g. "ACS/WFC" as MAST spells it
    native_scale: float  # arcsec / pix
    calibrated_suffix: str  # exposure product to reduce, e.g. "FLC"
    reference_env_key: str  # CRDS reference-path variable, e.g. "jref"
    supports_cte_correction: bool
    default_drizzle_kwargs: Dict[str, object]
    saturation_dn: float  # conservative full-well / saturation level

    def scale_ratio(self, final_scale: float) -> float:
        """s = output scale / native scale, as used by the Casertano factor."""
        return final_scale / self.native_scale


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
