"""
The instrument-adapter boundary (design doc + roadmap): everything
instrument-specific lives behind an `InstrumentAdapter`; no module outside
`autoreduce.instruments` may name a detector.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union


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
    # Synthetic-source injection (docs/design/simulate.md): the units the
    # inject_image pixel values carry for this instrument — "electrons/s"
    # (HST; detector-referred flux) or "Jy" (JWST; the MJy/sr conversion
    # goes through the frame's own PIXAR_SR, flux-exact and gain-free).
    inject_units: str = "electrons/s"
    # Nominal detector gain (e-/DN) — needed only to size the injected
    # source's Poisson draw on surface-brightness frames (MJy/sr); the
    # injected MEAN never depends on it. None where unused (HST e-/s
    # frames are already electron-referred).
    e_per_dn: Optional[float] = None
    # Which DQ bits AstroDrizzle should treat as GOOD, as a function of
    # exposure count (issue #65). Rows are
    # ``(min_exposures, driz_sep_bits, final_bits)`` in ascending order and
    # carry MDRIZTAB's own semantics: the applicable row is the last one whose
    # ``min_exposures <= n``. Mirrors STScI's shipped MDRIZTAB reference files
    # rather than enabling ``mdriztab=True``, which would also import
    # final_scale/final_pixfrac/final_kernel/final_rot and silently revert the
    # deliberate lensing deviations in hst_acs_pipeline.md stage 3.
    #
    # None = emit no bits keywords at all, leaving the backend's own default
    # untouched — correct for the non-AstroDrizzle backends (jwst_image3,
    # nirc2_native), which have no such keyword.
    dq_bits_rows: Optional[Tuple[Tuple[int, int, int], ...]] = None

    def dq_bits_for(self, n_exposures: int) -> Optional[Tuple[int, int]]:
        """
        ``(driz_sep_bits, final_bits)`` for this exposure count, or None when
        the instrument declares no table.

        The value is genuinely N-dependent: single-exposure data uses 65535
        (every bit good) because with one exposure there is nothing to fill a
        masked pixel with, so the standard recipe keeps flagged pixels rather
        than punching holes. A flat constant would be wrong.
        """
        if self.dq_bits_rows is None:
            return None
        if n_exposures < 1:
            raise ValueError(f"need at least one exposure, got {n_exposures}")
        applicable = [row for row in self.dq_bits_rows if row[0] <= n_exposures]
        if not applicable:
            raise ValueError(
                f"instrument {self.key!r} has no DQ-bits row covering "
                f"{n_exposures} exposure(s); rows start at "
                f"{self.dq_bits_rows[0][0]}"
            )
        _, driz_sep_bits, final_bits = applicable[-1]
        return driz_sep_bits, final_bits

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

    def max_single_exposure_seconds(self, headers) -> Optional[float]:
        """
        The longest single-exposure time in the stack, or None when no
        per-exposure full-well cut applies. Saturation is per exposure,
        not per stack: a star saturates when its rate fills the well
        within one exposure, so peak caps divide the full well by this —
        never the mosaic total.

        HST reads EXPTIME; Keck frames integrate ITIME x COADDS. JWST
        mosaics are in surface-brightness units (MJy/sr) where a
        full-well cut is meaningless; saturated cores arrive as NaN/DQ-
        blank from the level-2 pipeline (refinement tracked in
        docs/design/jwst.md open items).
        """
        if self.observatory == "hst":
            t = max(float(h.get("EXPTIME", 0.0)) for h in headers)
            if t <= 0.0:
                raise ValueError("no exposure carries a positive EXPTIME header")
            return t
        if self.observatory == "keck":
            return max(
                float(h["ITIME"]) * int(h["COADDS"]) for h in headers
            )
        return None


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


@dataclass(frozen=True)
class SurveyCutoutAdapter:
    """
    Static description of one survey cutout-service path (docs/design/
    surveys.md). Third domain alongside imaging and visibility: these
    surveys deliver *pre-reduced* coadds through public services, so the
    branch fetches and packages — it never reduces. Products are colour
    context, not modeling inputs; noise and PSF are optional by design.
    """

    domain = "cutout"

    key: str  # registry key, e.g. "legacy_surveys"
    observatory: str  # e.g. "legacy" — also the service dispatch key
    bands: Tuple[str, ...]  # default bands fetched (TargetSpec.survey_bands overrides)
    native_scale: float  # arcsec / pix of the service's cutouts
    # Whether the service ships a per-pixel variance product the branch
    # can turn into an RMS map (Legacy: an invvar HDU on the same request).
    noise_available: bool = False


# The shared registry stores all families; consumers dispatch on `.domain`
# before touching family-specific fields.
AnyAdapter = Union[
    InstrumentAdapter, VisibilityInstrumentAdapter, SurveyCutoutAdapter
]

_REGISTRY: Dict[str, AnyAdapter] = {}


def register(adapter: AnyAdapter) -> AnyAdapter:
    if adapter.key in _REGISTRY:
        raise ValueError(f"instrument adapter already registered: {adapter.key}")
    _REGISTRY[adapter.key] = adapter
    return adapter


def get(key: str) -> AnyAdapter:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown instrument {key!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def registered_keys() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
