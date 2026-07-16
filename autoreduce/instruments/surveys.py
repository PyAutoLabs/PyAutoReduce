"""
Survey cutout-service adapters (docs/design/surveys.md) — the cutout
domain's registry entries. Ground-based colour context, not modeling data:

- ``legacy_surveys`` — DESI Legacy Imaging Surveys DR10 (DECam griz;
  covers the DES footprint and far beyond). The one service here that
  ships variance: ``&invvar`` appends an inverse-variance HDU to the
  same cutout request.
- ``sdss`` — SDSS frames via astroquery (ugriz, 0.396"/pix); data-only.
- ``panstarrs`` — Pan-STARRS PS1 stack cutouts via the STScI fitscut
  service (grizy, 0.25"/pix); data-only.

HSC PDR is deliberately absent: its cutout service is credential-gated
(STARs account), recorded as deferred in the design doc.
"""

from .adapter import SurveyCutoutAdapter, register

LEGACY_SURVEYS = register(
    SurveyCutoutAdapter(
        key="legacy_surveys",
        observatory="legacy",
        bands=("g", "r", "z"),
        native_scale=0.262,
        noise_available=True,
    )
)

SDSS = register(
    SurveyCutoutAdapter(
        key="sdss",
        observatory="sdss",
        bands=("g", "r", "i"),
        native_scale=0.396,
    )
)

PANSTARRS = register(
    SurveyCutoutAdapter(
        key="panstarrs",
        observatory="ps1",
        bands=("g", "r", "i"),
        native_scale=0.25,
    )
)
