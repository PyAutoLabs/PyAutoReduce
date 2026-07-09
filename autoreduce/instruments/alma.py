"""
ALMA — adapter #8 (phase 5), the first visibility-domain instrument
(docs/design/alma.md).

There is no detector geometry here: ALMA delivers calibrated measurement
sets (one per execution-block uid), and the reduction is table extraction +
assembly rather than image combination. Everything instrument-shaped —
which archive to query, how the calibrated products are laid out — stays
behind this adapter so the visibility stages remain instrument-agnostic
(a future JVLA adapter reuses them unchanged).
"""

from .adapter import VisibilityInstrumentAdapter, register

# Calibrated measurement-set directory naming, as delivered by the ALMA
# pipeline / ARC restore: uid___<uid>.ms.split.cal (the recipe's convention).
MS_SUFFIX = ".ms.split.cal"
MS_PREFIX = "uid___"

ALMA = register(
    VisibilityInstrumentAdapter(
        key="alma",
        observatory="alma",
        archive="alma",
    )
)


def ms_name(uid: str, *parts) -> str:
    """The measurement-set directory name for a uid (+ optional split tags)."""
    tag = "_".join(str(p) for p in parts)
    stem = f"{MS_PREFIX}{uid}" if not tag else f"{MS_PREFIX}{uid}_{tag}"
    return stem + MS_SUFFIX
