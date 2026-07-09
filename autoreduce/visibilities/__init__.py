"""
The visibility branch (docs/design/alma.md): split -> extract -> assemble.

`split` and `extract` speak CASA (casatasks / casatools, imported inside
functions); `assemble` is pure numpy — the seam the unit tests cover.
"""

from .assemble import (
    assemble_ms_products,
    concatenate,
    stokes_i_combine,
    uv_wavelengths_from_uvw,
)
from .extract import MsColumns
