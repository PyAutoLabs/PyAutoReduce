"""
Instrument adapters: everything instrument-specific (detector geometry,
calibration file conventions, units, recommended drizzle parameters) lives
behind an adapter so the pipeline stages stay instrument-agnostic. HST/ACS is
the first adapter; WFC3 and JWST follow (see ``docs/design/roadmap.md``).
"""

from .adapter import InstrumentAdapter, get, register, registered_keys
from .acs_wfc import ACS_WFC
from .wfc3_uvis import WFC3_UVIS
from .wfc3_ir import WFC3_IR
from .nircam import NIRCAM_SW, NIRCAM_LW, nircam_adapter_for_filter
from .nirc2 import NIRC2_NARROW, NIRC2_WIDE, NIRC2_DETECTOR
