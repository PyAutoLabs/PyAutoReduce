"""
Validation utilities shared by the integration/acceptance scripts: sub-pixel
registration and reference-parity statistics (the SLACS-parity method, reused
verbatim for every instrument since phase 1).
"""

from .parity import registered_ratios, subpixel_offset
