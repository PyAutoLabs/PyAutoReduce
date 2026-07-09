"""
Running sky subtraction (design doc keck_ao.md, stage 3) — the defining
ground-based NIR stage. numpy/astropy only.
"""

from .running import group_by_time_gaps, running_sky_subtract
