"""
Ground-based frame calibration (design doc keck_ao.md, stage 2) — the stage
space-based level-2 products make moot. numpy/astropy only.
"""

from .nir_frames import (
    CalibrationFrames,
    build_calibrations,
    calibrate_frame,
    load_calibration_sets,
)
