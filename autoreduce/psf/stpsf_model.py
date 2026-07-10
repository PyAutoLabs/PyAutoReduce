"""
STPSF tier-2b model PSFs for JWST frames (issue #29).

The fallback for frames whose own star field cannot support a tier-1 ePSF:
an STPSF (formerly WebbPSF) model evaluated at the frame's detector and the
target's detector position, taken from the ``DET_DIST`` extension —
detector-sampled *including geometric distortion*, i.e. the PSF in the same
native distorted pixels the frame products live in.

The standing literature caveat rides every tier-2b kernel (jwst.md PSF
tiering): empirical PSFs consistently beat model PSFs for decomposition
work — the fallback is flagged in the diagnostics, never silent.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from ..instruments import InstrumentAdapter
from ..target import TargetSpec
from .epsf import normalise_kernel


def model_frame_psf(
    primary,
    target_xy: Tuple[float, float],
    spec: TargetSpec,
    adapter: InstrumentAdapter,
    det_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """STPSF model PSF for one frame at the target's detector position.

    The target may project (slightly) off-chip on edge frames — exactly the
    star-starved frames that need this fallback — so the evaluation
    position is clamped to the detector and the clamp is recorded.
    """
    # poppy auto-detects cupy and binds its array module across poppy at
    # import time; on WSL2-style toolchains the cupy CUDA JIT fails
    # outright, and CPU FFTs are more than fast enough at these fov sizes.
    # The only reliable pin is making poppy's cupy probe fail BEFORE its
    # first import — post-import conf flags don't rebind the module-level
    # xp references. Nothing in this stack uses cupy legitimately.
    import sys

    if "poppy" not in sys.modules and "cupy" not in sys.modules:
        sys.modules["cupy"] = None
    try:
        import stpsf
    except ImportError as err:
        raise ImportError(
            "tier-2b model PSFs need stpsf — pip install autoreduce[psf] "
            "(or pip install stpsf, plus its reference data)"
        ) from err
    import poppy

    poppy.conf.use_cupy = False

    detector = str(primary.get("DETECTOR", "")).strip().upper() or None
    inst = stpsf.NIRCam()
    inst.filter = spec.filter_name
    if detector:
        inst.detector = detector
    ny, nx = det_shape if det_shape is not None else (2048, 2048)
    pos_x = float(np.clip(float(target_xy[0]), 0.0, nx - 1))
    pos_y = float(np.clip(float(target_xy[1]), 0.0, ny - 1))
    clamped = (pos_x, pos_y) != (float(target_xy[0]), float(target_xy[1]))
    inst.detector_position = (pos_x, pos_y)

    fov = max(spec.psf_full_shape)
    if fov % 2 == 0:
        fov += 1
    result = inst.calc_psf(fov_pixels=fov, oversample=2)
    kernel = result["DET_DIST"].data.astype(float)

    psf_full = normalise_kernel(kernel, spec.psf_full_shape)
    psf = normalise_kernel(kernel, spec.psf_shape)
    diagnostics = {
        "method": "stpsf-tier2b",
        "stpsf_version": getattr(stpsf, "__version__", "unknown"),
        "detector": detector,
        "detector_position": [pos_x, pos_y],
        "position_clamped": bool(clamped),
        "ext": "DET_DIST (detector-sampled, geometric distortion included)",
        "caveat": (
            "model-PSF fallback — the JWST literature consistently prefers "
            "empirical PSFs for decomposition (jwst.md PSF tiering); "
            "flagged, never silent"
        ),
    }
    return psf, psf_full, diagnostics
