"""
Shared combine-backend plumbing: the scratch-directory discipline and the
provenance fragment both backends emit identically.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..instruments import InstrumentAdapter
from ..noise.rms import casertano_r
from ..target import TargetSpec
from .diagnostics import check_weight_uniformity


@contextmanager
def chdir_scratch(output_dir: Path):
    """Resolve, create and chdir into a backend scratch dir; always restore."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(output_dir)
    try:
        yield output_dir
    finally:
        os.chdir(cwd)


def combine_provenance(
    spec: TargetSpec,
    adapter: InstrumentAdapter,
    exposures: List[Path],
    wht: np.ndarray,
    kwargs_key: str,
    kwargs: Dict,
    head: Dict = None,
    tail: Dict = None,
) -> Dict:
    """
    The provenance dict every combine backend records, assembled in the
    canonical key order (kept stable — reduction.json is byte-compared by the
    refactor witnesses). `head`/`tail` carry backend-specific extras.
    """
    out = dict(head or {})
    out.update(
        {
            "n_exposures": len(exposures),
            "exposures": [Path(p).name for p in exposures],
            "single_exposure_branch": len(exposures) == 1,
            kwargs_key: kwargs,
            "correlated_noise_factor": casertano_r(
                spec.final_pixfrac, adapter.scale_ratio(spec.final_scale)
            ),
            "weight_uniformity": check_weight_uniformity(wht),
        }
    )
    out.update(tail or {})
    return out
