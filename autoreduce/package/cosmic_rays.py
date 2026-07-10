"""
Per-frame cosmic-ray masks via deepCR (Zhang & Bloom 2020), the deep-learned
CR detector for single HST exposures.

driz_cr rejects cosmic rays against a median stack, so its DQ flags exist
only when several exposures overlap; per-frame modeling needs a mask for
every frame on its own, single-exposure visits included. deepCR provides
exactly that — a deviation from the STScI-default stack rejection, justified
by the per-frame requirement and documented in the design doc
(hst_acs_pipeline.md, per-exposure frame products).

Mask-only by contract: deepCR's inpainting half is never used — bad pixels
are masked-by-noise downstream, never fabricated.
"""

from typing import Callable, Dict

import numpy as np

# deepCR ships instrument-specific learned models (names are the deepCR>=0.3
# registry keys): ACS/WFC is the published baseline (Zhang & Bloom 2020);
# WFC3/UVIS is the label-free retrain (Chen et al. 2024). The manifest
# records the exact model, so datasets remain re-maskable. WFC3/IR is absent
# deliberately: calwf3 up-the-ramp fitting already flags IR cosmic rays in DQ.
DEEPCR_MODELS: Dict[str, str] = {
    "acs_wfc": "ACS-WFC",
    "wfc3_uvis": "WFC3-UVIS",
}

# Published default operating point (Zhang & Bloom 2020).
DEEPCR_THRESHOLD = 0.5


def masker_for(adapter_key: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build the deepCR model for one instrument, once; returns ``mask(sci)``.

    The returned callable takes a native-unit (ELECTRONS, sky-in — the frame
    state deepCR was trained on) SCI array and returns a boolean CR mask.
    The model load is the expensive step, so callers reuse one masker across
    every chip of a run.
    """
    if adapter_key not in DEEPCR_MODELS:
        raise KeyError(
            f"no deepCR model registered for instrument {adapter_key!r}; "
            f"known: {sorted(DEEPCR_MODELS)}"
        )
    try:
        from deepCR import deepCR
    except ImportError as err:
        raise ImportError(
            "frame_products needs deepCR for per-frame cosmic-ray masks — "
            "pip install autoreduce[frames] (or pip install deepCR)"
        ) from err
    model = deepCR(mask=DEEPCR_MODELS[adapter_key], device="CPU")

    def mask(sci: np.ndarray) -> np.ndarray:
        # Off-chip regions of a partial cutout arrive as NaN; zero them for
        # the network (they are masked-by-noise downstream regardless).
        clean_input = np.nan_to_num(
            np.asarray(sci, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        out = model.clean(clean_input, threshold=DEEPCR_THRESHOLD, inpaint=False)
        return np.asarray(out, dtype=bool)

    return mask


def cr_method_record(adapter_key: str) -> Dict:
    """The manifest/provenance description of how CR masks were produced."""
    if adapter_key in DEEPCR_MODELS:
        return {
            "method": "deepCR",
            "model": DEEPCR_MODELS[adapter_key],
            "threshold": DEEPCR_THRESHOLD,
        }
    if adapter_key == "wfc3_ir":
        # IR CRs are flagged during calwf3 up-the-ramp fitting and arrive in
        # DQ; there is nothing for deepCR to add.
        return {"method": "ramp-fitting (calwf3)", "model": None, "threshold": None}
    if adapter_key in ("nircam_sw", "nircam_lw"):
        # JWST ramps *remove* cosmic rays during slope fitting (jump
        # detection); stack outliers arrive as DO_NOT_USE flags via image3's
        # outlier_detection in the _crf products. Nothing for deepCR to add
        # (and no JWST model exists).
        return {
            "method": "ramp-jump (calwebb stage 1) + image3 outlier_detection (crf)",
            "model": None,
            "threshold": None,
        }
    raise KeyError(f"no per-frame CR method for instrument {adapter_key!r}")
