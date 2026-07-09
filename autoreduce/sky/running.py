"""
Running sky from temporally adjacent, object-masked frames.

The Auger-method recipe the SHARP reductions descend from: for each dithered
science frame, the sky is the median over the nearest-in-time window of
*other* frames, with objects masked so they never imprint on the sky model,
and the mask refined once from the first-pass sky-subtracted frames. The
frame's own data never contributes to its own sky. K'-band sky varies on
minutes timescales, which is exactly why the window is temporal, not global.

Sky levels (e- per frame) are returned per frame — the noise stage's
background variance term is built from them. B-spline residual-background
modelling is a documented open item, not silently absent (the 10-40"
NIRC2 fields are flat at the level the blank-sky closure tests).
"""

import warnings
from typing import Dict, List, Tuple

import numpy as np


def _object_mask(frame: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """True where a source dominates; NaNs (bad pixels) are masked too."""
    finite = np.isfinite(frame)
    if not finite.any():
        raise ValueError("frame has no finite pixels — calibration produced garbage")
    centre = np.median(frame[finite])
    spread = 1.4826 * np.median(np.abs(frame[finite] - centre))
    if spread <= 0.0:
        return ~finite
    return (~finite) | (frame > centre + n_sigma * spread)


def _window_indices(i: int, n: int, window: int) -> List[int]:
    """Indices of the `window` frames nearest in sequence to i, excluding i."""
    if n < 2:
        raise ValueError(
            "running sky needs >= 2 frames; a single-frame reduction has no "
            "sky model — observe a sky or use a wider dither set"
        )
    order = sorted(range(n), key=lambda j: (abs(j - i), j))
    return [j for j in order if j != i][: max(1, min(window, n - 1))]


def running_sky_subtract(
    frames: List[np.ndarray],
    window: int,
    n_sigma: float = 3.0,
) -> Tuple[List[np.ndarray], Dict]:
    """
    Subtract a per-frame running sky; return (subtracted frames, provenance).

    Two passes: object masks from the raw frames seed the first sky; the
    masks are then rebuilt from the first-pass subtracted frames (fainter
    wings emerge once the sky pedestal is gone) and the sky re-estimated.
    Masked pixels fall back to the unmasked median sky of the window frame —
    a sky estimate must exist everywhere or the frame is unusable.
    """
    n = len(frames)
    masks = [_object_mask(f, n_sigma) for f in frames]

    def one_pass(current_masks):
        # Scaled sky, the standard NIR recipe: each neighbour is normalised
        # to unit median before combining, so the model carries the sky
        # *structure*; the *level* is the frame's own masked median. This
        # removes the bias a drifting sky level puts on edge-of-sequence
        # frames, whose temporal windows are one-sided.
        masked_medians = []
        for frame, mask in zip(frames, current_masks):
            sky_pixels = frame[~mask & np.isfinite(frame)]
            if sky_pixels.size == 0:
                raise ValueError(
                    "a frame has no unmasked sky pixels — the object mask "
                    "covers the full field; in-field sky estimation is not "
                    "possible for this dither pattern"
                )
            masked_medians.append(float(np.median(sky_pixels)))

        subtracted = []
        for i, frame in enumerate(frames):
            neighbour_stack = []
            for j in _window_indices(i, n, window):
                if masked_medians[j] <= 0.0:
                    raise ValueError(
                        f"frame {j} has non-positive sky level "
                        f"({masked_medians[j]}); raw NIR frames must carry "
                        f"a positive sky pedestal — calibration is broken"
                    )
                neighbour = frames[j] / masked_medians[j]
                neighbour = np.where(current_masks[j], np.nan, neighbour)
                neighbour_stack.append(neighbour)
            stack = np.stack(neighbour_stack)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                # All-NaN columns are expected (pixels masked in every window
                # frame) and handled as holes right below.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                structure = np.nanmedian(stack, axis=0)
            # Pixels masked in every window frame still need a sky value.
            holes = ~np.isfinite(structure)
            if holes.all():
                raise ValueError(
                    f"sky model for frame {i} is empty — the object masks "
                    f"cover the full field; the dither pattern cannot "
                    f"support in-field sky estimation"
                )
            if holes.any():
                structure[holes] = 1.0
            subtracted.append(frame - structure * masked_medians[i])
        return subtracted, masked_medians

    first_pass, _ = one_pass(masks)
    masks = [_object_mask(f, n_sigma) for f in first_pass]
    subtracted, levels = one_pass(masks)

    provenance = {
        "recipe": "scaled running sky: unit-median structure from "
        "object-masked adjacent frames x own masked median, 2 passes",
        "window": int(window),
        "mask_n_sigma": float(n_sigma),
        "sky_levels_e": levels,
        "masked_fraction_final": [float(m.mean()) for m in masks],
    }
    return subtracted, provenance
