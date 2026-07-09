"""
Running sky from temporally adjacent, object-masked frames.

The Auger-method recipe the SHARP reductions descend from, as **scaled sky**:
for each frame, the sky *structure* is the median over the object-masked,
unit-median-normalised window of temporally adjacent frames, and the sky
*level* is the frame's own masked median — which removes the bias a drifting
K'-band sky (minutes timescales) puts on edge-of-sequence frames. The mask
is refined once from the first-pass residuals. A frame never contributes to
its own sky.

Callers must pass a **temporally contiguous** frame set: use
``group_by_time_gaps`` to split multi-night science or interleaved PSF-star
visits first — window adjacency is positional, so a set spanning a gap would
silently borrow sky from a different night/visit. A single-frame group falls
back to its own sigma-clipped median (the only estimate available), with the
recipe recorded.

Sky levels (e- per frame) are returned per frame — the noise stage's
background variance term is built from them. B-spline residual-background
modelling is a documented open item, not silently absent.
"""

import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..noise.rms import mad_sigma


def group_by_time_gaps(mjds: Sequence[float], gap_s: float) -> List[List[int]]:
    """Indices grouped into temporally contiguous runs split at MJD gaps."""
    if len(mjds) == 0:
        raise ValueError("no frames to group")
    order = sorted(range(len(mjds)), key=lambda i: mjds[i])
    groups, current = [], [order[0]]
    for prev, this in zip(order, order[1:]):
        if (mjds[this] - mjds[prev]) * 86400.0 > gap_s:
            groups.append(current)
            current = []
        current.append(this)
    groups.append(current)
    return groups


def _object_mask(frame: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """True where a source dominates; NaNs (bad pixels) are masked too."""
    finite = np.isfinite(frame)
    if not finite.any():
        raise ValueError("frame has no finite pixels — calibration produced garbage")
    centre = np.median(frame[finite])
    spread = mad_sigma(frame)
    if spread <= 0.0:
        return ~finite
    return (~finite) | (frame > centre + n_sigma * spread)


def _window_indices(i: int, n: int, window: int) -> List[int]:
    """Indices of the `window` frames nearest in sequence to i, excluding i."""
    order = sorted(range(n), key=lambda j: (abs(j - i), j))
    return [j for j in order if j != i][: max(1, min(window, n - 1))]


def running_sky_subtract(
    frames: List[np.ndarray],
    window: int,
    n_sigma: float = 3.0,
) -> Tuple[List[np.ndarray], Dict]:
    """
    Subtract a per-frame sky; return (subtracted frames, provenance).

    Two passes: object masks from the raw frames seed the first sky; the
    masks are then rebuilt from the first-pass subtracted frames (fainter
    wings emerge once the sky pedestal is gone) and the sky re-estimated.
    """
    n = len(frames)
    if n == 1:
        # A lone frame has no neighbours: its own masked median is the only
        # sky estimate available (short PSF-star visits). Recorded, never
        # silent.
        from astropy.stats import sigma_clipped_stats

        frame = frames[0]
        level, _, _ = sigma_clipped_stats(frame[np.isfinite(frame)])
        provenance = {
            "recipe": "single frame: own sigma-clipped median",
            "window": 0,
            "mask_n_sigma": float(n_sigma),
            "sky_levels_e": [float(level)],
            "masked_fraction_final": [float(_object_mask(frame, n_sigma).mean())],
        }
        return [frame - level], provenance

    masks = [_object_mask(f, n_sigma) for f in frames]

    def one_pass(current_masks):
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

        # Normalise each frame to unit median once per pass; every window
        # that references frame j reuses the same array.
        normalised = []
        for j, (frame, mask) in enumerate(zip(frames, current_masks)):
            if masked_medians[j] <= 0.0:
                raise ValueError(
                    f"frame {j} has non-positive sky level "
                    f"({masked_medians[j]}); raw NIR frames must carry "
                    f"a positive sky pedestal — calibration is broken"
                )
            normalised.append(
                np.where(mask, np.nan, frame / masked_medians[j])
            )

        subtracted = []
        for i, frame in enumerate(frames):
            stack = np.stack(
                [normalised[j] for j in _window_indices(i, n, window)]
            )
            with np.errstate(all="ignore"), warnings.catch_warnings():
                # All-NaN columns are expected (pixels masked in every window
                # frame) and handled as holes right below.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                structure = np.nanmedian(stack, axis=0)
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
