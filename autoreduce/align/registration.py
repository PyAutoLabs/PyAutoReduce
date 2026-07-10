"""
Frame registration by phase cross-correlation (design doc keck_ao.md,
stage 4). numpy-only.

NIRC2 header pointing is approximate (arcsecond-level); relative frame
offsets come from the data. Phase correlation of each frame against the
reference frame gives the integer shift; a parabolic fit to the correlation
peak refines it to sub-pixel. Bad pixels (NaN) are zero-filled for the
transform only — a fraction of dead pixels does not move the peak.
"""

from typing import List, Tuple

import numpy as np


def phase_offset(
    reference: np.ndarray, frame: np.ndarray, whiten: bool = True
) -> Tuple[float, float]:
    """(dy, dx) such that shifting `frame` by it aligns it to `reference`.

    ``whiten=True`` (the default) is phase correlation proper — the sharp,
    contrast-independent peak that noisy star fields need (the Keck path).
    For smooth, nearly noise-free structure (e.g. frame-products cutouts of
    one galaxy) whitening amplifies the empty high frequencies into ringing
    whose sidelobes can beat the true peak — pass ``whiten=False`` there to
    correlate the structure directly.
    """
    if reference.shape != frame.shape:
        raise ValueError(
            f"shape mismatch: {reference.shape} vs {frame.shape}"
        )
    a = np.nan_to_num(reference, nan=0.0)
    b = np.nan_to_num(frame, nan=0.0)
    a = a - a.mean()
    b = b - b.mean()
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cross = fa * np.conj(fb)
    if whiten:
        norm = np.abs(cross)
        norm[norm == 0.0] = 1.0
        cross = cross / norm
    corr = np.fft.irfft2(cross, s=reference.shape)

    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = float(peak[0]), float(peak[1])

    # Parabolic sub-pixel refinement around the peak, per axis.
    def refine(values: np.ndarray, idx: int) -> float:
        prev_v, this_v, next_v = (
            values[(idx - 1) % len(values)],
            values[idx],
            values[(idx + 1) % len(values)],
        )
        denom = prev_v - 2.0 * this_v + next_v
        if denom == 0.0:
            return 0.0
        return float(np.clip(0.5 * (prev_v - next_v) / denom, -0.5, 0.5))

    dy += refine(corr[:, peak[1]], peak[0])
    dx += refine(corr[peak[0], :], peak[1])

    ny, nx = reference.shape
    if dy > ny / 2:
        dy -= ny
    if dx > nx / 2:
        dx -= nx
    # The correlation peak lands at (ref - frame); the convention here (what
    # the combine pixmap subtracts) is frame - ref.
    return -dy, -dx


def offsets_to_reference(frames: List[np.ndarray]) -> List[Tuple[float, float]]:
    """Per-frame (dy, dx) offsets of every frame relative to the first."""
    reference = frames[0]
    return [(0.0, 0.0)] + [phase_offset(reference, f) for f in frames[1:]]
