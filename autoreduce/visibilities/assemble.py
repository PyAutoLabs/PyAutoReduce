"""
Visibility assembly (design doc alma.md, assemble stage) — pure numpy.

Turns per-(uid, spw) `MsColumns` into the `al.Interferometer` array
contract: visibilities, uv_wavelengths and noise_map, each `(Nvis, 2)`.
Three operations, in order:

1. **Stokes-I combine** over the parallel hands with the MS weights
   (weight = 1/sigma^2): I = sum(w D)/sum(w), sigma_I = 1/sqrt(sum w),
   where a hand contributes only if its weight is positive and its datum
   finite. Visibilities with no contributing hand are dropped (counted in
   provenance), never zero-filled.
2. **UVW -> wavelengths** per channel frequency (u f / c), giving each
   channel its own uv sample of the same baseline.
3. **Flatten + concatenate** channels within an MS and then all (uid, spw)
   MS into the final arrays.
"""

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .extract import MsColumns

SPEED_OF_LIGHT_M_S = 299792458.0


@dataclass(frozen=True)
class VisibilitySet:
    """One assembled (uid, spw) block, already in autolens axis order."""

    visibilities: np.ndarray  # (n, 2) real/imag
    uv_wavelengths: np.ndarray  # (n, 2) u/v
    noise_map: np.ndarray  # (n, 2) sigma on real/imag (equal by convention)
    provenance: Dict

    def __post_init__(self):
        n = self.visibilities.shape[0]
        for name in ("visibilities", "uv_wavelengths", "noise_map"):
            arr = getattr(self, name)
            if arr.shape != (n, 2):
                raise ValueError(
                    f"{name} has shape {arr.shape}, expected ({n}, 2)"
                )


def uv_wavelengths_from_uvw(
    uvw: np.ndarray, chan_freqs: np.ndarray
) -> np.ndarray:
    """
    (u, v) in wavelengths per channel: shape (n_chan, n_rows, 2). The MS
    stores one metric baseline per row; each channel samples it at its own
    frequency.
    """
    uvw = np.asarray(uvw, dtype=float)
    chan_freqs = np.atleast_1d(np.asarray(chan_freqs, dtype=float))
    if uvw.ndim != 2 or uvw.shape[0] != 3:
        raise ValueError(f"uvw must have shape (3, n_rows): {uvw.shape}")
    scale = chan_freqs[:, None] / SPEED_OF_LIGHT_M_S  # (n_chan, 1)
    u = uvw[0][None, :] * scale
    v = uvw[1][None, :] * scale
    return np.stack((u, v), axis=-1)


def stokes_i_combine(data: np.ndarray, weight: np.ndarray):
    """
    Weighted parallel-hand average per visibility.

    data: complex (n_pol, n_chan, n_rows); weight: (n_pol, n_rows), the MS
    per-row weights (weight = 1/sigma^2 per complex visibility). Returns
    (stokes_i (n_chan, n_rows), sigma (n_chan, n_rows), keep (n_chan, n_rows)
    bool). A hand contributes only where its weight is positive AND its
    datum is finite — a non-finite visibility must not keep its weight in
    the denominator (that would bias the average low, silently). Positions
    with no contributing hand are flagged out via ``keep``, counted by the
    caller, never zero-filled.
    """
    data = np.asarray(data)
    weight = np.asarray(weight, dtype=float)
    if data.ndim != 3:
        raise ValueError(f"data must be (n_pol, n_chan, n_rows): {data.shape}")
    if weight.shape != (data.shape[0], data.shape[2]):
        raise ValueError(
            f"weight shape {weight.shape} does not match data {data.shape}"
        )
    w = np.where(np.isfinite(weight) & (weight > 0.0), weight, 0.0)
    finite = np.isfinite(data)
    if finite.all():
        # Common case: no per-element masking needed, keep the cheap path
        # (no (n_pol, n_chan, n_rows) weight materialisation).
        w_sum = np.broadcast_to(w.sum(axis=0)[None, :], data.shape[1:])
        numerator = np.einsum("pr,pcr->cr", w, data)
    else:
        w_eff = w[:, None, :] * finite
        w_sum = w_eff.sum(axis=0)
        numerator = np.einsum(
            "pcr,pcr->cr", w_eff, np.where(finite, data, 0.0)
        )
    keep = w_sum > 0.0
    if not np.any(keep):
        raise ValueError("no visibility carries positive weight")
    safe_w_sum = np.where(keep, w_sum, 1.0)
    stokes_i = numerator / safe_w_sum
    sigma = 1.0 / np.sqrt(safe_w_sum)
    return stokes_i, sigma, keep


def assemble_ms_products(columns: MsColumns) -> VisibilitySet:
    """One (uid, spw) MS -> flattened autolens-order arrays + provenance."""
    stokes_i, sigma, keep = stokes_i_combine(columns.data, columns.weight)
    uv = uv_wavelengths_from_uvw(columns.uvw, columns.chan_freq)
    n_chan, n_rows = stokes_i.shape

    kept = keep.ravel()
    visibilities = np.stack(
        (stokes_i.real.ravel()[kept], stokes_i.imag.ravel()[kept]), axis=-1
    )
    uv_wavelengths = uv.reshape(n_chan * n_rows, 2)[kept]
    # The MS weight is per complex visibility: the same sigma applies to the
    # real and imaginary parts.
    sigma_flat = sigma.ravel()[kept]
    noise_map = np.stack((sigma_flat, sigma_flat), axis=-1)

    row_keep = keep.any(axis=0)
    return VisibilitySet(
        visibilities=visibilities.astype(float),
        uv_wavelengths=uv_wavelengths.astype(float),
        noise_map=noise_map.astype(float),
        provenance={
            "n_rows": int(n_rows),
            "n_channels": int(n_chan),
            "n_visibilities": int(np.count_nonzero(kept)),
            "n_visibilities_dropped_invalid": int(kept.size - np.count_nonzero(kept)),
            "chan_freq_hz": [float(f) for f in np.atleast_1d(columns.chan_freq)],
            "n_antennas": int(
                np.union1d(
                    columns.antenna1[row_keep], columns.antenna2[row_keep]
                ).size
            ),
            "n_scans": int(np.unique(columns.scan[row_keep]).size),
        },
    )


def concatenate(sets: Sequence[VisibilitySet], labels: Sequence[str]) -> VisibilitySet:
    """All (uid, spw) blocks into the final dataset, provenance per block."""
    if len(sets) == 0:
        raise ValueError("no visibility sets to concatenate")
    if len(labels) != len(sets):
        raise ValueError(f"{len(labels)} labels for {len(sets)} sets")
    return VisibilitySet(
        visibilities=np.concatenate([s.visibilities for s in sets], axis=0),
        uv_wavelengths=np.concatenate([s.uv_wavelengths for s in sets], axis=0),
        noise_map=np.concatenate([s.noise_map for s in sets], axis=0),
        provenance={
            "n_visibilities": int(sum(s.provenance["n_visibilities"] for s in sets)),
            "blocks": {
                label: s.provenance for label, s in zip(labels, sets)
            },
        },
    )
