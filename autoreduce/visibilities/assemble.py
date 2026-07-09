"""
Visibility assembly (design doc alma.md, assemble stage) — pure numpy.

Turns per-(uid, spw) `MsColumns` into the `al.Interferometer` array
contract: visibilities, uv_wavelengths and noise_map, each `(Nvis, 2)`.
Three operations, in order:

1. **Stokes-I combine** over the parallel hands with the MS weights
   (weight = 1/sigma^2): I = sum(w D)/sum(w), sigma_I = 1/sqrt(sum w).
   Rows with no positive weight in either hand are dropped loudly (counted
   in provenance), never zero-filled.
2. **UVW -> wavelengths** per channel frequency (u f / c), giving each
   channel its own uv sample of the same baseline.
3. **Flatten + concatenate** channels within an MS and then all (uid, spw)
   MS into the final arrays.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

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
    (stokes_i (n_chan, n_valid), sigma (n_valid,), keep (n_rows,) bool).
    Zero/negative/non-finite weights contribute nothing; rows with no
    contributing hand at all are flagged out via ``keep``.
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
    w_sum = w.sum(axis=0)  # (n_rows,)
    keep = w_sum > 0.0
    if not np.any(keep):
        raise ValueError("no visibility row carries positive weight")
    numerator = np.einsum("pr,pcr->cr", w, np.nan_to_num(data))
    stokes_i = numerator[:, keep] / w_sum[keep][None, :]
    sigma = 1.0 / np.sqrt(w_sum[keep])
    return stokes_i, sigma, keep


def assemble_ms_products(columns: MsColumns) -> VisibilitySet:
    """One (uid, spw) MS -> flattened autolens-order arrays + provenance."""
    stokes_i, sigma, keep = stokes_i_combine(columns.data, columns.weight)
    uv = uv_wavelengths_from_uvw(columns.uvw[:, keep], columns.chan_freq)
    n_chan, n_rows = stokes_i.shape

    visibilities = np.stack(
        (stokes_i.real.ravel(), stokes_i.imag.ravel()), axis=-1
    )
    uv_wavelengths = uv.reshape(n_chan * n_rows, 2)
    # The MS weight is per complex visibility: the same sigma applies to the
    # real and imaginary parts. Channels of one row share the row weight.
    sigma_flat = np.broadcast_to(sigma[None, :], (n_chan, n_rows)).ravel()
    noise_map = np.stack((sigma_flat, sigma_flat), axis=-1)

    n_dropped = int(np.size(keep) - np.count_nonzero(keep))
    return VisibilitySet(
        visibilities=visibilities.astype(float),
        uv_wavelengths=uv_wavelengths.astype(float),
        noise_map=noise_map.astype(float),
        provenance={
            "n_rows": int(np.size(keep)),
            "n_rows_dropped_zero_weight": n_dropped,
            "n_channels": int(n_chan),
            "n_visibilities": int(n_chan * n_rows),
            "chan_freq_hz": [float(f) for f in np.atleast_1d(columns.chan_freq)],
            "n_antennas": int(
                np.union1d(columns.antenna1[keep], columns.antenna2[keep]).size
            ),
            "n_scans": int(np.unique(columns.scan[keep]).size),
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
