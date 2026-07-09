"""
Measurement-set table extraction (design doc alma.md, extract stage).

A direct port of the reference recipe's ``getcol_wrapper`` family onto the
modular ``casatools.table`` tool (no monolithic CASA shell), plus the
``WEIGHT`` column the recipe stopped short of — `al.Interferometer` needs a
noise map, and the MS weight convention (weight = 1/sigma^2) is where it
comes from.

casatools is imported inside functions so the package imports without it.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MsColumns:
    """
    The raw per-measurement-set arrays the assemble stage consumes, in MS
    axis order:

    - data: complex, (n_pol, n_chan, n_rows)
    - uvw: metres, (3, n_rows)
    - weight: 1/sigma^2 per complex visibility, (n_pol, n_rows)
    - chan_freq: Hz, (n_chan,)
    - antenna1 / antenna2 / time / scan: (n_rows,)
    """

    data: np.ndarray
    uvw: np.ndarray
    weight: np.ndarray
    chan_freq: np.ndarray
    antenna1: np.ndarray
    antenna2: np.ndarray
    time: np.ndarray
    scan: np.ndarray

    def __post_init__(self):
        n_pol, n_chan, n_rows = self.data.shape
        expect = {
            "uvw": (3, n_rows),
            "weight": (n_pol, n_rows),
            "chan_freq": (n_chan,),
            "antenna1": (n_rows,),
            "antenna2": (n_rows,),
            "time": (n_rows,),
            "scan": (n_rows,),
        }
        for name, shape in expect.items():
            got = np.shape(getattr(self, name))
            if got != shape:
                raise ValueError(
                    f"MS column {name} has shape {got}, expected {shape} "
                    f"for data shape {self.data.shape}"
                )


def getcol(ms: Path, table: str, colname: str) -> np.ndarray:
    """One column from one MS table ('' = the main table), as delivered."""
    from casatools import table as table_tool

    ms = Path(ms)
    if not ms.is_dir():
        raise IOError(f"measurement set does not exist: {ms}")
    tb = table_tool()
    tb.open(str(ms / table) if table else str(ms))
    try:
        col = tb.getcol(colname)
    finally:
        tb.close()
    return np.asarray(col)


def columns_from(ms: Path) -> MsColumns:
    """
    Everything the assemble stage needs from one (already split) MS.

    Shapes are normalised to the `MsColumns` contract rather than squeezed:
    a single-channel (fully collapsed) spw keeps its channel axis, so the
    downstream code has one code path for continuum and line widths.
    """
    data = np.asarray(getcol(ms, "", "DATA"))
    if data.ndim == 2:  # some tools drop the length-1 channel axis
        data = data[:, None, :]
    if data.ndim != 3:
        raise ValueError(
            f"{Path(ms).name}: DATA has shape {data.shape}, expected "
            f"(n_pol, n_chan, n_rows)"
        )
    chan_freq = np.atleast_1d(
        np.squeeze(getcol(ms, "SPECTRAL_WINDOW", "CHAN_FREQ"))
    ).astype(float)
    if chan_freq.ndim != 1:
        raise ValueError(
            f"{Path(ms).name}: expected one spectral window after split, "
            f"got CHAN_FREQ shape {chan_freq.shape} — split per spw first"
        )
    return MsColumns(
        data=data,
        uvw=np.asarray(getcol(ms, "", "UVW"), dtype=float),
        weight=np.asarray(getcol(ms, "", "WEIGHT"), dtype=float),
        chan_freq=chan_freq,
        antenna1=np.asarray(getcol(ms, "", "ANTENNA1")),
        antenna2=np.asarray(getcol(ms, "", "ANTENNA2")),
        time=np.asarray(getcol(ms, "", "TIME"), dtype=float),
        scan=np.asarray(getcol(ms, "", "SCAN_NUMBER")),
    )


def num_channels_per_spw(ms: Path) -> np.ndarray:
    """NUM_CHAN per spectral window (drives width=0 full-collapse)."""
    return np.atleast_1d(
        np.squeeze(getcol(ms, "SPECTRAL_WINDOW", "NUM_CHAN"))
    ).astype(int)
