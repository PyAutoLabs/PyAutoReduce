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
    return _getcols(ms, table, (colname,))[colname]


def _getcols(ms: Path, table: str, colnames) -> dict:
    """Several columns from one MS table under a single table open."""
    from casatools import table as table_tool

    ms = Path(ms)
    if not ms.is_dir():
        raise IOError(f"measurement set does not exist: {ms}")
    tb = table_tool()
    tb.open(str(ms / table) if table else str(ms))
    try:
        return {name: np.asarray(tb.getcol(name)) for name in colnames}
    finally:
        tb.close()


def columns_from(ms: Path) -> MsColumns:
    """
    Everything the assemble stage needs from one (already split) MS.

    Shapes are normalised to the `MsColumns` contract rather than squeezed:
    a single-channel (fully collapsed) spw keeps its channel axis, so the
    downstream code has one code path for continuum and line widths.
    """
    chan_freq = np.atleast_1d(
        np.squeeze(getcol(ms, "SPECTRAL_WINDOW", "CHAN_FREQ"))
    ).astype(float)
    if chan_freq.ndim != 1:
        raise ValueError(
            f"{Path(ms).name}: expected one spectral window after split, "
            f"got CHAN_FREQ shape {chan_freq.shape} — split per spw first"
        )
    main = _getcols(
        ms,
        "",
        (
            "DATA",
            "UVW",
            "WEIGHT",
            "ANTENNA1",
            "ANTENNA2",
            "TIME",
            "SCAN_NUMBER",
        ),
    )
    data = main["DATA"]
    if data.ndim == 2:
        # Some tools drop a length-1 axis. Only the channel axis can be
        # re-inserted unambiguously, and only when this spw really has one
        # channel — anything else must fail loudly, not be guessed at.
        if chan_freq.size == 1:
            data = data[:, None, :]
        else:
            raise ValueError(
                f"{Path(ms).name}: DATA has shape {data.shape} but the spw "
                f"has {chan_freq.size} channels — cannot tell which axis "
                f"was dropped"
            )
    if data.ndim != 3:
        raise ValueError(
            f"{Path(ms).name}: DATA has shape {data.shape}, expected "
            f"(n_pol, n_chan, n_rows)"
        )
    return MsColumns(
        data=data,
        uvw=main["UVW"].astype(float),
        weight=main["WEIGHT"].astype(float),
        chan_freq=chan_freq,
        antenna1=main["ANTENNA1"],
        antenna2=main["ANTENNA2"],
        time=main["TIME"].astype(float),
        scan=main["SCAN_NUMBER"],
    )


def num_channels_per_spw(ms: Path) -> np.ndarray:
    """NUM_CHAN per spectral window (drives width=0 full-collapse)."""
    return np.atleast_1d(
        np.squeeze(getcol(ms, "SPECTRAL_WINDOW", "NUM_CHAN"))
    ).astype(int)
