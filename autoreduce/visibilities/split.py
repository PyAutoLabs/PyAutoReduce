"""
CASA split orchestration (design doc alma.md, split stage) — the reference
recipe's ``main_func`` flow on modular ``casatasks``:

1. isolate the science field from each uid's calibrated MS,
2. per spectral window, average channels by ``width``.

Both steps are idempotent (an existing output MS is reused), matching the
recipe's own re-run behaviour. ``datacolumn="data"`` throughout: calibrated
`.ms.split.cal` deliveries carry the calibrated visibilities in DATA.
``keepflags=False`` drops flagged rows instead of carrying zero-weight
placeholders. casatasks is imported inside functions.
"""

import shutil
from pathlib import Path

import numpy as np

from ..instruments.alma import ms_name


def field_ms_path(work_dir: Path, uid: str, field: str) -> Path:
    return Path(work_dir) / ms_name(uid, field)


def spw_ms_path(work_dir: Path, uid: str, field: str, spw: str, width: int) -> Path:
    return Path(work_dir) / ms_name(uid, field, "spw", spw, "width", width)


def _split(vis: Path, out: Path, field: str, spw: str, width=None) -> Path:
    """
    One idempotent casatasks.split. The task writes to a ``.partial``
    directory that is renamed into place only on success, so an existing
    output MS always means a *complete* prior split — an interrupted run
    (OOM, SIGKILL) never leaves a half-written MS that a resume would
    silently reuse.
    """
    if out.is_dir():
        return out
    from casatasks import split

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    kwargs = dict(
        vis=str(vis),
        outputvis=str(partial),
        keepmms=True,
        field=field,
        spw=str(spw),
        datacolumn="data",
        keepflags=False,
    )
    if width is not None:
        kwargs["width"] = width
    split(**kwargs)
    if not partial.is_dir():
        raise IOError(f"casatasks.split produced no output MS: {partial}")
    partial.rename(out)
    return out


def split_field(ms: Path, uid: str, field: str, work_dir: Path) -> Path:
    """Isolate one science field from a calibrated per-uid MS."""
    return _split(ms, field_ms_path(work_dir, uid, field), field=field, spw="")


def split_spw(
    field_ms: Path, uid: str, field: str, spw: str, width: int, work_dir: Path
) -> Path:
    """One spw, channel-averaged by ``width`` (already resolved, >= 1)."""
    if width < 1:
        raise ValueError(f"width must be resolved to >= 1 before split: {width}")
    return _split(
        field_ms,
        spw_ms_path(work_dir, uid, field, spw, width),
        field=field,
        spw=spw,
        width=width,
    )


def resolve_width(width: int, spw: str, num_chan_by_spw) -> int:
    """
    The channel-averaging width to use for one spw: an explicit positive
    ``width`` passes through; ``0`` means collapse the whole spw (the
    continuum default), read from the parent MS's NUM_CHAN.
    """
    if width > 0:
        return int(width)
    if width < 0:
        raise ValueError(f"width must be >= 0: {width}")
    if num_chan_by_spw is None:
        raise ValueError("width=0 (collapse the spw) needs NUM_CHAN per spw")
    index = int(spw)
    num_chan = np.asarray(num_chan_by_spw)
    if index >= num_chan.size:
        raise ValueError(
            f"spw {spw} out of range: parent MS has {num_chan.size} spectral "
            f"windows"
        )
    return int(num_chan[index])
