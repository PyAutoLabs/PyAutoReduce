"""
Keck Observatory Archive acquisition (design doc keck_ao.md, stage 1).

KOA serves raw level-0 frames only (level-1 quick-look products are not
science grade), so acquisition for a ground-based reduction means three
retrievals, all through PyKOA's TAP service:

- **science frames** of the target (by KOA identifier when the spec pins
  them, else by cone + program + filter + camera),
- **the night's calibrations** — darks matched to the science ITIME/COADDS
  and flats in the science filter — because ground-based calibration is the
  pipeline's own job, not an archive product,
- **PSF-star frames** named by the spec (tier-A PSF strategy).

The geometric-distortion solution is acquisition too (the CRDS-analogue
seam): the epoch-matched lookup tables are synced into the shared references
cache and recorded, with checksums, in provenance.

PyKOA is imported inside functions so the package imports without it.
"""

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..instruments import InstrumentAdapter
from ..instruments.nirc2 import DISTORTION_SOLUTIONS, distortion_solution_for_mjd

# KOA instrument code for NIRC2 in TAP table names (koa_nirc2).
_KOA_TABLE = "koa_nirc2"


def _query_tap(adql: str, out_path: Path) -> "object":
    """Run one PyKOA TAP query to a VOTable on disk; return the astropy table."""
    from astropy.table import Table
    from pykoa.koa import Koa

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Koa.query_adql(adql, str(out_path), overwrite=True, format="votable")
    if not out_path.exists():
        # PyKOA reports some failures on stdout instead of raising.
        raise RuntimeError(f"KOA TAP query produced no result file: {adql}")
    return Table.read(out_path, format="votable")


def _camera_clause(adapter: InstrumentAdapter) -> str:
    camera = {"nirc2_narrow": "narrow", "nirc2_wide": "wide"}[adapter.key]
    return f"lower(camname) = '{camera}'"


def query_science_frames(
    ra: float,
    dec: float,
    adapter: InstrumentAdapter,
    filter_name: str,
    work_dir: Path,
    proposal_ids: Optional[Sequence[str]] = None,
    koa_ids: Optional[Sequence[str]] = None,
):
    """
    Metadata table of the science frames to reduce.

    Pinned KOA ids take precedence (exact, reproducible frame set); otherwise
    a 30" cone around the target restricted to object frames in the requested
    camera + filter (and program, when given).
    """
    if koa_ids:
        ids = ", ".join(f"'{k}'" for k in koa_ids)
        adql = f"select * from {_KOA_TABLE} where koaid in ({ids})"
    else:
        clauses = [
            f"contains(point('icrs', ra, dec), circle('icrs', {ra}, {dec}, 0.00833)) = 1",
            _camera_clause(adapter),
            f"lower(filter) like '%{filter_name.lower()}%'",
            "lower(koaimtyp) = 'object'",
        ]
        if proposal_ids:
            progs = ", ".join(f"'{p.lower()}'" for p in proposal_ids)
            clauses.append(f"lower(progid) in ({progs})")
        adql = f"select * from {_KOA_TABLE} where " + " and ".join(clauses)
    table = _query_tap(adql, work_dir / "koa_science_query.xml")
    if len(table) == 0:
        raise FileNotFoundError(
            f"KOA returned no NIRC2 science frames for the query: {adql}"
        )
    return table


# NIR flat fields are stable over weeks; when the science night itself has
# none (routine — SHARP's own nights carry only darks), the nearest
# flat-bearing night inside this window is used and recorded.
FLAT_SEARCH_DAYS = 14


def _query_flats_on(date_clause: str, adapter, filter_name: str, work_dir, tag: str):
    adql = (
        f"select * from {_KOA_TABLE} where ({date_clause}) and "
        f"{_camera_clause(adapter)} and "
        f"lower(koaimtyp) in ('flatlamp', 'flatlampoff', 'domeflat') "
        f"and lower(filter) like '%{filter_name.lower()}%'"
    )
    return _query_tap(adql, work_dir / f"koa_flat_query_{tag}.xml")


def query_night_calibrations(
    dates: Sequence[str],
    setups: Sequence[Tuple[float, int]],
    adapter: InstrumentAdapter,
    filter_name: str,
    work_dir: Path,
):
    """
    Darks matched to the science (ITIME, COADDS) ``setups`` from the science
    ``dates`` (ISO strings), plus flats in the science filter — from the
    science nights when they exist, else from the nearest flat-bearing night
    within ``FLAT_SEARCH_DAYS`` (preferring nights with lamp-off pairs).
    Inputs are plain values (not a query table) so cached re-runs can rebuild
    them from frame headers.
    """
    from datetime import date as date_cls
    from datetime import timedelta

    from astropy.table import vstack

    dates = sorted(dates)
    date_clause = " or ".join(f"date_obs = date '{d}'" for d in dates)
    pairs = sorted(set(setups))
    itime_clause = " or ".join(
        f"(abs(itime - {it}) < 0.005 and coadds = {co})" for it, co in pairs
    )
    dark_adql = (
        f"select * from {_KOA_TABLE} where ({date_clause}) and "
        f"{_camera_clause(adapter)} and lower(koaimtyp) = 'dark' and ({itime_clause})"
    )
    darks = _query_tap(dark_adql, work_dir / "koa_dark_query.xml")
    # Darks are optional by design: the running sky subtraction removes
    # dark + sky together (the SHARP recipe is flat + sky only).

    flats = _query_flats_on(date_clause, adapter, filter_name, work_dir, "night")
    if len(flats) == 0:
        d0 = date_cls.fromisoformat(dates[0])
        d1 = date_cls.fromisoformat(dates[-1])
        lo = (d0 - timedelta(days=FLAT_SEARCH_DAYS)).isoformat()
        hi = (d1 + timedelta(days=FLAT_SEARCH_DAYS)).isoformat()
        window = (
            f"date_obs >= date '{lo}' and date_obs <= date '{hi}'"
        )
        nearby = _query_flats_on(window, adapter, filter_name, work_dir, "window")
        if len(nearby) == 0:
            raise FileNotFoundError(
                f"KOA has no NIRC2 flats for filter {filter_name} within "
                f"+/-{FLAT_SEARCH_DAYS} days of {dates}; a ground-based "
                f"reduction cannot proceed without a flat"
            )
        # Nearest flat night, preferring one with lamp-off pairs.
        def _score(day: str) -> tuple:
            rows = nearby[[str(d)[:10] == day for d in nearby["date_obs"]]]
            has_off = any(str(k).lower() == "flatlampoff" for k in rows["koaimtyp"])
            distance = min(
                abs((date_cls.fromisoformat(day) - d).days) for d in (d0, d1)
            )
            return (not has_off, distance)

        flat_days = sorted(
            {str(d)[:10] for d in nearby["date_obs"]}, key=_score
        )
        chosen = flat_days[0]
        flats = nearby[[str(d)[:10] == chosen for d in nearby["date_obs"]]]

    if len(darks) and len(flats):
        return vstack([darks, flats], metadata_conflicts="silent")
    return flats if len(flats) else darks


def download_frames(table, dest_dir: Path, tag: str) -> List[Path]:
    """Download every frame in a metadata table into dest_dir/<tag>/."""
    from pykoa.koa import Koa

    out_dir = Path(dest_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    # PyKOA downloads from a metadata table written to disk.
    meta_path = out_dir / f"koa_{tag}_download.xml"
    table.write(meta_path, format="votable", overwrite=True)
    Koa.download(str(meta_path), "votable", str(out_dir))
    fits_files = sorted(
        p for p in out_dir.rglob("*.fits*") if "download" not in p.name
    )
    if len(fits_files) < len(table):
        raise FileNotFoundError(
            f"KOA download incomplete for {tag}: {len(fits_files)} files "
            f"for {len(table)} frames in {out_dir}"
        )
    return fits_files


def sync_distortion_solution(
    references_dir: Path, adapter: InstrumentAdapter, mjd: float
) -> Dict:
    """
    Ensure the epoch-matched distortion lookup tables are in the references
    cache; return the provenance fragment (epoch, paths, checksums).

    Narrow camera only: the published solutions do not cover the wide camera
    (design doc, open items) — the combine backend enforces this loudly.
    """
    epoch = distortion_solution_for_mjd(mjd)
    urls = DISTORTION_SOLUTIONS[epoch]
    ref_dir = Path(references_dir) / adapter.crds_reference_subpath
    ref_dir.mkdir(parents=True, exist_ok=True)
    paths, checksums = [], []
    for url in urls:
        dest = ref_dir / Path(url).name
        if not dest.exists():
            with urllib.request.urlopen(url, timeout=120) as resp, open(
                dest, "wb"
            ) as f:
                shutil.copyfileobj(resp, f)
        paths.append(dest)
        checksums.append(hashlib.sha256(dest.read_bytes()).hexdigest()[:16])
    return {
        "distortion_epoch": epoch,
        "distortion_files": [p.name for p in paths],
        "distortion_sha256_16": checksums,
        "distortion_paths": [str(p) for p in paths],
    }


def frame_facts_from_headers(paths: Sequence[Path]) -> List[Dict]:
    """
    The per-frame facts later stages need, read from the FITS headers (the
    ground truth, available on cached re-runs when no query table exists):
    path, mjd, itime (per coadd, s), coadds. Sorted by MJD — temporal order
    is what the running sky is defined over. Loud on missing keywords: a
    frame without ITIME/COADDS/MJD-OBS cannot be calibrated.
    """
    from astropy.io import fits

    facts = []
    for path in paths:
        header = fits.getheader(path)
        try:
            facts.append(
                {
                    "path": Path(path),
                    "mjd": float(header["MJD-OBS"]),
                    "itime": float(header["ITIME"]),
                    "coadds": int(header["COADDS"]),
                    "date_obs": str(header["DATE-OBS"])[:10],
                    # Sampling mode drives the effective read noise (MCDS
                    # averaging); default to plain CDS when absent.
                    "sampmode": int(header.get("SAMPMODE", 2)),
                    "multisam": int(header.get("MULTISAM", 1)),
                }
            )
        except KeyError as err:
            raise KeyError(
                f"{Path(path).name}: NIRC2 header lacks {err} — not a raw "
                f"KOA level-0 frame?"
            ) from None
    return sorted(facts, key=lambda d: d["mjd"])
