"""
ALMA archive acquisition (design doc alma.md, stage 1).

The canonical pipeline input is a **calibrated measurement set per
execution-block uid** (`uid___<uid>.ms.split.cal`). The archive does not
serve those as a plain anonymous download; they arrive by one of

- an ARC delivery (EU CalMS service, EA/NA helpdesk, NRAO SRDP restore for
  Cycle 5+ pipeline-calibrated data) — the modeler's modern workflow: point
  `alma_ms_dir` at the delivered directory;
- a local `scriptForPI.py` restore of the product + raw tarballs, which
  *are* directly downloadable — that download is what this module automates
  through `astroquery.alma`.

astroquery is imported inside functions so the package imports without it.
"""

import hashlib
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..instruments.alma import MS_SUFFIX, ms_name


def resolve_calibrated_ms(ms_dir: Path, uids: Sequence[str]) -> List[Path]:
    """
    The calibrated per-uid measurement sets inside ``ms_dir``, loud on any
    missing uid. A measurement set is a directory, never a file.
    """
    ms_dir = Path(ms_dir)
    paths, missing = [], []
    for uid in uids:
        path = ms_dir / ms_name(uid)
        if path.is_dir():
            paths.append(path)
        else:
            missing.append(path.name)
    if missing:
        present = sorted(p.name for p in ms_dir.glob("*.ms*") if p.is_dir())
        raise FileNotFoundError(
            f"calibrated measurement set(s) not found in {ms_dir}: {missing}; "
            f"measurement sets present: {present or 'none'}. Expected the "
            f"ARC-delivered / scriptForPI-restored layout uid___<uid>.ms.split.cal "
            f"(docs/design/alma.md, 'Calibrated-MS acquisition')"
        )
    return paths


def query_project(project_code: str):
    """Archive rows for one project code (one row per member OUS product)."""
    from astroquery.alma import Alma

    alma = Alma()
    result = alma.query(payload={"project_code": project_code}, public=None)
    if len(result) == 0:
        raise FileNotFoundError(
            f"ALMA archive returned no observations for project {project_code!r}"
        )
    return result


def download_product_tarballs(
    project_code: str,
    dest_dir: Path,
    member_ous_uids: Optional[Sequence[str]] = None,
) -> List[Path]:
    """
    Download the product/auxiliary tarballs for a project (optionally
    restricted to specific member OUS uids) into ``dest_dir``. Returns the
    local tarball paths. These are the scriptForPI restore inputs — not yet
    calibrated measurement sets (see module docstring).
    """
    from astroquery.alma import Alma

    alma = Alma()
    result = query_project(project_code)
    ous_uids = sorted(set(str(u) for u in result["member_ous_uid"]))
    if member_ous_uids:
        wanted = set(member_ous_uids)
        ous_uids = [u for u in ous_uids if u in wanted]
        if not ous_uids:
            raise FileNotFoundError(
                f"project {project_code!r} has no member OUS matching "
                f"{sorted(wanted)}; archive lists {ous_uids}"
            )
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    urls = []
    for ous in ous_uids:
        info = alma.get_data_info(ous, expand_tarfiles=False)
        urls += [
            str(u) for u in info["access_url"] if str(u).endswith(".tar")
        ]
    if not urls:
        raise FileNotFoundError(
            f"ALMA archive lists no tar products for {project_code!r} "
            f"(member OUS: {ous_uids})"
        )
    downloaded = alma.download_files(urls, savedir=str(dest_dir))
    return [Path(p) for p in downloaded]


def tarball_checksums(tarballs: Sequence[Path]) -> Dict[str, str]:
    """Streaming sha256 (16-hex prefix, the koa idiom) per tarball name."""
    checksums = {}
    for tarball in tarballs:
        digest = hashlib.sha256()
        with open(tarball, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        checksums[Path(tarball).name] = digest.hexdigest()[:16]
    return checksums


def extract_calibrated_ms_from_tarballs(
    tarballs: Sequence[Path], dest_dir: Path
) -> List[Path]:
    """
    Pull any already-calibrated measurement sets (`*.ms.split.cal`) out of
    downloaded tarballs — some deliveries (ARC calibrated-MS tarballs)
    contain them directly; standard product tarballs do not, and then the
    result is empty and the caller raises with restore guidance.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    # PEP 706 extraction filtering; present on 3.12+ and the backported
    # security releases of older interpreters (casatools environments can
    # lag behind).
    extract_kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    found = set()
    for tarball in tarballs:
        try:
            with tarfile.open(tarball) as tar:
                members = [m for m in tar.getmembers() if MS_SUFFIX in m.name]
                for member in members:
                    tar.extract(member, path=dest_dir, **extract_kwargs)
                    # Record the top-level MS directory, not each table file.
                    parts = Path(member.name).parts
                    for i, part in enumerate(parts):
                        if part.endswith(MS_SUFFIX):
                            found.add(dest_dir.joinpath(*parts[: i + 1]))
                            break
        except tarfile.ReadError as err:
            raise IOError(
                f"tarball {tarball} is unreadable (truncated download from "
                f"an interrupted run?): {err} — delete it and re-run to "
                f"re-download"
            ) from None
    return sorted(found)


def restore_guidance(project_code: str, tarball_dir: Path) -> str:
    """The loud, actionable message when only restore inputs are available."""
    return (
        f"the ALMA product tarballs for {project_code!r} are downloaded in "
        f"{tarball_dir}, but they contain no calibrated measurement sets. "
        f"Obtain calibrated MS via (a) scriptForPI.py restore under the CASA "
        f"version named in the QA2 README (unpack the tarballs, cd */script, "
        f"casa -c 'execfile(\"member.....scriptForPI.py\")'), or (b) an ARC "
        f"calibrated-MS delivery (EU CalMS / NRAO SRDP for Cycle 5+), then "
        f"point TargetSpec.alma_ms_dir at the resulting directory "
        f"(docs/design/alma.md, 'Calibrated-MS acquisition')"
    )
