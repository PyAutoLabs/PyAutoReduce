"""
CRDS reference-file sync (design doc stage 1, spike finding).

AstroDrizzle's IVM weighting resolves calibration files through the
adapter's reference environment variable (``jref$`` for ACS), so best
references must exist locally before the drizzle stage. References are
shared across targets and are never evicted.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List

from ..instruments import InstrumentAdapter


def configure_environment(references_root: Path, adapter: InstrumentAdapter) -> dict:
    """
    Set the CRDS variables for this process. Must run before drizzlepac /
    the jwst pipeline are imported anywhere in the process. Returns the
    mapping applied.

    Deliberately overrides any inherited CRDS_PATH/jref: the pipeline is a
    pure function of the target spec plus the archive, so its reference files
    live in *its* cache, not wherever the shell environment happens to point.
    The server URL is the adapter's (hst-crds vs jwst-crds).
    """
    env = {
        "CRDS_SERVER_URL": adapter.crds_server_url,
        "CRDS_PATH": str(references_root),
    }
    # HST-style tools resolve references through an iraf-style variable
    # (jref$/iref$); the jwst pipeline reads CRDS_PATH directly.
    if adapter.reference_env_key != "CRDS_PATH":
        env[adapter.reference_env_key] = (
            str(Path(references_root) / adapter.crds_reference_subpath) + "/"
        )
    os.environ.update(env)
    return env


def references_present(references_root: Path, adapter: InstrumentAdapter) -> bool:
    """True if the instrument's reference directory exists and is non-empty."""
    ref_dir = Path(references_root) / adapter.crds_reference_subpath
    return ref_dir.is_dir() and any(ref_dir.iterdir())


def should_sync(
    observatory: str, sync_references: bool, present: bool
) -> bool:
    """
    Decide the bestrefs step for this run; pure function, unit-testable.

    CRDS revises reference files on its own schedule, independent of the
    exposure cache — so a fully-cached rerun still re-syncs by default
    (bestrefs is a cheap metadata check when nothing is stale; issue #63).
    ``sync_references=False`` is the explicit offline opt-out, valid only
    when a previous run already populated the reference cache. The jwst
    pipeline syncs its own references lazily through CRDS_PATH, so the
    explicit bestrefs step is HST-only.
    """
    if observatory != "hst":
        return False
    if sync_references:
        return True
    if not present:
        raise RuntimeError(
            "sync_references=False but no CRDS references are cached — "
            "an offline run needs a reference cache populated by a "
            "previous synced run"
        )
    return False


def sync_best_references(exposures: List[Path]) -> None:
    """Fetch + assign best references for the exposures (network)."""
    if not exposures:
        raise ValueError("no exposures to sync references for")
    cmd = [
        sys.executable,
        "-m",
        "crds.bestrefs",
        "--files",
        *[str(p) for p in exposures],
        "--sync-references=1",
        "--update-bestrefs",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(
            result.stdout.splitlines()[-5:] + result.stderr.splitlines()[-5:]
        )
        raise RuntimeError(f"crds.bestrefs failed (exit {result.returncode}):\n{tail}")
