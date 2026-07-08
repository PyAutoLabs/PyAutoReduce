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

CRDS_SERVER_URL = "https://hst-crds.stsci.edu"


def configure_environment(references_root: Path, adapter: InstrumentAdapter) -> dict:
    """
    Set the CRDS variables for this process. Must run before drizzlepac is
    imported anywhere in the process. Returns the mapping applied.

    Deliberately overrides any inherited CRDS_PATH/jref: the pipeline is a
    pure function of the target spec plus the archive, so its reference files
    live in *its* cache, not wherever the shell environment happens to point.
    """
    env = {
        "CRDS_SERVER_URL": CRDS_SERVER_URL,
        "CRDS_PATH": str(references_root),
        adapter.reference_env_key: str(
            Path(references_root) / adapter.crds_reference_subpath
        )
        + "/",
    }
    os.environ.update(env)
    return env


def references_present(references_root: Path, adapter: InstrumentAdapter) -> bool:
    """True if the instrument's reference directory exists and is non-empty."""
    ref_dir = Path(references_root) / adapter.crds_reference_subpath
    return ref_dir.is_dir() and any(ref_dir.iterdir())


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
