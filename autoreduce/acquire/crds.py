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
    """
    env = {
        "CRDS_SERVER_URL": CRDS_SERVER_URL,
        "CRDS_PATH": str(references_root),
        adapter.reference_env_key: str(
            Path(references_root) / "references" / "hst" / adapter.key.split("_")[0]
        )
        + "/",
    }
    for key, value in env.items():
        os.environ.setdefault(key, value)
    return env


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
