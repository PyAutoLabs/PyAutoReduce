"""
The provenance record (design doc stage 6): ``reduction.json`` restores
permanently what the legacy stripped-header datasets lost — where every
pixel came from and how it was made.
"""

import json
import platform
import time
from pathlib import Path
from typing import Dict


def software_versions() -> Dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("autoreduce", "astropy", "numpy", "photutils", "drizzlepac", "astroquery"):
        try:
            versions[package] = __import__(package).__version__
        except Exception:
            versions[package] = "not-installed"
    return versions


def write_reduction_json(out_dir: Path, record: Dict) -> Path:
    """Write the accumulated per-stage provenance with a metadata envelope."""
    payload = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "software": software_versions(),
        **record,
    }
    path = Path(out_dir) / "reduction.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
