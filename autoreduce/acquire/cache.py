"""
The transient exposure cache (design doc stage 1).

Full-frame exposures are transient: download per target, reduce, package,
evict. A manifest records what came from where so eviction never costs
reproducibility. CRDS reference files live under the same root but are the
one component never evicted per target — they are shared across targets.
"""

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

MANIFEST_NAME = "cache_manifest.json"
REFERENCES_DIRNAME = "crds"


@dataclass
class ExposureCache:
    """Size-capped per-target exposure storage with a provenance manifest."""

    root: Path
    size_cap_bytes: Optional[int] = None  # None = uncapped

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- manifest -----------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def read_manifest(self) -> Dict:
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            if "targets" not in manifest:
                raise ValueError(
                    f"{self.manifest_path} is not an ExposureCache manifest "
                    f"(keys: {sorted(manifest)}); refusing to guess — point the "
                    f"cache at a fresh directory (spike-era caches are not "
                    f"compatible)"
                )
            return manifest
        return {"targets": {}}

    def _write_manifest(self, manifest: Dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2))

    # -- per-target lifecycle ------------------------------------------------

    def target_dir(self, target_name: str) -> Path:
        return self.root / target_name

    def record_download(
        self, target_name: str, files: List[str], source: str
    ) -> None:
        """Register downloaded exposures so a re-run can re-fetch deterministically."""
        manifest = self.read_manifest()
        manifest["targets"][target_name] = {
            "files": sorted(str(f) for f in files),
            "source": source,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evicted": False,
        }
        self._write_manifest(manifest)

    def exposures_for(self, target_name: str) -> List[Path]:
        entry = self.read_manifest()["targets"].get(target_name)
        if entry is None or entry["evicted"]:
            return []
        paths = [Path(f) for f in entry["files"]]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"cache manifest lists exposures that are gone (not via evict): "
                f"{[str(m) for m in missing]}"
            )
        return paths

    def evict(self, target_name: str) -> None:
        """Drop a target's exposures; the manifest keeps the provenance."""
        manifest = self.read_manifest()
        entry = manifest["targets"].get(target_name)
        if entry is None:
            raise KeyError(f"no cache entry for target {target_name!r}")
        target_dir = self.target_dir(target_name)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        entry["evicted"] = True
        self._write_manifest(manifest)

    # -- size cap -------------------------------------------------------------

    def size_bytes(self) -> int:
        """Total evictable payload (excludes the shared CRDS references)."""
        total = 0
        for path in self.root.rglob("*"):
            if (
                path.is_file()
                and REFERENCES_DIRNAME not in path.parts
                and path.name != MANIFEST_NAME
            ):
                total += path.stat().st_size
        return total

    def enforce_cap(self) -> List[str]:
        """
        Evict oldest completed targets until under the cap. Returns the
        evicted target names. Targets are eligible only once marked evictable
        (their products written) via ``mark_completed``.
        """
        if self.size_cap_bytes is None:
            return []
        manifest = self.read_manifest()
        evicted: List[str] = []
        entries = sorted(
            (
                (name, e)
                for name, e in manifest["targets"].items()
                if not e["evicted"] and e.get("completed", False)
            ),
            key=lambda item: item[1]["downloaded_at"],
        )
        for name, _ in entries:
            if self.size_bytes() <= self.size_cap_bytes:
                break
            self.evict(name)
            evicted.append(name)
        return evicted

    def mark_completed(self, target_name: str) -> None:
        """Products for this target are written; its exposures may be evicted."""
        manifest = self.read_manifest()
        entry = manifest["targets"].get(target_name)
        if entry is None:
            raise KeyError(f"no cache entry for target {target_name!r}")
        entry["completed"] = True
        self._write_manifest(manifest)

    # -- CRDS references -------------------------------------------------------

    @property
    def references_dir(self) -> Path:
        return self.root / REFERENCES_DIRNAME
