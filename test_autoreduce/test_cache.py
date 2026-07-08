import json

import pytest

from autoreduce.acquire.cache import ExposureCache


def _fake_exposure(cache, target, name, size=1000):
    target_dir = cache.target_dir(target)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / name
    path.write_bytes(b"x" * size)
    return path


class TestManifest:
    def test_record_and_read_back(self, tmp_path):
        cache = ExposureCache(tmp_path)
        p = _fake_exposure(cache, "lens1", "a_flc.fits")
        cache.record_download("lens1", [str(p)], source="mast")
        assert cache.exposures_for("lens1") == [p]
        manifest = json.loads(cache.manifest_path.read_text())
        assert manifest["targets"]["lens1"]["source"] == "mast"

    def test_missing_files_fail_loudly(self, tmp_path):
        cache = ExposureCache(tmp_path)
        p = _fake_exposure(cache, "lens1", "a_flc.fits")
        cache.record_download("lens1", [str(p)], source="mast")
        p.unlink()
        with pytest.raises(FileNotFoundError, match="gone"):
            cache.exposures_for("lens1")

    def test_unknown_target_returns_empty(self, tmp_path):
        cache = ExposureCache(tmp_path)
        assert cache.exposures_for("nope") == []

    def test_incompatible_manifest_schema_fails_loudly(self, tmp_path):
        cache = ExposureCache(tmp_path)
        # A spike-era manifest: same filename, different schema.
        cache.manifest_path.write_text(json.dumps({"target": {}, "flc_files": []}))
        with pytest.raises(ValueError, match="not an ExposureCache manifest"):
            cache.exposures_for("lens1")


class TestEviction:
    def test_evict_removes_files_keeps_provenance(self, tmp_path):
        cache = ExposureCache(tmp_path)
        p = _fake_exposure(cache, "lens1", "a_flc.fits")
        cache.record_download("lens1", [str(p)], source="mast")
        cache.evict("lens1")
        assert not p.exists()
        manifest = cache.read_manifest()
        assert manifest["targets"]["lens1"]["evicted"]
        assert manifest["targets"]["lens1"]["files"]  # provenance retained
        assert cache.exposures_for("lens1") == []

    def test_evict_unknown_target_raises(self, tmp_path):
        with pytest.raises(KeyError):
            ExposureCache(tmp_path).evict("nope")

    def test_cap_evicts_oldest_completed_first(self, tmp_path):
        cache = ExposureCache(tmp_path, size_cap_bytes=2500)
        for i, name in enumerate(["old", "mid", "new"]):
            p = _fake_exposure(cache, name, "a_flc.fits", size=1000)
            cache.record_download(name, [str(p)], source="mast")
            # Distinct timestamps: rewrite downloaded_at deterministically.
            manifest = cache.read_manifest()
            manifest["targets"][name]["downloaded_at"] = f"2026-07-08T00:0{i}:00Z"
            cache._write_manifest(manifest)
        cache.mark_completed("old")
        cache.mark_completed("mid")
        # "new" is not completed: never evicted even over cap.
        evicted = cache.enforce_cap()
        assert evicted == ["old"]
        assert cache.size_bytes() <= 2500

    def test_uncapped_never_evicts(self, tmp_path):
        cache = ExposureCache(tmp_path)
        p = _fake_exposure(cache, "lens1", "a_flc.fits")
        cache.record_download("lens1", [str(p)], source="mast")
        cache.mark_completed("lens1")
        assert cache.enforce_cap() == []
        assert p.exists()

    def test_references_excluded_from_size(self, tmp_path):
        cache = ExposureCache(tmp_path)
        refs = cache.references_dir / "references" / "hst" / "acs"
        refs.mkdir(parents=True)
        (refs / "flat.fits").write_bytes(b"x" * 10_000)
        assert cache.size_bytes() == 0
