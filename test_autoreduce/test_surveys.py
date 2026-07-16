import numpy as np
import pytest

from autoreduce import instruments
from autoreduce.surveys import fetch as fetch_mod
from autoreduce.surveys import pipeline as survey_pipeline
from autoreduce.target import TargetSpec

RA, DEC = 2.0, -0.1


class TestAdapters:
    def test_registered_with_cutout_domain(self):
        for key in ("legacy_surveys", "sdss", "panstarrs"):
            adapter = instruments.get(key)
            assert adapter.domain == "cutout"

    def test_only_legacy_ships_noise(self):
        assert instruments.get("legacy_surveys").noise_available
        assert not instruments.get("sdss").noise_available
        assert not instruments.get("panstarrs").noise_available


class TestUrls:
    def test_legacy_cutout_url(self):
        url = fetch_mod.legacy_cutout_url(RA, DEC, 64, ("g", "r", "z"), invvar=True)
        assert "fits-cutout?ra=2.0&dec=-0.1" in url
        assert "layer=ls-dr10" in url
        assert "bands=grz" in url and "size=64" in url
        assert url.endswith("&invvar")

    def test_legacy_url_without_invvar(self):
        url = fetch_mod.legacy_cutout_url(RA, DEC, 64, ("g",), invvar=False)
        assert "invvar" not in url

    def test_ps1_urls(self):
        assert "filters=gri" in fetch_mod.ps1_filenames_url(RA, DEC, ("g", "r", "i"))
        url = fetch_mod.ps1_fitscut_url("/rings/stk.g.unconv.fits", RA, DEC, 128)
        assert "red=/rings/stk.g.unconv.fits" in url
        assert "size=128" in url and "format=fits" in url


class TestRmsFromInvvar:
    def test_conversion_and_nan_policy(self):
        ivar = np.array([[4.0, 0.0], [np.nan, 25.0]])
        rms = survey_pipeline.rms_from_invvar(ivar)
        assert rms[0, 0] == pytest.approx(0.5)
        assert rms[1, 1] == pytest.approx(0.2)
        assert np.isnan(rms[0, 1]) and np.isnan(rms[1, 0])

    def test_all_bad_is_loud(self):
        with pytest.raises(ValueError, match="no positive pixels"):
            survey_pipeline.rms_from_invvar(np.zeros((3, 3)))


class TestSpecDial:
    def test_survey_bands_from_yaml(self, tmp_path):
        path = tmp_path / "spec.yaml"
        path.write_text("name: t\nra: 2.0\ndec: -0.1\nsurvey_bands: [g, r]\n")
        assert TargetSpec.from_yaml(path).survey_bands == ("g", "r")


class TestReduceSurveyTarget:
    def _fake_fetch(self, with_ivar):
        def fetcher(ra, dec, size, bands):
            from astropy.io import fits

            out = {}
            for band in bands:
                payload = {
                    "data": np.full((size, size), 2.0),
                    "header": fits.Header({"BUNIT": "nanomaggy"}),
                }
                if with_ivar:
                    payload["ivar"] = np.full((size, size), 4.0)
                out[band] = payload
            return out

        return fetcher

    def test_packages_data_and_optional_noise(self, tmp_path, monkeypatch):
        from astropy.io import fits

        monkeypatch.setitem(
            fetch_mod.FETCHERS, "legacy", self._fake_fetch(with_ivar=True)
        )
        spec = TargetSpec(
            name="t", ra=RA, dec=DEC, instrument="legacy_surveys",
            cutout_shape=(21, 21), survey_bands=("g", "r"),
        )
        record = survey_pipeline.reduce_survey_target(
            spec, instruments.get("legacy_surveys"), tmp_path
        )
        data = fits.getdata(tmp_path / "legacy_surveys" / "g" / "data.fits")
        noise = fits.getdata(tmp_path / "legacy_surveys" / "g" / "noise_map.fits")
        assert data.shape == (21, 21)
        assert noise[0, 0] == pytest.approx(0.5)
        assert record["acquire"]["bands_delivered"] == ["g", "r"]
        assert "legacy_surveys/r/noise_map.fits" in record["package"]["products"]
        assert record["package"]["products_optional"]["psf"].startswith("not produced")

    def test_data_only_survey_states_why(self, tmp_path, monkeypatch):
        monkeypatch.setitem(
            fetch_mod.FETCHERS, "sdss", self._fake_fetch(with_ivar=False)
        )
        spec = TargetSpec(name="t", ra=RA, dec=DEC, instrument="sdss")
        record = survey_pipeline.reduce_survey_target(
            spec, instruments.get("sdss"), tmp_path
        )
        assert not any("noise_map" in p for p in record["package"]["products"])
        assert "no variance product" in (
            record["package"]["products_optional"]["noise_map"]
        )
        assert not (tmp_path / "sdss" / "g" / "noise_map.fits").exists()

    def test_reduce_target_dispatches_cutout_domain(self, tmp_path, monkeypatch):
        from autoreduce.pipeline import reduce_target

        monkeypatch.setitem(
            fetch_mod.FETCHERS, "ps1", self._fake_fetch(with_ivar=False)
        )
        spec = TargetSpec(name="t", ra=RA, dec=DEC, instrument="panstarrs")
        record = reduce_target(spec, tmp_path / "cache", tmp_path / "out")
        assert record["instrument"] == "panstarrs"
        assert (tmp_path / "out" / "t" / "reduction.json").exists()
        assert (tmp_path / "out" / "t" / "panstarrs" / "g" / "data.fits").exists()
