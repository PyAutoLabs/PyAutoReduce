import numpy as np
import pytest

from autoreduce import instruments
from autoreduce.instruments.nircam import nircam_adapter_for_filter
from autoreduce.noise.jwst_rms import noise_map_from_error


class TestNIRCamAdapters:
    def test_channels_registered(self):
        assert "nircam_sw" in instruments.registered_keys()
        assert "nircam_lw" in instruments.registered_keys()

    def test_jwst_routing_fields(self):
        for key in ("nircam_sw", "nircam_lw"):
            a = instruments.get(key)
            assert a.observatory == "jwst"
            assert a.combine_backend == "jwst_image3"
            assert a.crds_server_url == "https://jwst-crds.stsci.edu"
            assert a.mast_obs_collection == "JWST"
            assert a.calibrated_suffix == "CAL"
            assert not a.supports_cte_correction

    def test_hst_adapters_unchanged(self):
        # Phase-1/2 regression: HST adapters keep the astrodrizzle backend
        # and the HST CRDS server via the new defaulted fields.
        for key in ("acs_wfc", "wfc3_uvis", "wfc3_ir"):
            a = instruments.get(key)
            assert a.observatory == "hst"
            assert a.combine_backend == "astrodrizzle"
            assert a.crds_server_url == "https://hst-crds.stsci.edu"
            assert a.mast_obs_collection == "HST"

    def test_scales_match_cosmos_web_convention(self):
        assert instruments.get("nircam_sw").recommended_final_scale == 0.03
        assert instruments.get("nircam_lw").recommended_final_scale == 0.06

    def test_filter_routing(self):
        assert nircam_adapter_for_filter("F115W").key == "nircam_sw"
        assert nircam_adapter_for_filter("F150W").key == "nircam_sw"
        assert nircam_adapter_for_filter("F277W").key == "nircam_lw"
        assert nircam_adapter_for_filter("f444w").key == "nircam_lw"
        with pytest.raises(KeyError, match="unknown NIRCam filter"):
            nircam_adapter_for_filter("F814W")


class TestBackendDispatch:
    def test_unknown_backend_rejected(self):
        from dataclasses import replace

        from autoreduce.drizzle.combine import combine
        from autoreduce.target import TargetSpec

        broken = replace(instruments.get("acs_wfc"), combine_backend="magic")
        spec = TargetSpec(name="x", ra=0.0, dec=0.0)
        with pytest.raises(ValueError, match="unknown combine backend"):
            combine([], spec, broken, output_dir="/tmp/nowhere")


class TestJWSTCRDSEnvironment:
    def test_jwst_env_uses_jwst_server_and_no_iraf_var(self, tmp_path, monkeypatch):
        import os

        from autoreduce.acquire.crds import configure_environment

        monkeypatch.delenv("CRDS_SERVER_URL", raising=False)
        monkeypatch.delenv("CRDS_PATH", raising=False)
        env = configure_environment(tmp_path, instruments.get("nircam_lw"))
        assert env["CRDS_SERVER_URL"] == "https://jwst-crds.stsci.edu"
        assert env["CRDS_PATH"] == str(tmp_path)
        # No jref/iref-style variable for the jwst pipeline.
        assert set(env) == {"CRDS_SERVER_URL", "CRDS_PATH"}
        assert os.environ["CRDS_SERVER_URL"] == "https://jwst-crds.stsci.edu"

    def test_hst_env_still_sets_iraf_var(self, tmp_path, monkeypatch):
        from autoreduce.acquire.crds import configure_environment

        monkeypatch.delenv("jref", raising=False)
        env = configure_environment(tmp_path, instruments.get("acs_wfc"))
        assert env["CRDS_SERVER_URL"] == "https://hst-crds.stsci.edu"
        assert env["jref"].endswith("references/hst/acs/")


class TestJWSTNoise:
    def test_reads_err_and_applies_r(self):
        err = np.full((10, 10), 0.02)
        sci = np.random.default_rng(0).normal(0.0, 0.02, (10, 10))
        noise, consistency = noise_map_from_error(err, sci, correlated_noise_factor=1.5)
        assert noise == pytest.approx(np.full((10, 10), 0.03))
        assert consistency["err_5th_percentile_pre_R"] == pytest.approx(0.02)

    def test_bad_err_pixels_become_nan(self):
        err = np.array([[0.02, 0.0], [np.nan, 0.02]])
        sci = np.zeros((2, 2))
        noise, _ = noise_map_from_error(err, sci)
        assert np.isnan(noise[0, 1]) and np.isnan(noise[1, 0])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            noise_map_from_error(np.zeros((2, 2)), np.zeros((3, 3)))

    def test_sub_unity_r_raises(self):
        with pytest.raises(ValueError):
            noise_map_from_error(
                np.ones((2, 2)), np.ones((2, 2)), correlated_noise_factor=0.5
            )


def test_find_stars_handles_nan_borders():
    from autoreduce.psf.stars import StarSelection, find_stars

    rng = np.random.default_rng(2)
    sci = rng.normal(0.0, 0.01, (200, 200))
    sci[:20, :] = np.nan  # coverage border
    yy, xx = np.mgrid[0:200, 0:200]
    for x0, y0 in [(60, 60), (140, 150), (100, 60)]:
        sci += 5.0 * np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 1.2**2)))
    stars = find_stars(
        sci,
        StarSelection(min_separation_pix=10.0, edge_margin_pix=25),
        target_xy=(0.0, 0.0),
        peak_max=None,
    )
    assert stars is not None and len(stars) == 3


class TestFootprintFilter:
    def _cal_file(self, tmp_path, name, crval):
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS

        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = list(crval)
        wcs.wcs.crpix = [50.5, 50.5]
        wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]  # 100" x 100" footprint
        sci = fits.ImageHDU(np.zeros((100, 100), dtype="f4"), header=wcs.to_header())
        sci.name = "SCI"
        path = tmp_path / name
        fits.HDUList([fits.PrimaryHDU(), sci]).writeto(path)
        return path

    def test_keeps_covering_drops_off_target(self, tmp_path):
        from autoreduce.acquire.footprint import filter_to_target

        on = self._cal_file(tmp_path, "on_cal.fits", (150.1, 1.893))
        off = self._cal_file(tmp_path, "off_cal.fits", (150.5, 2.4))
        covering, skipped = filter_to_target(
            [on, off], ra=150.1, dec=1.893, margin_arcsec=10.0
        )
        assert covering == [on] and skipped == [off]

    def test_nothing_covering_is_loud(self, tmp_path):
        import pytest as _pytest

        from autoreduce.acquire.footprint import filter_to_target

        off = self._cal_file(tmp_path, "off_cal.fits", (150.5, 2.4))
        with _pytest.raises(LookupError, match="cover"):
            filter_to_target([off], ra=150.1, dec=1.893, margin_arcsec=10.0)
