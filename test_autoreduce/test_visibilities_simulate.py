import numpy as np
import pytest

from autoreduce.target import TargetSpec
from autoreduce.visibilities import simulate as sim_mod

RA, DEC = 137.0, 2.1


def _input(tmp_path, flux=0.01, shape=(64, 64)):
    from astropy.io import fits

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    img = np.exp(-0.5 * ((yy - 32) ** 2 + (xx - 32) ** 2) / 9.0)
    img = flux * img / img.sum()
    path = tmp_path / "model.fits"
    fits.PrimaryHDU(img.astype(np.float32)).writeto(path)
    return path, img


def _spec(tmp_path, **extra):
    path, _ = _input(tmp_path)
    return TargetSpec(
        name="simtest",
        ra=RA,
        dec=DEC,
        instrument="alma",
        inject_image=str(path),
        inject_pixel_scale=0.02,
        **extra,
    )


class TestSkymodel:
    def test_axes_units_and_flux(self, tmp_path):
        from astropy.io import fits

        img = np.full((32, 32), 2.0e-5)
        out = sim_mod.skymodel_fits(img, 0.02, RA, DEC, 230.0, tmp_path / "sky.fits")
        with fits.open(out) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
        assert data.shape == (1, 1, 32, 32)
        assert data.sum() == pytest.approx(img.sum(), rel=1e-6)
        assert hdr["BUNIT"] == "Jy/pixel"
        assert hdr["CTYPE1"] == "RA---SIN" and hdr["CTYPE2"] == "DEC--SIN"
        assert hdr["CRVAL1"] == RA and hdr["CRVAL2"] == DEC
        assert hdr["CDELT2"] == pytest.approx(0.02 / 3600.0)
        assert hdr["CRVAL4"] == pytest.approx(230.0e9)

    def test_bad_scale_is_loud(self, tmp_path):
        with pytest.raises(ValueError, match="positive"):
            sim_mod.skymodel_fits(
                np.ones((8, 8)), 0.0, RA, DEC, 230.0, tmp_path / "sky.fits"
            )


class TestSimobserveKwargs:
    def test_thermal_noise_default(self, tmp_path):
        spec = _spec(tmp_path)
        kwargs = sim_mod.simobserve_kwargs(spec, tmp_path / "sky.fits")
        assert kwargs["project"] == "simtest_sim"
        assert kwargs["thermalnoise"] == "tsys-atm"
        assert kwargs["user_pwv"] == 0.5
        assert kwargs["integration"] == "10s"
        assert kwargs["totaltime"] == "1800s"
        assert kwargs["antennalist"] == "alma.cycle8.3.cfg"
        assert kwargs["graphics"] == "none"

    def test_pwv_zero_disables_noise(self, tmp_path):
        spec = _spec(tmp_path, alma_sim_pwv_mm=0.0)
        kwargs = sim_mod.simobserve_kwargs(spec, tmp_path / "sky.fits")
        assert kwargs["thermalnoise"] == ""
        assert "user_pwv" not in kwargs


class TestMsPath:
    def test_noisy_and_clean_conventions(self, tmp_path):
        project = tmp_path / "simtest_sim"
        noisy = sim_mod.simulated_ms_path(project, "alma.cycle8.3.cfg", noisy=True)
        clean = sim_mod.simulated_ms_path(project, "alma.cycle8.3.cfg", noisy=False)
        assert noisy.name == "simtest_sim.alma.cycle8.3.noisy.ms"
        assert clean.name == "simtest_sim.alma.cycle8.3.ms"


class TestDials:
    def test_negative_totaltime_is_loud(self, tmp_path):
        with pytest.raises(ValueError, match="alma_sim_totaltime_s"):
            _spec(tmp_path, alma_sim_totaltime_s=-1.0)

    def test_negative_pwv_is_loud(self, tmp_path):
        with pytest.raises(ValueError, match="alma_sim_pwv_mm"):
            _spec(tmp_path, alma_sim_pwv_mm=-0.1)


class TestDispatch:
    def test_alma_inject_reaches_simulate(self, tmp_path, monkeypatch):
        from autoreduce.pipeline import reduce_target

        def boom(spec, work_dir):
            raise RuntimeError("simulate reached")

        monkeypatch.setattr(sim_mod, "simulate_ms", boom)
        spec = _spec(tmp_path)
        with pytest.raises(RuntimeError, match="simulate reached"):
            reduce_target(spec, tmp_path / "cache", tmp_path / "out")

    def test_real_alma_path_still_requires_dials(self, tmp_path):
        from autoreduce.pipeline import reduce_target

        spec = TargetSpec(name="t", ra=RA, dec=DEC, instrument="alma")
        with pytest.raises(ValueError, match="alma_uids"):
            reduce_target(spec, tmp_path / "cache", tmp_path / "out")

    def test_cutout_domain_injection_stays_loud(self, tmp_path):
        from autoreduce.pipeline import reduce_target

        spec = _spec(tmp_path)
        spec = TargetSpec(**{**spec.as_dict(), "instrument": "legacy_surveys"})
        with pytest.raises(ValueError, match="inject_image supports"):
            reduce_target(spec, tmp_path / "cache", tmp_path / "out")
