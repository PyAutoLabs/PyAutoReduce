import numpy as np
import pytest

from autoreduce import instruments
from autoreduce.inject import imaging as inject_mod
from autoreduce.target import TargetSpec

RA, DEC = 2.0, -0.1


def _frame_hdul(bunit="ELECTRONS", exptime=500.0, shape=(200, 200), nchips=1):
    from astropy.io import fits
    from astropy.wcs import WCS

    primary = fits.PrimaryHDU()
    primary.header["ROOTNAME"] = "j8pu42vlq"
    primary.header["EXPTIME"] = exptime
    primary.header["INSTRUME"] = "ACS"
    primary.header["TELESCOP"] = "HST"
    hdus = [primary]
    for chip in range(1, nchips + 1):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [RA, DEC]
        wcs.wcs.crpix = [100.5, 100.5]
        wcs.wcs.cdelt = [-0.05 / 3600.0, 0.05 / 3600.0]
        hdr = wcs.to_header()
        hdr["BUNIT"] = bunit
        sci = np.full(shape, 140.0)
        err = np.full(shape, 5.0)
        dq = np.zeros(shape, dtype=np.int32)
        hdus += [
            fits.ImageHDU(data=sci, header=hdr, name="SCI", ver=chip),
            fits.ImageHDU(data=err, header=hdr.copy(), name="ERR", ver=chip),
            fits.ImageHDU(data=dq, header=hdr.copy(), name="DQ", ver=chip),
        ]
    return fits.HDUList(hdus)


def _gaussian_input(shape=(61, 61), sigma=3.0, flux=1000.0):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    cy, cx = shape[0] // 2, shape[1] // 2
    img = np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / sigma**2)
    return flux * img / img.sum()


def _delta_psf(shape=(11, 11)):
    psf = np.zeros(shape)
    psf[shape[0] // 2, shape[1] // 2] = 1.0
    return psf


def _write_input(tmp_path, data, name="input.fits"):
    from astropy.io import fits

    path = tmp_path / name
    fits.PrimaryHDU(data.astype(np.float32)).writeto(path, overwrite=True)
    return path


def _spec(tmp_path, input_data=None, **extra):
    data = _gaussian_input() if input_data is None else input_data
    image_path = _write_input(tmp_path, data)
    psf_path = tmp_path / "psf.fits"
    from astropy.io import fits

    fits.PrimaryHDU(_delta_psf()).writeto(psf_path, overwrite=True)
    extra.setdefault("inject_psf", str(psf_path))
    return TargetSpec(
        name="t",
        ra=RA,
        dec=DEC,
        inject_image=str(image_path),
        inject_pixel_scale=0.025,
        **extra,
    )


class TestTargetSpecDials:
    def test_inject_image_requires_pixel_scale(self):
        with pytest.raises(ValueError, match="inject_pixel_scale"):
            TargetSpec(name="t", ra=RA, dec=DEC, inject_image="model.fits")

    def test_inject_dials_without_image_are_loud(self):
        with pytest.raises(ValueError, match="inject_image is None"):
            TargetSpec(name="t", ra=RA, dec=DEC, inject_pixel_scale=0.03)

    def test_inject_position_must_be_pair(self):
        with pytest.raises(ValueError, match="inject_position"):
            TargetSpec(
                name="t",
                ra=RA,
                dec=DEC,
                inject_image="model.fits",
                inject_pixel_scale=0.03,
                inject_position=(1.0,),
            )

    def test_from_yaml_tuples_inject_position(self, tmp_path):
        path = tmp_path / "spec.yaml"
        path.write_text(
            "name: t\nra: 2.0\ndec: -0.1\ninject_image: model.fits\n"
            "inject_pixel_scale: 0.03\ninject_position: [2.0, -0.1]\n"
        )
        spec = TargetSpec.from_yaml(path)
        assert spec.inject_position == (2.0, -0.1)


class TestInputContract:
    def test_negative_pixels_are_loud(self, tmp_path):
        data = _gaussian_input()
        data[0, 0] = -1.0
        spec = _spec(tmp_path, input_data=data)
        with pytest.raises(ValueError, match="negative"):
            inject_mod.load_input_image(spec)

    def test_non_finite_pixels_are_loud(self, tmp_path):
        data = _gaussian_input()
        data[0, 0] = np.nan
        spec = _spec(tmp_path, input_data=data)
        with pytest.raises(ValueError, match="non-finite"):
            inject_mod.load_input_image(spec)

    def test_position_defaults_to_target(self, tmp_path):
        spec = _spec(tmp_path)
        _data, wcs = inject_mod.load_input_image(spec)
        assert wcs.wcs.crval[0] == pytest.approx(RA)
        assert wcs.wcs.crval[1] == pytest.approx(DEC)


class TestUnitsFactor:
    def test_electrons_add_as_counts(self):
        assert inject_mod._injection_units_factor("ELECTRONS", 500.0) == 1.0

    def test_cps_divides_by_exptime(self):
        assert inject_mod._injection_units_factor(
            "ELECTRONS/S", 500.0
        ) == pytest.approx(1.0 / 500.0)

    def test_unknown_bunit_is_loud(self):
        with pytest.raises(ValueError, match="unrecognised SCI BUNIT"):
            inject_mod._injection_units_factor("MJY/SR", 500.0)

    def test_cps_without_exptime_is_loud(self):
        with pytest.raises(ValueError, match="EXPTIME"):
            inject_mod._injection_units_factor("ELECTRONS/S", 0.0)


class TestConvolve:
    def test_unit_kernel_preserves_flux(self):
        image = _gaussian_input(flux=250.0)
        kernel = np.ones((5, 5)) / 25.0
        out = inject_mod.convolve_with_psf(image, kernel)
        assert out.sum() == pytest.approx(image.sum(), rel=1e-6)
        assert out.shape == image.shape

    def test_even_kernel_is_loud(self):
        with pytest.raises(ValueError, match="odd"):
            inject_mod.convolve_with_psf(np.ones((10, 10)), np.ones((4, 4)) / 16.0)


class TestRenderToChip:
    def test_flux_conserved_onto_frame_grid(self):
        pytest.importorskip("drizzle")
        from astropy.wcs import WCS

        input_cps = _gaussian_input(flux=1000.0)
        in_wcs = inject_mod.input_wcs(input_cps.shape, 0.025, RA, DEC)
        chip_wcs = WCS(naxis=2)
        chip_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        chip_wcs.wcs.crval = [RA, DEC]
        chip_wcs.wcs.crpix = [100.5, 100.5]
        chip_wcs.wcs.cdelt = [-0.05 / 3600.0, 0.05 / 3600.0]
        rendered = inject_mod.render_to_chip(input_cps, in_wcs, chip_wcs, (200, 200))
        assert rendered.sum() == pytest.approx(1000.0, rel=1e-3)

    def test_off_chip_footprint_renders_nothing(self):
        pytest.importorskip("drizzle")
        from astropy.wcs import WCS

        input_cps = _gaussian_input(flux=1000.0)
        in_wcs = inject_mod.input_wcs(input_cps.shape, 0.025, RA + 1.0, DEC)
        chip_wcs = WCS(naxis=2)
        chip_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        chip_wcs.wcs.crval = [RA, DEC]
        chip_wcs.wcs.crpix = [100.5, 100.5]
        chip_wcs.wcs.cdelt = [-0.05 / 3600.0, 0.05 / 3600.0]
        rendered = inject_mod.render_to_chip(input_cps, in_wcs, chip_wcs, (200, 200))
        assert rendered.sum() == pytest.approx(0.0, abs=1e-12)


class TestInjectIntoExposures:
    def _run(self, tmp_path, spec=None, exptime=500.0, bunit="ELECTRONS"):
        pytest.importorskip("drizzle")
        exposure = tmp_path / "j8pu42vlq_flc.fits"
        if not exposure.exists():
            _frame_hdul(bunit=bunit, exptime=exptime).writeto(exposure)
        spec = spec or _spec(tmp_path)
        adapter = instruments.get(spec.instrument)
        work_dir = tmp_path / "work"
        work_dir.mkdir(exist_ok=True)
        return inject_mod.inject_into_exposures([exposure], spec, adapter, work_dir)

    def test_counts_added_and_cache_untouched(self, tmp_path):
        from astropy.io import fits

        paths, fragment = self._run(tmp_path)
        assert paths[0].parent.name == "injected"
        injected_sci = fits.getdata(paths[0], ("SCI", 1))
        original_sci = fits.getdata(tmp_path / "j8pu42vlq_flc.fits", ("SCI", 1))
        added = injected_sci.sum() - original_sci.sum()
        # Poisson realisation of flux * exptime = 1000 * 500 e-.
        assert added == pytest.approx(1000.0 * 500.0, rel=0.01)
        assert fragment["total_injected_e"] == pytest.approx(added, rel=1e-6)
        assert original_sci.sum() == pytest.approx(140.0 * 200 * 200)

    def test_err_grows_in_quadrature_and_dq_untouched(self, tmp_path):
        from astropy.io import fits

        paths, _ = self._run(tmp_path)
        err = fits.getdata(paths[0], ("ERR", 1))
        dq = fits.getdata(paths[0], ("DQ", 1))
        assert np.all(err >= 5.0 - 1e-9)
        assert err.max() > 5.0
        assert not dq.any()

    def test_header_stamp_and_provenance(self, tmp_path):
        from astropy.io import fits

        paths, fragment = self._run(tmp_path)
        primary = fits.getheader(paths[0])
        assert primary["INJECTED"] is True
        assert primary["INJIMG"] == "input.fits"
        assert fragment["frames"][0]["exposure"] == "j8pu42vlq_flc.fits"
        assert fragment["frames"][0]["chips"][0]["extver"] == 1
        assert fragment["psf_source"].startswith("inject_psf")

    def test_seed_reproducibility(self, tmp_path):
        from astropy.io import fits

        paths_a, _ = self._run(tmp_path)
        sci_a = fits.getdata(paths_a[0], ("SCI", 1)).copy()
        (paths_a[0]).unlink()
        paths_b, _ = self._run(tmp_path)
        sci_b = fits.getdata(paths_b[0], ("SCI", 1))
        assert np.array_equal(sci_a, sci_b)

    def test_different_seed_differs(self, tmp_path):
        from astropy.io import fits

        paths_a, _ = self._run(tmp_path)
        sci_a = fits.getdata(paths_a[0], ("SCI", 1)).copy()
        (paths_a[0]).unlink()
        paths_b, _ = self._run(tmp_path, spec=_spec(tmp_path, inject_seed=1))
        sci_b = fits.getdata(paths_b[0], ("SCI", 1))
        assert not np.array_equal(sci_a, sci_b)

    def test_cps_frames_scale_by_exptime(self, tmp_path):
        from astropy.io import fits

        paths, _ = self._run(tmp_path, bunit="ELECTRONS/S")
        injected_sci = fits.getdata(paths[0], ("SCI", 1))
        added = injected_sci.sum() - 140.0 * 200 * 200
        # cps frames receive counts / exptime: the added rate is the flux.
        assert added == pytest.approx(1000.0, rel=0.01)


class TestPipelineGate:
    def test_non_hst_instrument_is_loud(self, tmp_path):
        from autoreduce.pipeline import reduce_target

        spec = _spec(tmp_path, instrument="nirc2_narrow")
        with pytest.raises(ValueError, match="HST astrodrizzle path only"):
            reduce_target(spec, tmp_path / "cache", tmp_path / "out")

    def test_jwst_backend_is_loud(self, tmp_path):
        from autoreduce.pipeline import reduce_target

        spec = _spec(tmp_path, instrument="nircam_lw")
        with pytest.raises(ValueError, match="HST astrodrizzle path only"):
            reduce_target(spec, tmp_path / "cache", tmp_path / "out")
