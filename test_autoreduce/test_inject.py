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
    extra.setdefault("inject_pixel_scale", 0.025)
    return TargetSpec(
        name="t",
        ra=RA,
        dec=DEC,
        inject_image=str(image_path),
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


class TestChipUnits:
    def _hst(self):
        return instruments.get("acs_wfc")

    def _jwst_header(self, photmjsr=0.5, pixar_sr=2.0e-14):
        from astropy.io import fits

        return fits.Header({"PHOTMJSR": photmjsr, "PIXAR_SR": pixar_sr})

    def test_electrons_add_as_counts(self):
        from astropy.io import fits

        e_per_in, sci_per_e, _ = inject_mod._chip_units(
            "ELECTRONS", fits.Header(), 500.0, self._hst()
        )
        assert e_per_in == 500.0 and sci_per_e == 1.0

    def test_cps_divides_by_exptime(self):
        from astropy.io import fits

        e_per_in, sci_per_e, _ = inject_mod._chip_units(
            "ELECTRONS/S", fits.Header(), 500.0, self._hst()
        )
        assert e_per_in == 500.0
        assert sci_per_e == pytest.approx(1.0 / 500.0)

    def test_mjysr_mean_is_gain_free_and_flux_exact(self):
        pixar_sr = 2.0e-14
        adapter = instruments.get("nircam_sw")
        e_per_jy, sci_per_e, note = inject_mod._chip_units(
            "MJY/SR", self._jwst_header(pixar_sr=pixar_sr), 1000.0, adapter
        )
        # electrons_per_input x sci_per_electron == SB per Jy: 1/(PIXAR_SR x 1e6),
        # independent of the nominal gain and exposure time.
        assert e_per_jy * sci_per_e == pytest.approx(1.0 / (pixar_sr * 1e6))
        assert "gain-free" in note or "e_per_dn" in note

    def test_mjysr_missing_photometry_is_loud(self):
        from astropy.io import fits

        with pytest.raises(ValueError, match="PHOTMJSR and PIXAR_SR"):
            inject_mod._chip_units(
                "MJY/SR", fits.Header(), 1000.0, instruments.get("nircam_sw")
            )

    def test_mjysr_without_gain_is_loud(self):
        from dataclasses import replace

        gainless = replace(instruments.get("nircam_sw"), e_per_dn=None)
        with pytest.raises(ValueError, match="e_per_dn"):
            inject_mod._chip_units("MJY/SR", self._jwst_header(), 1000.0, gainless)

    def test_unknown_bunit_is_loud(self):
        from astropy.io import fits

        with pytest.raises(ValueError, match="unrecognised SCI BUNIT"):
            inject_mod._chip_units("COUNTS", fits.Header(), 500.0, self._hst())

    def test_cps_without_exptime_is_loud(self):
        from astropy.io import fits

        with pytest.raises(ValueError, match="EXPTIME"):
            inject_mod._chip_units("ELECTRONS/S", fits.Header(), 0.0, self._hst())


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


def _cal_hdul(exptime=1000.0, shape=(200, 200), photmjsr=0.5, pixar_sr=2.0e-14):
    """Synthetic JWST _cal-like MEF: MJy/sr SCI with photometry keywords."""
    from astropy.io import fits
    from astropy.wcs import WCS

    primary = fits.PrimaryHDU()
    primary.header["XPOSURE"] = exptime
    primary.header["TELESCOP"] = "JWST"
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [RA, DEC]
    wcs.wcs.crpix = [100.5, 100.5]
    wcs.wcs.cdelt = [-0.031 / 3600.0, 0.031 / 3600.0]
    hdr = wcs.to_header()
    hdr["BUNIT"] = "MJy/sr"
    hdr["PHOTMJSR"] = photmjsr
    hdr["PIXAR_SR"] = pixar_sr
    sci = np.full(shape, 0.02)
    err = np.full(shape, 0.005)
    dq = np.zeros(shape, dtype=np.int32)
    return fits.HDUList(
        [
            primary,
            fits.ImageHDU(data=sci, header=hdr, name="SCI", ver=1),
            fits.ImageHDU(data=err, header=hdr.copy(), name="ERR", ver=1),
            fits.ImageHDU(data=dq, header=hdr.copy(), name="DQ", ver=1),
        ]
    )


class TestInjectJwst:
    def _run(self, tmp_path, flux_jy=1.0e-6, pixar_sr=2.0e-14):
        pytest.importorskip("drizzle")
        exposure = tmp_path / "jw01727_cal.fits"
        if not exposure.exists():
            _cal_hdul(pixar_sr=pixar_sr).writeto(exposure)
        spec = _spec(
            tmp_path,
            input_data=_gaussian_input(flux=flux_jy),
            instrument="nircam_sw",
            inject_pixel_scale=0.015,
        )
        adapter = instruments.get(spec.instrument)
        work_dir = tmp_path / "work"
        work_dir.mkdir(exist_ok=True)
        return (
            inject_mod.inject_into_exposures([exposure], spec, adapter, work_dir),
            flux_jy,
            pixar_sr,
        )

    def test_mean_surface_brightness_added_is_flux_exact(self, tmp_path):
        from astropy.io import fits

        (paths, fragment), flux_jy, pixar_sr = self._run(tmp_path)
        injected = fits.getdata(paths[0], ("SCI", 1))
        # Sum(SB added) x pixel area = injected flux, gain-free:
        # sum_sb = flux_Jy / (PIXAR_SR x 1e6).
        added_sb = injected.sum() - 0.02 * 200 * 200
        assert added_sb == pytest.approx(
            flux_jy / (pixar_sr * 1e6), rel=0.02
        )
        assert fragment["input_units"] == "Jy"
        assert "e_per_dn" in (fragment["units_note"] or "")

    def test_err_grows_in_sb_units(self, tmp_path):
        from astropy.io import fits

        (paths, _), _, _ = self._run(tmp_path)
        err = fits.getdata(paths[0], ("ERR", 1))
        assert np.all(err >= 0.005 - 1e-12)
        assert err.max() > 0.005


class TestNoOverlap:
    def test_footprint_missing_all_chips_is_a_clean_noop(self, tmp_path):
        pytest.importorskip("drizzle")
        from astropy.io import fits

        exposure = tmp_path / "j8pu42vlq_flc.fits"
        _frame_hdul().writeto(exposure)
        spec = _spec(tmp_path, inject_position=(RA + 1.0, DEC))
        adapter = instruments.get(spec.instrument)
        work_dir = tmp_path / "work"
        work_dir.mkdir(exist_ok=True)
        paths, fragment = inject_mod.inject_into_exposures(
            [exposure], spec, adapter, work_dir
        )
        # Frames pass through untouched but stamped; fragment stays coherent.
        sci = fits.getdata(paths[0], ("SCI", 1))
        assert sci.sum() == pytest.approx(140.0 * 200 * 200)
        assert fragment["total_injected_e"] == 0.0
        assert fragment["units_note"] is None
        assert fragment["frames"][0]["chips"] == []


def _prepared_keck_frame(data, itime=0.181, coadds=60):
    from astropy.io import fits

    header = fits.Header()
    header["ITIME"] = itime
    header["COADDS"] = coadds
    header["BUNIT"] = "ELECTRONS"
    return fits.PrimaryHDU(data.astype(np.float32), header=header)


class TestInjectKeck:
    def _frames(self, tmp_path, shift=(5, -3)):
        rng = np.random.default_rng(0)
        base = rng.normal(0.0, 1.0, (200, 200))
        yy, xx = np.mgrid[0:200, 0:200]
        base += 500.0 * np.exp(-0.5 * ((yy - 80) ** 2 + (xx - 120) ** 2) / 4.0)
        rolled = np.roll(base, shift, axis=(0, 1))
        paths = []
        for i, data in enumerate((base, rolled)):
            path = tmp_path / f"n{i:04d}_prep.fits"
            if not path.exists():
                _prepared_keck_frame(data).writeto(path)
            paths.append(path)
        return paths

    def _inject(self, tmp_path, spec=None, **spec_extra):
        pytest.importorskip("drizzle")
        from autoreduce.inject import keck as keck_mod

        paths = self._frames(tmp_path)
        spec = spec or _spec(
            tmp_path, instrument="nirc2_narrow",
            inject_pixel_scale=0.005, **spec_extra,
        )
        adapter = instruments.get("nirc2_narrow")
        work_dir = tmp_path / "work"
        work_dir.mkdir(exist_ok=True)
        distortion = np.zeros((2, 200, 200))
        return (
            keck_mod.inject_into_prepared(paths, spec, adapter, work_dir, distortion),
            paths,
        )

    def test_electrons_added_scale_with_itime_coadds(self, tmp_path):
        from astropy.io import fits

        (new_paths, fragment), originals = self._inject(tmp_path)
        added = fits.getdata(new_paths[0]).astype(float) - fits.getdata(
            originals[0]
        ).astype(float)
        # flux 1000 e-/s x (0.181 x 60) s per frame.
        assert added.sum() == pytest.approx(1000.0 * 0.181 * 60, rel=0.02)
        assert fragment["total_injected_e"] == pytest.approx(
            sum(f["injected_e"] for f in fragment["frames"])
        )
        assert "offsets_to_reference" in fragment["placement"]

    def test_placement_follows_measured_offsets(self, tmp_path):
        from astropy.io import fits

        (new_paths, fragment), originals = self._inject(tmp_path)
        centres = [f["centre_yx"] for f in fragment["frames"]]
        # The injected centres differ by the frame offsets the pre-pass
        # measured — which for rolled frames is the roll itself.
        dy = centres[1][0] - centres[0][0]
        dx = centres[1][1] - centres[0][1]
        assert dy == pytest.approx(5.0, abs=0.2)
        assert dx == pytest.approx(-3.0, abs=0.2)
        # And the added light lands where the fragment says it does.
        added = fits.getdata(new_paths[0]).astype(float) - fits.getdata(
            originals[0]
        ).astype(float)
        yy, xx = np.mgrid[0:200, 0:200]
        cy = (added * yy).sum() / added.sum()
        cx = (added * xx).sum() / added.sum()
        assert cy == pytest.approx(centres[0][0], abs=0.5)
        assert cx == pytest.approx(centres[0][1], abs=0.5)

    def test_nan_bad_pixels_stay_nan(self, tmp_path):
        from astropy.io import fits

        paths = self._frames(tmp_path)
        data = fits.getdata(paths[0]).astype(float)
        data[100, 100] = np.nan
        _prepared_keck_frame(data).writeto(paths[0], overwrite=True)
        (new_paths, _), _ = self._inject(tmp_path)
        injected = fits.getdata(new_paths[0])
        assert np.isnan(injected[100, 100])

    def test_inject_psf_required(self, tmp_path):
        from autoreduce.inject import keck as keck_mod

        spec = _spec(tmp_path, instrument="nirc2_narrow", inject_pixel_scale=0.005)
        spec = spec.__class__(**{**spec.as_dict(), "inject_psf": None})
        with pytest.raises(ValueError, match="requires TargetSpec.inject_psf"):
            keck_mod.inject_into_prepared(
                [], spec, instruments.get("nirc2_narrow"), tmp_path, np.zeros((2, 2, 2))
            )


class TestPipelineGate:
    def test_alma_is_loud(self, tmp_path):
        from autoreduce.pipeline import reduce_target

        spec = _spec(tmp_path, instrument="alma")
        with pytest.raises(ValueError, match="phases 1-2b"):
            reduce_target(spec, tmp_path / "cache", tmp_path / "out")

    def test_jwst_and_keck_admitted_past_gate(self, tmp_path, monkeypatch):
        # The gate must not reject nircam/nirc2; acquisition is stubbed to
        # fail loudly AFTER the gate so no network is touched.
        from autoreduce import pipeline as pipeline_mod

        def boom(ctx):
            raise RuntimeError("gate passed; acquire reached")

        monkeypatch.setattr(pipeline_mod, "_acquire", boom)
        for instrument in ("nircam_lw", "nirc2_narrow"):
            spec = _spec(tmp_path, instrument=instrument)
            with pytest.raises(RuntimeError, match="gate passed"):
                pipeline_mod.reduce_target(
                    spec, tmp_path / "cache", tmp_path / "out"
                )
