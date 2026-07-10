import json
import sys

import numpy as np
import pytest

from autoreduce import instruments
from autoreduce.package import cosmic_rays as cr_mod
from autoreduce.package import frames as frames_mod
from autoreduce.target import TargetSpec

RA, DEC = 2.0, -0.1


def _frame_hdul(
    nchips=2,
    bunit="ELECTRONS",
    exptime=500.0,
    mdrizsky=40.0,
    sci_value=140.0,
    err_value=5.0,
    sip=False,
    chip2_far=False,
    shape=(200, 200),
):
    """Synthetic calibrated MEF: PRIMARY + (SCI, ERR, DQ) per chip, TAN WCS."""
    from astropy.io import fits
    from astropy.wcs import WCS

    primary = fits.PrimaryHDU()
    primary.header["ROOTNAME"] = "j8pu42vlq"
    primary.header["EXPTIME"] = exptime
    primary.header["EXPSTART"] = 52345.1
    primary.header["INSTRUME"] = "ACS"
    primary.header["TELESCOP"] = "HST"
    hdus = [primary]
    for chip in range(1, nchips + 1):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [50.0, 50.0] if (chip == 2 and chip2_far) else [RA, DEC]
        wcs.wcs.crpix = [100.5, 100.5]
        wcs.wcs.cdelt = [-0.05 / 3600.0, 0.05 / 3600.0]
        hdr = wcs.to_header()
        hdr["BUNIT"] = bunit
        hdr["MDRIZSKY"] = mdrizsky
        hdr["CCDCHIP"] = 3 - chip
        if sip:
            hdr["CTYPE1"] = "RA---TAN-SIP"
            hdr["CTYPE2"] = "DEC--TAN-SIP"
            hdr["A_ORDER"] = 2
            hdr["B_ORDER"] = 2
            hdr["A_2_0"] = 1.0e-8
            hdr["B_0_2"] = 1.0e-8
        sci = np.full(shape, sci_value)
        err = np.full(shape, err_value)
        dq = np.zeros(shape, dtype=np.int32)
        hdus += [
            fits.ImageHDU(data=sci, header=hdr, name="SCI", ver=chip),
            fits.ImageHDU(data=err, header=hdr.copy(), name="ERR", ver=chip),
            fits.ImageHDU(data=dq, header=hdr.copy(), name="DQ", ver=chip),
        ]
    return fits.HDUList(hdus)


def _write_exposure(tmp_path, name="j8pu42vlq_flc.fits", **kwargs):
    path = tmp_path / name
    _frame_hdul(**kwargs).writeto(path)
    return path


def _spec(**extra):
    return TargetSpec(
        name="t", ra=RA, dec=DEC, cutout_shape=(51, 51), frame_products=True, **extra
    )


@pytest.fixture
def no_stpsf(monkeypatch):
    """Tier-2b stubbed out: unit tests never run real STPSF (data files +
    ~10s/frame); the fallback records an unavailable outcome instead."""
    from autoreduce.psf import stpsf_model

    def unavailable(*args, **kwargs):
        raise ImportError("stpsf stubbed out in unit tests")

    monkeypatch.setattr(stpsf_model, "model_frame_psf", unavailable)


@pytest.fixture
def no_cr(monkeypatch):
    """deepCR replaced by an all-clear masker (torch never imported)."""
    monkeypatch.setattr(
        cr_mod, "masker_for", lambda key: lambda sci: np.zeros(sci.shape, dtype=bool)
    )


def _run(tmp_path, exposures, spec=None, driz_cr_run=True):
    return frames_mod.package_frame_products(
        exposures,
        spec or _spec(),
        instruments.get("acs_wfc"),
        tmp_path / "out",
        driz_cr_run=driz_cr_run,
    )


def _manifest(tmp_path):
    with open(tmp_path / "out" / "frames" / "manifest.json") as f:
        return json.load(f)


class TestFrameCutoutShape:
    def test_equal_scales_pass_through_odd(self):
        assert frames_mod.frame_cutout_shape((281, 281), 0.05, 0.05) == (281, 281)

    def test_coarser_native_scale_shrinks(self):
        # Same sky footprint at 0.128"/pix native from a 0.065"/pix mosaic.
        assert frames_mod.frame_cutout_shape((281, 281), 0.065, 0.128) == (143, 143)

    def test_even_result_is_bumped_odd(self):
        assert frames_mod.frame_cutout_shape((50, 50), 0.05, 0.05) == (51, 51)

    def test_bad_scales_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            frames_mod.frame_cutout_shape((51, 51), 0.05, 0.0)


class TestFrameProducts:
    def test_products_written_per_chip(self, no_cr, tmp_path):
        _run(tmp_path, [_write_exposure(tmp_path)])
        frames_dir = tmp_path / "out" / "frames"
        for chip in (1, 2):
            chip_dir = frames_dir / f"j8pu42vlq_chip{chip}"
            for product in ("data.fits", "noise_map.fits", "dq.fits", "cr_mask.fits"):
                assert (chip_dir / product).exists()
        manifest = _manifest(tmp_path)
        assert manifest["frame_cutout_shape"] == [51, 51]
        assert len(manifest["frames"]) == 2
        assert manifest["frames"][0]["source_file"] == "j8pu42vlq_flc.fits"
        assert manifest["frames"][0]["ccdchip"] == 2

    def test_fragment_counts_and_products(self, no_cr, tmp_path):
        fragment = _run(tmp_path, [_write_exposure(tmp_path)])
        assert fragment["n_exposures"] == 1
        assert fragment["n_chips_written"] == 2
        assert fragment["n_chips_skipped"] == 0
        assert fragment["data_units"] == "ELECTRONS/S"
        assert fragment["manifest"] == "frames/manifest.json"

    def test_units_electrons_divided_by_exptime(self, no_cr, tmp_path):
        from astropy.io import fits

        _run(tmp_path, [_write_exposure(tmp_path)])
        chip_dir = tmp_path / "out" / "frames" / "j8pu42vlq_chip1"
        with fits.open(chip_dir / "data.fits") as hdul:
            # (140 e- - 40 e- sky) / 500 s
            assert hdul[0].data[25, 25] == pytest.approx(0.2)
            assert hdul[0].header["BUNIT"] == "ELECTRONS/S"
        with fits.open(chip_dir / "noise_map.fits") as hdul:
            assert hdul[0].data[25, 25] == pytest.approx(5.0 / 500.0)
        entry = _manifest(tmp_path)["frames"][0]
        assert entry["unit_conversion"] == "SCI,ERR / EXPTIME"
        assert entry["sky_subtracted"] == pytest.approx(40.0)
        assert entry["sky_keyword"] == "MDRIZSKY"

    def test_cps_input_needs_no_conversion(self, no_cr, tmp_path):
        from astropy.io import fits

        exposure = _write_exposure(
            tmp_path, bunit="ELECTRONS/S", sci_value=0.25, err_value=0.01,
            mdrizsky=0.05,
        )
        _run(tmp_path, [exposure])
        chip_dir = tmp_path / "out" / "frames" / "j8pu42vlq_chip1"
        with fits.open(chip_dir / "data.fits") as hdul:
            assert hdul[0].data[25, 25] == pytest.approx(0.2)
        assert _manifest(tmp_path)["frames"][0]["unit_conversion"] == (
            "none (already e-/s)"
        )

    def test_unknown_bunit_is_loud(self, no_cr, tmp_path):
        exposure = _write_exposure(tmp_path, bunit="COUNTS")
        with pytest.raises(ValueError, match="BUNIT"):
            _run(tmp_path, [exposure])

    def test_structured_cr_trail_masked_by_noise_dq_raw(self, no_cr, tmp_path):
        from astropy.io import fits

        path = tmp_path / "j8pu42vlq_flc.fits"
        hdul = _frame_hdul()
        # A contiguous driz_cr trail through the cutout — the mosaic's
        # isolated-bad-pixel policy would refuse this; frames must not.
        hdul["DQ", 1].data[95:106, 100] = 4096
        hdul.writeto(path)
        _run(tmp_path, [path])
        chip_dir = tmp_path / "out" / "frames" / "j8pu42vlq_chip1"
        with fits.open(chip_dir / "dq.fits") as f:
            dq = f[0].data
        with fits.open(chip_dir / "noise_map.fits") as f:
            noise = f[0].data
        with fits.open(chip_dir / "data.fits") as f:
            data = f[0].data
        trail = dq == 4096
        assert trail.sum() == 11
        assert noise[trail] == pytest.approx(1.0e8)
        assert (data[trail] == 0.0).all()
        assert _manifest(tmp_path)["frames"][0]["n_masked_pixels"] == 11

    def test_cr_mask_is_ord_in_and_written(self, monkeypatch, tmp_path):
        from astropy.io import fits

        def fake_masker(key):
            def mask(sci):
                out = np.zeros(sci.shape, dtype=bool)
                out[10:13, 10:13] = True
                return out

            return mask

        monkeypatch.setattr(cr_mod, "masker_for", fake_masker)
        _run(tmp_path, [_write_exposure(tmp_path)])
        chip_dir = tmp_path / "out" / "frames" / "j8pu42vlq_chip1"
        with fits.open(chip_dir / "cr_mask.fits") as f:
            assert f[0].data.sum() == 9
        with fits.open(chip_dir / "noise_map.fits") as f:
            assert f[0].data[10:13, 10:13] == pytest.approx(1.0e8)
        entry = _manifest(tmp_path)["frames"][0]
        assert entry["n_cr_pixels"] == 9
        assert entry["n_masked_pixels"] == 9

    def test_partial_overlap_masked_and_recorded(self, no_cr, tmp_path):
        from astropy.io import fits
        from astropy.wcs import WCS

        path = _write_exposure(tmp_path, nchips=1)
        # Aim the cutout at a pixel near the chip edge so it hangs off.
        with fits.open(path) as hdul:
            ra, dec = WCS(hdul["SCI", 1].header).pixel_to_world_values(5.0, 100.0)
        spec = TargetSpec(
            name="t", ra=float(ra), dec=float(dec), cutout_shape=(51, 51),
            frame_products=True,
        )
        _run(tmp_path, [path], spec=spec)
        entry = _manifest(tmp_path)["frames"][0]
        assert entry["offchip_fraction"] > 0.0
        chip_dir = tmp_path / "out" / "frames" / "j8pu42vlq_chip1"
        with fits.open(chip_dir / "noise_map.fits") as f:
            assert float(f[0].data.max()) == pytest.approx(1.0e8)

    def test_chip_without_overlap_skipped(self, no_cr, tmp_path):
        fragment = _run(tmp_path, [_write_exposure(tmp_path, chip2_far=True)])
        assert fragment["n_chips_written"] == 1
        assert fragment["n_chips_skipped"] == 1
        manifest = _manifest(tmp_path)
        assert manifest["skipped_chips"][0]["chip"] == 2
        assert not (tmp_path / "out" / "frames" / "j8pu42vlq_chip2").exists()

    def test_zero_chips_is_loud(self, no_cr, tmp_path):
        path = _write_exposure(tmp_path, nchips=1, chip2_far=False)
        spec = TargetSpec(
            name="t", ra=180.0, dec=50.0, cutout_shape=(51, 51), frame_products=True
        )
        with pytest.raises(ValueError, match="no chips"):
            _run(tmp_path, [path], spec=spec)

    def test_sip_survives_to_header(self, no_cr, tmp_path):
        from astropy.io import fits

        _run(tmp_path, [_write_exposure(tmp_path, nchips=1, sip=True)])
        chip_dir = tmp_path / "out" / "frames" / "j8pu42vlq_chip1"
        with fits.open(chip_dir / "data.fits") as f:
            assert f[0].header["A_ORDER"] == 2

    def test_lookup_distortion_keywords_stripped(self):
        # Real _flc SCI headers carry CPDIS/DP (NPOL) and D2IM lookup-table
        # distortion; astropy raises if WCS parses them without the open
        # HDUList, so the SIP-only cutout header must strip them (found by
        # the slacs0008 validation run). Synthetic files cannot carry fake
        # lookup keywords through the full-WCS path (no WCSDVARR extensions
        # to resolve), so the helper is tested directly.
        from astropy.wcs import WCS

        hdr = _frame_hdul(nchips=1)["SCI", 1].header
        hdr["CPDIS1"] = "Lookup"
        hdr["CPDIS2"] = "Lookup"
        for axis in (1, 2):
            hdr[f"DP{axis}.EXTVER"] = float(axis)
            hdr[f"DP{axis}.NAXES"] = 2.0
            hdr[f"DP{axis}.AXIS.1"] = 1.0
            hdr[f"DP{axis}.AXIS.2"] = 2.0
        with pytest.raises(ValueError, match="HDUList is required"):
            WCS(hdr, naxis=2)  # the crash the helper exists to prevent
        stripped = frames_mod._sip_only_header(hdr)
        for key in stripped:
            assert not str(key).startswith(("CPDIS", "CPERR", "DP1", "DP2", "D2IM"))
        assert WCS(stripped, naxis=2).has_celestial

    def test_target_pixel_roundtrip(self, no_cr, tmp_path):
        from astropy.io import fits
        from astropy.wcs import WCS

        _run(tmp_path, [_write_exposure(tmp_path, nchips=1)])
        entry = _manifest(tmp_path)["frames"][0]
        chip_dir = tmp_path / "out" / "frames" / "j8pu42vlq_chip1"
        with fits.open(chip_dir / "data.fits") as f:
            x, y = WCS(f[0].header).world_to_pixel_values(RA, DEC)
        assert float(x) == pytest.approx(entry["target_pixel"][0], abs=0.01)
        assert float(y) == pytest.approx(entry["target_pixel"][1], abs=0.01)
        # Odd shape: the target sits on the centre pixel.
        assert entry["target_pixel"][0] == pytest.approx(25.0, abs=0.51)

    def test_single_exposure_caveat_recorded(self, no_cr, tmp_path):
        _run(tmp_path, [_write_exposure(tmp_path)], driz_cr_run=False)
        manifest = _manifest(tmp_path)
        assert manifest["driz_cr_run"] is False
        assert "driz_cr_note" in manifest["dq_semantics"]

    def test_rerun_clears_stale_frame_dirs(self, no_cr, tmp_path):
        exposure = _write_exposure(tmp_path)
        _run(tmp_path, [exposure])
        stale = tmp_path / "out" / "frames" / "stale_chip9"
        stale.mkdir()
        _run(tmp_path, [exposure])
        assert not stale.exists()


class TestCosmicRays:
    def test_missing_deepcr_is_loud(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "deepCR", None)
        with pytest.raises(ImportError, match=r"autoreduce\[frames\]"):
            cr_mod.masker_for("acs_wfc")

    def test_unknown_instrument_is_loud(self):
        with pytest.raises(KeyError, match="no deepCR model"):
            cr_mod.masker_for("nirc2")

    def test_cr_method_records(self):
        assert cr_mod.cr_method_record("acs_wfc")["method"] == "deepCR"
        assert cr_mod.cr_method_record("wfc3_uvis")["threshold"] == 0.5
        assert "ramp-fitting" in cr_mod.cr_method_record("wfc3_ir")["method"]
        with pytest.raises(KeyError, match="CR method"):
            cr_mod.cr_method_record("nircam")


class TestPipelinePlumbing:
    def _run_pipeline(self, monkeypatch, tmp_path, frame_products):
        from autoreduce import pipeline as pipeline_mod

        order = []

        def stage(name, result=None):
            def _stage(ctx, *args):
                order.append(name)
                return result

            return _stage

        monkeypatch.setattr(pipeline_mod, "_acquire", stage("acquire"))
        monkeypatch.setattr(pipeline_mod, "_align", stage("align"))
        monkeypatch.setattr(pipeline_mod, "_ground_prepare", stage("ground"))

        def fake_combine(ctx):
            order.append("combine")
            ctx.record["drizzle"] = {"single_exposure_branch": False}
            return np.ones((5, 5)), {}, np.ones((5, 5)), 100.0

        monkeypatch.setattr(pipeline_mod, "_combine", fake_combine)
        monkeypatch.setattr(
            pipeline_mod, "_noise", lambda ctx, *a: (order.append("noise"), np.ones((5, 5)))[1]
        )
        monkeypatch.setattr(
            pipeline_mod,
            "_psf",
            lambda ctx, *a: (order.append("psf"), (np.ones((3, 3)), np.ones((5, 5))))[1],
        )

        def fake_package(ctx, *args):
            order.append("package")
            ctx.record["package"] = {"products": ["data.fits"]}

        monkeypatch.setattr(pipeline_mod, "_package", fake_package)
        monkeypatch.setattr(
            pipeline_mod, "_evict", lambda *a, **k: order.append("evict")
        )

        def fake_frames(exposures, spec, adapter, out_dir, driz_cr_run, source_note=None):
            order.append("frames")
            return {"n_chips_written": 2, "driz_cr_run": driz_cr_run}

        monkeypatch.setattr(
            pipeline_mod.frames_mod, "package_frame_products", fake_frames
        )
        spec = TargetSpec(name="t", ra=RA, dec=DEC, frame_products=frame_products)
        record = pipeline_mod.reduce_target(
            spec, cache_root=tmp_path / "cache", output_root=tmp_path / "out"
        )
        return order, record

    def test_flag_off_never_packages_frames(self, monkeypatch, tmp_path):
        order, record = self._run_pipeline(monkeypatch, tmp_path, frame_products=False)
        assert "frames" not in order
        assert "frames" not in record

    def test_flag_on_runs_after_package_before_evict(self, monkeypatch, tmp_path):
        order, record = self._run_pipeline(monkeypatch, tmp_path, frame_products=True)
        assert order.index("frames") == order.index("package") + 1
        assert order.index("frames") < order.index("evict")
        assert record["frames"]["n_chips_written"] == 2
        assert record["frames"]["driz_cr_run"] is True
        assert "frames/manifest.json" in record["package"]["products"]

    def test_non_hst_flag_is_loud_before_any_stage(self, tmp_path):
        from autoreduce import pipeline as pipeline_mod

        spec = TargetSpec(name="t", ra=RA, dec=DEC, instrument="nirc2_narrow",
                          frame_products=True)
        with pytest.raises(ValueError, match="HST and JWST only"):
            pipeline_mod.reduce_target(
                spec, cache_root=tmp_path / "cache", output_root=tmp_path / "out"
            )
        assert not (tmp_path / "cache").exists()


class TestSpecField:
    def test_default_off(self):
        assert TargetSpec(name="t", ra=RA, dec=DEC).frame_products is False

    def test_yaml_round_trip(self, tmp_path):
        path = tmp_path / "spec.yaml"
        path.write_text(
            "name: t\nra: 2.0\ndec: -0.1\nframe_products: true\n"
        )
        spec = TargetSpec.from_yaml(path)
        assert spec.frame_products is True
        assert spec.as_dict()["frame_products"] is True


def _paint_blob(hdul, chip, ra, dec, sigma=3.0, amp=200.0, offset_px=(0.0, 0.0)):
    """Add a Gaussian blob at sky position (ra, dec) via the chip's own WCS,
    optionally offset in pixels to simulate data misregistered vs its WCS."""
    from astropy.wcs import WCS

    sci = hdul["SCI", chip].data
    x, y = WCS(hdul["SCI", chip].header).world_to_pixel_values(ra, dec)
    y = float(y) + offset_px[0]
    x = float(x) + offset_px[1]
    yy, xx = np.mgrid[0 : sci.shape[0], 0 : sci.shape[1]]
    sci += amp * np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma**2)))


class TestRegistration:
    def _dithered_pair(self, tmp_path, offset_px=(0.0, 0.0)):
        """Two 1-chip exposures of the same sky blob; the second is dithered
        by (3, 2) whole pixels, which its WCS records exactly. offset_px
        additionally shifts the second frame's *data* relative to what its
        WCS claims (a registration error)."""
        paths = []
        for name, crpix_shift, off in (
            ("j8aaaaaaq_flc.fits", (0.0, 0.0), (0.0, 0.0)),
            ("j8bbbbbbq_flc.fits", (3.0, 2.0), offset_px),
        ):
            # sci == MDRIZSKY: zero background after sky subtraction, like a
            # real sky-subtracted frame; the noise floor is REQUIRED — phase
            # correlation whitens the spectrum, and a noise-free synthetic
            # image turns the empty high frequencies into ringing sidelobes
            # no real frame has.
            hdul = _frame_hdul(nchips=1, sci_value=40.0)
            hdul[0].header["ROOTNAME"] = name.split("_")[0]
            hdr = hdul["SCI", 1].header
            hdr["CRPIX1"] += crpix_shift[1]
            hdr["CRPIX2"] += crpix_shift[0]
            # A constellation, not one smooth blob: phase correlation needs
            # coherent power across many Fourier modes, which real lens
            # cutouts (galaxy + arcs + neighbours) have and a single
            # Gaussian does not.
            arcsec = 1.0 / 3600.0
            for dra, ddec, sigma, amp in (
                (0.0, 0.0, 3.0, 300.0),
                (0.4 * arcsec, 0.3 * arcsec, 1.5, 250.0),
                (-0.5 * arcsec, 0.2 * arcsec, 1.0, 200.0),
                (0.2 * arcsec, -0.6 * arcsec, 2.0, 150.0),
                (-0.3 * arcsec, -0.4 * arcsec, 1.2, 220.0),
            ):
                _paint_blob(
                    hdul, 1, RA + dra, DEC + ddec, sigma=sigma, amp=amp,
                    offset_px=off,
                )
            # Independent noise per frame — a shared field would itself
            # correlate and drag the measurement to the dither offset.
            rng = np.random.default_rng(len(paths))
            hdul["SCI", 1].data += rng.normal(0.0, 2.0, hdul["SCI", 1].data.shape)
            path = tmp_path / name
            hdul.writeto(path)
            paths.append(path)
        return paths

    def test_registered_frames_have_near_zero_residual(self, no_cr, tmp_path):
        _run(tmp_path, self._dithered_pair(tmp_path))
        manifest = _manifest(tmp_path)
        ref, other = manifest["frames"]
        assert ref["registration"]["reference"] is None
        assert other["registration"]["reference"] == ref["dir"]
        assert abs(other["registration"]["residual_dy_px"]) < 0.05
        assert abs(other["registration"]["residual_dx_px"]) < 0.05
        assert manifest["max_registration_residual_px"] < 0.05

    def test_misregistered_frame_residual_is_recovered(self, no_cr, tmp_path):
        # Second frame's data sits 2 px off in y from what its WCS claims —
        # a genuine registration error the manifest must report. (An integer
        # injection: the estimator's sub-pixel fidelity is ~0.1-0.3 px by
        # design, as the manifest's registration_note records.)
        _run(tmp_path, self._dithered_pair(tmp_path, offset_px=(2.0, 0.0)))
        manifest = _manifest(tmp_path)
        other = manifest["frames"][1]
        assert abs(abs(other["registration"]["residual_dy_px"]) - 2.0) < 0.15
        assert abs(other["registration"]["residual_dx_px"]) < 0.15
        assert manifest["max_registration_residual_px"] == pytest.approx(
            np.hypot(
                other["registration"]["residual_dy_px"],
                other["registration"]["residual_dx_px"],
            )
        )

    def test_registration_notice_printed(self, no_cr, tmp_path, capsys):
        # Deliberately loud during use (user request, issue #19) — pin it so
        # it isn't silently dropped until deliberately retired.
        _run(tmp_path, self._dithered_pair(tmp_path))
        out = capsys.readouterr().out
        assert "[frames] inter-exposure registration" in out
        assert "ABSOLUTE catalog alignment" in out
        assert "max relative residual" in out

    def test_header_solution_recorded(self, no_cr, tmp_path):
        path = tmp_path / "j8ccccccq_flc.fits"
        hdul = _frame_hdul(nchips=1)
        hdr = hdul["SCI", 1].header
        hdr["WCSNAME"] = "IDC_x-FIT_REL_GSC242"
        hdr["WCSTYPE"] = "undistorted a posteriori solution relatively aligned to GSC242"
        hdr["RMS_RA"] = 44.5
        hdr["RMS_DEC"] = 42.0
        hdr["NMATCHES"] = 30
        _paint_blob(hdul, 1, RA, DEC)
        hdul.writeto(path)
        _run(tmp_path, [path])
        reg = _manifest(tmp_path)["frames"][0]["registration"]
        assert reg["wcsname"] == "IDC_x-FIT_REL_GSC242"
        assert reg["rms_ra_mas"] == pytest.approx(44.5)
        assert reg["nmatches"] == 30
        assert reg["residual_dy_px"] == 0.0 and reg["reference"] is None


class TestFramePsf:
    def test_insufficient_stars_is_recorded_not_fatal(self, no_cr, tmp_path, capsys):
        # A starless frame ships its data products, records the PSF outcome,
        # and says so loudly — it must not hard-stop the reduction.
        fragment = _run(tmp_path, [_write_exposure(tmp_path, nchips=1)])
        entry = _manifest(tmp_path)["frames"][0]
        assert entry["psf"]["method"] == "none"
        assert "usable stars" in entry["psf"]["reason"]
        chip_dir = tmp_path / "out" / "frames" / "j8pu42vlq_chip1"
        assert not (chip_dir / "psf.fits").exists()
        assert fragment["n_frames_with_psf"] == 0
        assert "per-frame ePSF NOT viable" in capsys.readouterr().out

    def test_peak_max_native_units(self):
        from autoreduce.psf.frame_epsf import _native_peak_max
        from autoreduce.psf.stars import StarSelection

        adapter = instruments.get("acs_wfc")
        selection = StarSelection()
        cap = selection.saturation_fraction * adapter.saturation_dn
        assert _native_peak_max("ELECTRONS", 522.0, adapter, selection) == (
            pytest.approx(cap)
        )
        assert _native_peak_max("ELECTRONS/S", 500.0, adapter, selection) == (
            pytest.approx(cap / 500.0)
        )
        with pytest.raises(ValueError, match="EXPTIME"):
            _native_peak_max("ELECTRONS/S", 0.0, adapter, selection)
        with pytest.raises(ValueError, match="BUNIT"):
            _native_peak_max("COUNTS", 522.0, adapter, selection)

    def test_starry_frame_builds_native_epsf(self, no_cr, tmp_path):
        # A frame with a usable star field gets psf.fits/psf_full.fits on
        # its native pixel grid, with tier-1 diagnostics in the manifest.
        rng = np.random.default_rng(3)
        hdul = _frame_hdul(nchips=1, shape=(400, 400))
        hdr = hdul["SCI", 1].header
        # Star ring outside the target-exclusion radius (50 px), inside the
        # edge margin (46 px), separated by > 25 px.
        sci = hdul["SCI", 1].data
        yy, xx = np.mgrid[0 : sci.shape[0], 0 : sci.shape[1]]
        n_stars = 12
        for k in range(n_stars):
            ang = 2.0 * np.pi * k / n_stars
            r = 90.0 + 20.0 * (k % 2)
            cy, cx = 199.5 + r * np.sin(ang), 199.5 + r * np.cos(ang)
            sci += 5000.0 * np.exp(
                -(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 1.5**2))
            )
        sci += rng.normal(0.0, 2.0, sci.shape)
        path = tmp_path / "j8pu42vlq_flc.fits"
        hdul.writeto(path)

        fragment = _run(tmp_path, [path])
        entry = _manifest(tmp_path)["frames"][0]
        assert entry["psf"]["method"] == "epsf-frame-tier1"
        assert entry["psf"]["n_stars_used"] >= 8
        assert fragment["n_frames_with_psf"] == 1

        from astropy.io import fits

        chip_dir = tmp_path / "out" / "frames" / "j8pu42vlq_chip1"
        psf = fits.getdata(chip_dir / "psf.fits").astype(float)
        assert psf.shape == (21, 21)
        assert psf.sum() == pytest.approx(1.0, abs=1e-4)
        assert (tmp_path / "out" / "frames" / "j8pu42vlq_chip1" / "psf_full.fits").exists()

    def test_dq_flagged_spike_is_patched_away(self, no_cr, tmp_path):
        # A CR-like flagged spike is local-median patched out of the
        # estimator's working image — it must not enter the star list, and
        # the patch is recorded in the diagnostics.
        from autoreduce.psf.frame_epsf import build_frame_epsf

        hdul = _frame_hdul(nchips=1, shape=(400, 400))
        sci = hdul["SCI", 1].data
        cy, cx = 289, 199
        sci[cy - 1 : cy + 2, cx - 1 : cx + 2] += 30000.0
        hdul["DQ", 1].data[cy - 1 : cy + 2, cx - 1 : cx + 2] = 4096
        psf, psf_full, diag = build_frame_epsf(
            hdul, 1, _spec(), instruments.get("acs_wfc")
        )
        assert psf is None
        assert diag["method"] == "none"
        assert diag["n_candidates"] == 0
        assert diag["n_patched_pixels"] == 9
        assert diag["cr_screen"] == "DQ-patched"


class TestPsfFromFrames:
    def _starry_exposure(self, tmp_path):
        hdul = _frame_hdul(nchips=1, shape=(400, 400))
        sci = hdul["SCI", 1].data
        yy, xx = np.mgrid[0 : sci.shape[0], 0 : sci.shape[1]]
        for k in range(12):
            ang = 2.0 * np.pi * k / 12
            r = 90.0 + 20.0 * (k % 2)
            cy, cx = 199.5 + r * np.sin(ang), 199.5 + r * np.cos(ang)
            sci += 5000.0 * np.exp(
                -(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 1.5**2))
            )
        sci += np.random.default_rng(3).normal(0.0, 2.0, sci.shape)
        # The default WCS puts the target at (99.5, 99.5) — on-chip, as the
        # combination requires.
        path = tmp_path / "j8pu42vlq_flc.fits"
        hdul.writeto(path)
        return path, hdul["SCI", 1].header

    def test_drop_convolve_preserves_flux_and_widens(self):
        from autoreduce.psf.frame_combine import _drop_convolve

        kernel = np.zeros((31, 31))
        yy, xx = np.mgrid[0:31, 0:31]
        kernel = np.exp(-(((yy - 15) ** 2 + (xx - 15) ** 2) / (2.0 * 1.5**2)))
        kernel /= kernel.sum()
        out = _drop_convolve(kernel, pixfrac=0.8)
        assert out.sum() == pytest.approx(1.0, abs=1e-8)
        # Second moment grows by the box variance pixfrac^2/12 per axis.
        def var_y(k):
            return float((k * (yy - 15) ** 2).sum() / k.sum())
        assert var_y(out) - var_y(kernel) == pytest.approx(0.8**2 / 12.0, rel=0.05)

    def test_local_jacobian_scale(self):
        from astropy.wcs import WCS
        from autoreduce.psf.frame_combine import _local_jacobian

        def tan_wcs(scale_deg):
            w = WCS(naxis=2)
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            w.wcs.crval = [RA, DEC]
            w.wcs.crpix = [100.5, 100.5]
            w.wcs.cdelt = [-scale_deg, scale_deg]
            return w

        frame = tan_wcs(0.05 / 3600.0)
        mosaic = tan_wcs(0.10 / 3600.0)  # 2x coarser
        jac = _local_jacobian(frame, (99.5, 99.5), mosaic)
        assert jac == pytest.approx(0.5 * np.eye(2), abs=1e-6)

    def test_combined_psf_identity_geometry(self, no_cr, tmp_path):
        # Mosaic grid == frame grid: the combination reduces to the (drop-
        # convolved) frame ePSF — normalized, centred, method recorded.
        from autoreduce.psf.frame_combine import combined_mosaic_psf

        path, hdr = self._starry_exposure(tmp_path)
        psf, psf_full, diag = combined_mosaic_psf(
            [path], _spec(), instruments.get("acs_wfc"), hdr
        )
        assert diag["method"] == "epsf-frames-combined"
        assert diag["n_frames_combined"] == 1
        assert diag["frames"][0]["weight_exptime"] == pytest.approx(500.0)
        assert psf.shape == (21, 21)
        assert psf.sum() == pytest.approx(1.0, abs=1e-4)
        peak = np.unravel_index(np.argmax(psf), psf.shape)
        assert peak == (10, 10)

    def test_no_viable_frames_is_loud(self, no_cr, tmp_path):
        from autoreduce.psf.frame_combine import combined_mosaic_psf

        path = _write_exposure(tmp_path, nchips=1)  # starless
        from astropy.io import fits

        hdr = fits.open(path)["SCI", 1].header
        with pytest.raises(ValueError, match="no frame yields"):
            combined_mosaic_psf([path], _spec(), instruments.get("acs_wfc"), hdr)

    def test_non_hst_guard_covers_psf_from_frames(self, tmp_path):
        from autoreduce import pipeline as pipeline_mod

        spec = TargetSpec(
            name="t", ra=RA, dec=DEC, instrument="nirc2_narrow", psf_from_frames=True
        )
        with pytest.raises(ValueError, match="HST and JWST only"):
            pipeline_mod.reduce_target(
                spec, cache_root=tmp_path / "c", output_root=tmp_path / "o"
            )

    def test_spec_default_off(self):
        assert TargetSpec(name="t", ra=RA, dec=DEC).psf_from_frames is False


def _jwst_hdul(bunit="MJy/sr", xposure=515.4, bkglevel=0.21, shape=(200, 200)):
    """Synthetic JWST _cal/_crf-shaped file: no ROOTNAME, XPOSURE exposure
    time, single SCI/ERR/DQ (EXTVER 1), MJy/sr units, skymatch BKGLEVEL."""
    from astropy.io import fits
    from astropy.wcs import WCS

    primary = fits.PrimaryHDU()
    primary.header["XPOSURE"] = xposure
    primary.header["EFFEXPTM"] = xposure
    primary.header["INSTRUME"] = "NIRCAM"
    primary.header["TELESCOP"] = "JWST"
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [RA, DEC]
    wcs.wcs.crpix = [100.5, 100.5]
    wcs.wcs.cdelt = [-0.063 / 3600.0, 0.063 / 3600.0]
    hdr = wcs.to_header()
    hdr["BUNIT"] = bunit
    hdr["BKGLEVEL"] = bkglevel
    sci = np.full(shape, 0.75)
    err = np.full(shape, 0.02)
    dq = np.zeros(shape, dtype=np.int32)
    return fits.HDUList(
        [
            primary,
            fits.ImageHDU(data=sci, header=hdr, name="SCI", ver=1),
            fits.ImageHDU(data=err, header=hdr.copy(), name="ERR", ver=1),
            fits.ImageHDU(data=dq, header=hdr.copy(), name="DQ", ver=1),
        ]
    )


@pytest.mark.usefixtures("no_stpsf")
class TestJwstFrames:
    def _run_jwst(self, tmp_path, hdul, name="jw01727043001_02101_00001_nrcb1_crf.fits",
                  driz_cr_run=True):
        path = tmp_path / name
        hdul.writeto(path)
        spec = TargetSpec(
            name="t", ra=RA, dec=DEC, cutout_shape=(51, 51),
            instrument="nircam_lw", frame_products=True,
        )
        return frames_mod.package_frame_products(
            [path], spec, instruments.get("nircam_lw"), tmp_path / "out",
            driz_cr_run=driz_cr_run,
            source_note="image3 _crf products (outlier-flagged, tweakreg WCS)",
        )

    def test_native_units_kept_and_recorded(self, no_cr, tmp_path):
        from astropy.io import fits

        fragment = self._run_jwst(tmp_path, _jwst_hdul())
        assert fragment["data_units"] == "MJy/sr"
        manifest = _manifest(tmp_path)
        assert manifest["data_units"] == "MJy/sr"
        entry = manifest["frames"][0]
        assert entry["unit_conversion"] == "none (native MJy/sr)"
        # rootname derived from the filename stem (no ROOTNAME keyword).
        assert entry["dir"] == "jw01727043001_02101_00001_nrcb1_chip1"
        chip_dir = tmp_path / "out" / "frames" / entry["dir"]
        with fits.open(chip_dir / "data.fits") as h:
            assert h[0].header["BUNIT"] == "MJy/sr"
            # 0.75 MJy/sr − 0.21 BKGLEVEL, no exptime division
            assert h[0].data[20, 20] == pytest.approx(0.54)
        assert entry["sky_subtracted"] == pytest.approx(0.21)
        assert entry["sky_keyword"] == "BKGLEVEL"
        assert entry["exptime"] == pytest.approx(515.4)

    def test_dq_policy_do_not_use_only(self, no_cr, tmp_path):
        from astropy.io import fits

        hdul = _jwst_hdul()
        # JUMP_DET (4) alone = good data (CR removed at ramp level);
        # DO_NOT_USE (1, e.g. set by image3 outlier_detection) = bad.
        hdul["DQ", 1].data[95, 95] = 4
        hdul["DQ", 1].data[105, 105] = 1 | 4
        fragment = self._run_jwst(tmp_path, hdul)
        entry = _manifest(tmp_path)["frames"][0]
        chip_dir = tmp_path / "out" / "frames" / entry["dir"]
        noise = fits.getdata(chip_dir / "noise_map.fits").astype(float)
        dq = fits.getdata(chip_dir / "dq.fits")
        jump_only = np.argwhere(dq == 4)
        do_not_use = np.argwhere((dq & 1) != 0)
        assert len(jump_only) == 1 and len(do_not_use) == 1
        y, x = jump_only[0]
        assert noise[y, x] < 1.0  # good pixel keeps its ERR
        y, x = do_not_use[0]
        assert noise[y, x] == pytest.approx(1.0e8)
        assert entry["n_masked_pixels"] == 1
        semantics = _manifest(tmp_path)["dq_semantics"]
        assert "DO_NOT_USE" in semantics["1"]
        assert "good data" in semantics["4"]

    def test_cr_method_and_source_recorded(self, no_cr, tmp_path):
        fragment = self._run_jwst(tmp_path, _jwst_hdul())
        assert "ramp-jump" in fragment["cr_method"]["method"]
        assert fragment["cr_method"]["model"] is None
        manifest = _manifest(tmp_path)
        assert manifest["source"].startswith("image3 _crf")
        assert manifest["version"] == 2

    def test_mjy_sr_peak_max_is_none(self):
        from autoreduce.psf.frame_epsf import _native_peak_max
        from autoreduce.psf.stars import StarSelection

        assert _native_peak_max(
            "MJy/sr", 515.4, instruments.get("nircam_lw"), StarSelection()
        ) is None

    def test_mixed_units_is_loud(self, no_cr, tmp_path):
        from astropy.io import fits

        p1 = tmp_path / "jw0001_nrcb1_crf.fits"
        _jwst_hdul().writeto(p1)
        hdul2 = _jwst_hdul(bunit="ELECTRONS")
        p2 = tmp_path / "jw0002_nrcb1_crf.fits"
        hdul2.writeto(p2)
        spec = TargetSpec(name="t", ra=RA, dec=DEC, cutout_shape=(51, 51),
                          instrument="nircam_lw", frame_products=True)
        with pytest.raises(ValueError, match="heterogeneous"):
            frames_mod.package_frame_products(
                [p1, p2], spec, instruments.get("nircam_lw"),
                tmp_path / "out", driz_cr_run=True,
            )


class TestRegistrationReliability:
    def test_heavily_masked_pairs_flagged_and_excluded(self, no_cr, tmp_path, capsys):
        # Both frames' cutouts hang mostly off-chip (JWST-style edge
        # dithers): residuals are mask-geometry artifacts, so they must be
        # flagged unreliable and the headline max must be an honest None.
        from astropy.io import fits
        from astropy.wcs import WCS

        paths = []
        for name, crpix_shift in (
            ("j8aaaaaaq_flc.fits", (0.0, 0.0)),
            ("j8bbbbbbq_flc.fits", (3.0, 2.0)),
        ):
            hdul = _frame_hdul(nchips=1, sci_value=40.0)
            hdul[0].header["ROOTNAME"] = name.split("_")[0]
            hdr = hdul["SCI", 1].header
            hdr["CRPIX1"] += crpix_shift[1]
            hdr["CRPIX2"] += crpix_shift[0]
            _paint_blob(hdul, 1, RA, DEC)
            rng = np.random.default_rng(len(paths))
            hdul["SCI", 1].data += rng.normal(0.0, 2.0, hdul["SCI", 1].data.shape)
            paths.append(tmp_path / name)
            hdul.writeto(paths[-1])
        # Aim the cutout near the chip corner: most of it is off-chip.
        with fits.open(paths[0]) as hdul:
            ra, dec = WCS(hdul["SCI", 1].header).pixel_to_world_values(3.0, 3.0)
        spec = TargetSpec(
            name="t", ra=float(ra), dec=float(dec), cutout_shape=(51, 51),
            frame_products=True,
        )
        _run(tmp_path, paths, spec=spec)
        manifest = _manifest(tmp_path)
        assert manifest["max_registration_residual_px"] is None
        assert all(
            e["registration"]["residual_reliable"] is False
            for e in manifest["frames"]
        )
        assert "UNMEASURED" in capsys.readouterr().out

    def test_clean_pair_is_reliable(self, no_cr, tmp_path):
        # The existing well-covered dithered pair: reliable flags set, and
        # the headline max is numeric.
        pair = TestRegistration()._dithered_pair(tmp_path)
        _run(tmp_path, pair)
        manifest = _manifest(tmp_path)
        assert manifest["max_registration_residual_px"] is not None
        assert all(
            e["registration"]["residual_reliable"] is True
            for e in manifest["frames"]
        )


class TestStpsfTier2b:
    def _starless_jwst(self, tmp_path):
        hdul = _jwst_hdul()
        hdul[0].header["DETECTOR"] = "NRCB1"
        path = tmp_path / "jw0009_nrcb1_crf.fits"
        hdul.writeto(path)
        return path

    def test_jwst_fallback_to_stpsf_model(self, no_cr, monkeypatch, tmp_path):
        from autoreduce.psf import frame_epsf, stpsf_model

        kernel21 = np.zeros((21, 21)); kernel21[10, 10] = 1.0
        kernel61 = np.zeros((61, 61)); kernel61[30, 30] = 1.0

        def fake_model(primary, target_xy, spec, adapter, det_shape=None):
            assert primary["DETECTOR"] == "NRCB1"
            return kernel21, kernel61, {
                "method": "stpsf-tier2b", "detector": "NRCB1",
                "detector_position": list(target_xy), "caveat": "model-PSF fallback",
            }

        monkeypatch.setattr(stpsf_model, "model_frame_psf", fake_model)
        from astropy.io import fits

        path = self._starless_jwst(tmp_path)
        with fits.open(path) as hdul:
            psf, psf_full, diag = frame_epsf.build_frame_epsf(
                hdul, 1, _spec(instrument="nircam_lw"), instruments.get("nircam_lw")
            )
        assert diag["method"] == "stpsf-tier2b"
        assert "usable stars" in diag["tier1_reason"]
        assert psf.shape == (21, 21) and psf_full.shape == (61, 61)

    def test_missing_stpsf_recorded_not_fatal(self, no_cr, monkeypatch, tmp_path):
        import sys

        from astropy.io import fits
        from autoreduce.psf import frame_epsf

        monkeypatch.setitem(sys.modules, "stpsf", None)
        path = self._starless_jwst(tmp_path)
        with fits.open(path) as hdul:
            psf, psf_full, diag = frame_epsf.build_frame_epsf(
                hdul, 1, _spec(instrument="nircam_lw"), instruments.get("nircam_lw")
            )
        assert psf is None
        assert diag["method"] == "none"
        assert "unavailable" in diag["tier2b"]
        assert "usable stars" in diag["reason"]

    def test_hst_starless_stays_none(self, no_cr, tmp_path):
        from astropy.io import fits
        from autoreduce.psf import frame_epsf

        path = _write_exposure(tmp_path, nchips=1)
        with fits.open(path) as hdul:
            psf, _, diag = frame_epsf.build_frame_epsf(
                hdul, 1, _spec(), instruments.get("acs_wfc")
            )
        assert psf is None
        assert diag["method"] == "none"
        assert "tier2b" not in diag
