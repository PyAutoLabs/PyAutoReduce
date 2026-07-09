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
        assert entry["mdrizsky_subtracted"] == pytest.approx(40.0)

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
        exposure = _write_exposure(tmp_path, bunit="MJY/SR")
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

        def fake_frames(exposures, spec, adapter, out_dir, driz_cr_run):
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

        spec = TargetSpec(name="t", ra=RA, dec=DEC, instrument="nircam_lw",
                          frame_products=True)
        with pytest.raises(ValueError, match="HST-only"):
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
