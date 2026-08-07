import numpy as np
import pytest

from autoreduce.psf.epsf import (
    InsufficientStarsError,
    build_epsf,
    normalise_kernel,
)
from autoreduce.psf.stars import (
    StarSelection,
    reject_crowded,
    reject_edges,
    reject_near,
)
from autoreduce.psf.fallback import ModelPSFUnavailableError, model_psf
from autoreduce.psf.starred_epsf import (
    StarredUnavailableError,
    _core_centroid,
    _deliver,
    _downsample_box,
    build_starred_epsf,
)


class TestStarCuts:
    def test_reject_crowded_pairs(self):
        x = np.array([10.0, 12.0, 100.0])
        y = np.array([10.0, 10.0, 100.0])
        keep = reject_crowded(x, y, min_separation=5.0)
        assert keep.tolist() == [False, False, True]

    def test_reject_edges(self):
        x = np.array([5.0, 50.0])
        y = np.array([50.0, 50.0])
        keep = reject_edges(x, y, shape=(100, 100), margin=10)
        assert keep.tolist() == [False, True]

    def test_reject_near_target(self):
        x = np.array([50.0, 90.0])
        y = np.array([50.0, 90.0])
        keep = reject_near(x, y, 50.0, 50.0, radius=10.0)
        assert keep.tolist() == [False, True]


class TestNormaliseKernel:
    def test_unit_sum_and_shape(self):
        psf = np.random.default_rng(0).random((61, 61)) + 1.0
        kernel = normalise_kernel(psf, (21, 21))
        assert kernel.shape == (21, 21)
        assert kernel.sum() == pytest.approx(1.0)

    def test_even_shape_rejected(self):
        with pytest.raises(ValueError, match="odd"):
            normalise_kernel(np.ones((61, 61)), (20, 20))

    def test_oversized_request_rejected(self):
        with pytest.raises(ValueError, match="exceeds"):
            normalise_kernel(np.ones((21, 21)), (61, 61))

    def test_zero_flux_rejected(self):
        with pytest.raises(ValueError, match="flux"):
            normalise_kernel(np.zeros((21, 21)), (11, 11))


class TestTierFailuresAreLoud:
    def test_too_few_stars_raises(self):
        with pytest.raises(InsufficientStarsError, match="tier 2"):
            build_epsf(np.zeros((100, 100)), None, (21, 21), (61, 61))

    def test_tier2_unimplemented_is_hard_stop(self):
        with pytest.raises(ModelPSFUnavailableError, match="hard stop"):
            model_psf("lens", "F814W", (21, 21), (61, 61))


class TestStarredTier1bSeam:
    """The optional Tier-1b STARRED back-end (PyAutoReduce#35). The star guard
    and the centroid-preserving delivery are numpy-only and tested here; the
    STARRED reconstruction itself needs the GPL/JAX extra (importorskip)."""

    def _star_table(self, n, shape=(200, 200)):
        from astropy.table import Table

        rng = np.random.default_rng(1)
        ny, nx = shape
        return Table(
            {
                "xcentroid": rng.uniform(40, nx - 40, n),
                "ycentroid": rng.uniform(40, ny - 40, n),
            }
        )

    def test_too_few_stars_is_a_loud_hard_stop(self):
        # Runs before the STARRED import, so it is loud with or without the
        # extra — never a silent degradation to Tier 1.
        with pytest.raises(InsufficientStarsError, match="STARRED Tier-1b"):
            build_starred_epsf(
                np.zeros((200, 200)), np.ones((200, 200)), self._star_table(3),
                (21, 21), (61, 61),
            )

    def test_missing_extra_is_a_hard_stop_with_enough_stars(self):
        try:
            import starred  # noqa: F401
        except ImportError:
            with pytest.raises(StarredUnavailableError, match="not installed"):
                build_starred_epsf(
                    np.zeros((200, 200)), np.ones((200, 200)), self._star_table(12),
                    (21, 21), (61, 61),
                )
        else:
            pytest.skip("starred installed; install-guard path not exercised")

    def test_delivery_recenters_onto_the_central_pixel(self):
        # An ASYMMETRIC PSF (core + off-centre wing) deliberately off the
        # super-grid centre: naive cropping leaves it off-centre, and a *global*
        # centre of mass is biased by the wing (the 0.69px failure the #35
        # end-to-end run exposed). _deliver must centre on the core -> peak on
        # the odd kernel's central pixel.
        n = 140  # matches the real super-grid size (stamp 70 * subsampling 2)
        yy, xx = np.mgrid[0:n, 0:n].astype(float)
        cy, cx = n / 2 + 5.3, n / 2 - 3.6  # core off-centre by a non-integer amount
        core = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 4.0**2))
        wing = 0.15 * np.exp(-((yy - cy - 22) ** 2 + (xx - cx - 16) ** 2) / (2 * 9.0**2))
        kernel = _deliver(core + wing, subsampling=2, shape=(21, 21))
        assert kernel.shape == (21, 21)
        assert np.isclose(kernel.sum(), 1.0)
        # The PSF is centred on its CORE (peak on the central pixel; core
        # centroid ~0) — NOT its global centre of mass, which the wing offsets
        # by >1px and which it would be wrong to force to the middle.
        assert np.unravel_index(int(np.argmax(kernel)), kernel.shape) == (10, 10)
        cy, cx = _core_centroid(kernel)
        assert np.hypot(cy - 10, cx - 10) < 0.1  # this wing is deliberately harsh; real PSFs land ~0.01

    def test_downsample_box_shape_and_mean(self):
        img = np.arange(64, dtype=float).reshape(8, 8)
        ds = _downsample_box(img, 2)
        assert ds.shape == (4, 4)
        assert np.isclose(ds[0, 0], img[:2, :2].mean())


class TestCutout:
    def _mosaic(self):
        from astropy.io import fits
        from astropy.wcs import WCS

        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [2.0, -0.1]
        wcs.wcs.crpix = [100.5, 100.5]
        wcs.wcs.cdelt = [-0.05 / 3600.0, 0.05 / 3600.0]
        header = wcs.to_header()
        header["BUNIT"] = "ELECTRONS/S"
        header["EXPTIME"] = 1566.0
        data = np.arange(200 * 200, dtype=float).reshape(200, 200)
        return data, header

    def test_cutout_preserves_wcs_and_metadata(self, tmp_path):
        from astropy.io import fits
        from astropy.wcs import WCS

        from autoreduce.package.cutout import cutout_to_fits

        data, header = self._mosaic()
        out = tmp_path / "data.fits"
        cut = cutout_to_fits(data, header, ra=2.0, dec=-0.1, shape=(51, 51), out_path=out)
        assert cut.shape == (51, 51)

        with fits.open(out) as hdul:
            out_header = hdul[0].header
            assert out_header["BUNIT"] == "ELECTRONS/S"
            assert out_header["EXPTIME"] == 1566.0
            scales = np.abs(np.diag(WCS(out_header).pixel_scale_matrix)) * 3600
            assert scales == pytest.approx([0.05, 0.05])
            # The cutout centre maps back to the requested sky position.
            x, y = WCS(out_header).world_to_pixel_values(2.0, -0.1)
            assert float(x) == pytest.approx(25.0, abs=0.51)
            assert float(y) == pytest.approx(25.0, abs=0.51)

    def test_cutout_off_mosaic_fails(self, tmp_path):
        from autoreduce.package.cutout import cutout_to_fits

        data, header = self._mosaic()
        with pytest.raises(Exception):
            cutout_to_fits(
                data, header, ra=50.0, dec=50.0, shape=(51, 51),
                out_path=tmp_path / "data.fits",
            )


def test_weight_uniformity_diagnostic():
    from autoreduce.drizzle.diagnostics import check_weight_uniformity, weight_uniformity

    flat = np.full((50, 50), 100.0)
    assert weight_uniformity(flat) == pytest.approx(0.0)
    rng = np.random.default_rng(1)
    speckled = np.abs(rng.normal(100.0, 40.0, size=(50, 50)))
    verdict = check_weight_uniformity(speckled)
    assert not verdict["acceptable"]
    with pytest.raises(ValueError, match="coverage"):
        weight_uniformity(np.zeros((5, 5)))


class TestLocalWeightDeficit:
    """
    The local coverage guard (issue #65 leg 2) — the detector for the ACS
    stripe class that both pre-existing guards pass silently.
    """

    # 0.05"/pix, so the 1.5" science radius is 30 px around the centre.
    PIXEL_SCALE = 0.05
    CENTER = (60.0, 60.0)

    def _striped(self, n_exposures=4):
        """
        Uniform coverage with one column through the lens core down to
        (N-1)/N — one exposure lost along a CCD column, finite and positive
        throughout, exactly the reported failure.
        """
        wht = np.full((121, 121), 100.0)
        wht[:, 60] = 100.0 * (n_exposures - 1) / n_exposures
        return wht

    def test_flat_coverage_is_clean(self):
        from autoreduce.drizzle.diagnostics import check_local_weight_deficit

        verdict = check_local_weight_deficit(
            np.full((121, 121), 100.0), self.CENTER, self.PIXEL_SCALE
        )
        assert verdict["acceptable"]
        assert verdict["science_median_ratio"] == pytest.approx(1.0)
        assert verdict["min_line_ratio"] == pytest.approx(1.0)

    def test_stripe_through_the_core_is_caught(self):
        from autoreduce.drizzle.diagnostics import check_local_weight_deficit

        verdict = check_local_weight_deficit(
            self._striped(4), self.CENTER, self.PIXEL_SCALE
        )
        assert not verdict["acceptable"]
        # One lost exposure of four: the column sits at 3/4 of the median...
        assert verdict["min_line_ratio"] == pytest.approx(0.75)
        assert verdict["min_line_axis"] == "column"
        assert verdict["min_line_index"] == 60
        # ...while the science region as a whole barely notices it, which is
        # precisely why a region-median test alone would not do.
        assert verdict["science_median_ratio"] == pytest.approx(1.0)

    def test_the_existing_guards_are_blind_to_the_same_map(self):
        # The regression this leg exists for: without the diagnostic above,
        # a striped reduction ships clean.
        from autoreduce.drizzle.diagnostics import check_weight_uniformity
        from autoreduce.noise.rms import mask_isolated_bad_pixels

        wht = self._striped(4)
        assert check_weight_uniformity(wht)["acceptable"]

        # The bad-pixel policy sees the noise map, so mirror the stripe into
        # noise space: reduced IVM weight -> elevated but finite noise.
        noise = np.full((121, 121), 1.0)
        noise[:, 60] = np.sqrt(4.0 / 3.0)
        _, _, diag = mask_isolated_bad_pixels(
            np.zeros_like(noise),
            noise,
            center_xy=self.CENTER,
            pixel_scale=self.PIXEL_SCALE,
        )
        # No pixel is non-finite or <= 0, so nothing is even a candidate —
        # not the clustering check, not the 1.5" lens-core protection.
        assert diag["n_masked_pixels"] == 0

    def test_row_oriented_stripe_is_caught_too(self):
        # A detector column lands on an image row after drizzling to a sky
        # frame at the right orientation, so both axes are tested.
        from autoreduce.drizzle.diagnostics import check_local_weight_deficit

        wht = np.full((121, 121), 100.0)
        wht[60, :] = 50.0
        verdict = check_local_weight_deficit(wht, self.CENTER, self.PIXEL_SCALE)
        assert not verdict["acceptable"]
        assert verdict["min_line_axis"] == "row"
        assert verdict["min_line_ratio"] == pytest.approx(0.5)

    def test_zero_coverage_inside_the_region_reads_as_total_loss(self):
        from autoreduce.drizzle.diagnostics import check_local_weight_deficit

        wht = np.full((121, 121), 100.0)
        wht[:, 60] = 0.0
        verdict = check_local_weight_deficit(wht, self.CENTER, self.PIXEL_SCALE)
        assert verdict["min_line_ratio"] == pytest.approx(0.0)
        assert not verdict["acceptable"]

    def test_uniformly_degraded_science_region_is_caught(self):
        # The complementary failure: no stripe, but the whole lens region
        # sits below the cutout median — min_line_ratio and the region
        # median agree here.
        from autoreduce.drizzle.diagnostics import check_local_weight_deficit

        wht = np.full((121, 121), 100.0)
        # The 1.5" radius is 30 px here, so the region spans 30..90.
        wht[30:91, 30:91] = 70.0
        verdict = check_local_weight_deficit(wht, self.CENTER, self.PIXEL_SCALE)
        assert not verdict["acceptable"]
        assert verdict["science_median_ratio"] == pytest.approx(0.7)

    def test_empty_and_off_cutout_inputs_raise(self):
        from autoreduce.drizzle.diagnostics import local_weight_deficit

        with pytest.raises(ValueError, match="coverage"):
            local_weight_deficit(np.zeros((10, 10)), (5.0, 5.0), self.PIXEL_SCALE)
        with pytest.raises(ValueError, match="off the cutout"):
            local_weight_deficit(
                np.full((10, 10), 1.0), (500.0, 500.0), self.PIXEL_SCALE
            )


def test_provenance_record(tmp_path):
    import json

    from autoreduce.package.provenance import write_reduction_json

    path = write_reduction_json(tmp_path, {"target": {"name": "lens"}})
    payload = json.loads(path.read_text())
    assert payload["target"]["name"] == "lens"
    assert "astropy" in payload["software"]
    assert payload["written_at"].endswith("Z")


def test_mast_query_hygiene():
    from autoreduce.acquire.mast import is_direct_observation

    assert is_direct_observation("j9op01010", "10886")
    assert not is_direct_observation("hst_skycell-p1322x03y02_acs_wfc_f814w_all", "--")
    assert not is_direct_observation("j9op01010", "--")
    # HAP visit-level associations carry numeric proposals but re-deliver
    # renamed copies of the member exposures — ingesting them alongside the
    # direct rows drizzles every exposure twice.
    assert not is_direct_observation("hst_10886_01_acs_wfc_f814w_j9op01l7", "10886")

    # MAST also attaches the HAP copies to the member exposure's own product
    # list, so the product table needs the same hygiene.
    from autoreduce.acquire.mast import is_direct_product

    assert is_direct_product("j9op01l7q_flc.fits")
    assert not is_direct_product("hst_10886_01_acs_wfc_f814w_j9op01l7_flc.fits")


def test_reject_crowded_matches_reference_loop():
    """Randomized equivalence vs the original O(N^2) loop implementation."""
    from autoreduce.psf.stars import reject_crowded

    def reference(x, y, min_separation):
        keep = np.ones(len(x), dtype=bool)
        for i in range(len(x)):
            d2 = (x - x[i]) ** 2 + (y - y[i]) ** 2
            d2[i] = np.inf
            if (d2 < min_separation**2).any():
                keep[i] = False
        return keep

    rng = np.random.default_rng(7)
    for n in (0, 1, 2, 50, 300):
        x = rng.uniform(0, 500, n)
        y = rng.uniform(0, 500, n)
        for sep in (1.0, 25.0, 100.0):
            assert (
                reject_crowded(x, y, sep) == reference(x, y, sep)
            ).all(), (n, sep)


def test_registered_ratios_recovers_known_shift_and_scale():
    from scipy.ndimage import shift as nd_shift

    from autoreduce.validation import registered_ratios

    rng = np.random.default_rng(3)
    ref_data = rng.normal(0.0, 0.01, (120, 120))
    yy, xx = np.mgrid[0:120, 0:120]
    ref_data += 8.0 * np.exp(-(((xx - 60) ** 2 + (yy - 60) ** 2) / (2 * 3.0**2)))
    ref_noise = np.full((120, 120), 0.01)

    new_data = 1.5 * nd_shift(ref_data, (1.25, -0.75), order=3)
    new_noise = 2.0 * ref_noise
    out = registered_ratios(new_data, new_noise, ref_data, ref_noise)
    # The offset is the shift applied to `new` to register it onto `ref` —
    # the negative of new's displacement.
    assert out["offset"][0] == pytest.approx(-1.25, abs=0.15)
    assert out["offset"][1] == pytest.approx(0.75, abs=0.15)
    assert out["data_ratio_median"] == pytest.approx(1.5, rel=0.05)
    assert out["noise_ratio_median"] == pytest.approx(2.0, rel=0.05)

    # Masked-by-noise pixels are excluded from the noise statistics.
    new_noise_masked = new_noise.copy()
    new_noise_masked[5, 5] = 1.0e8
    out2 = registered_ratios(new_data, new_noise_masked, ref_data, ref_noise)
    assert out2["noise_ratio_median"] == pytest.approx(2.0, rel=0.05)

    with pytest.raises(ValueError, match="shape mismatch"):
        registered_ratios(new_data[:100], new_noise[:100], ref_data, ref_noise)


class TestStarPassDecoupling:
    """_psf's star-source pass selection (issue #62): star finding is
    decoupled from the shipped science mosaic, and provenance always
    records which pass fed the stars."""

    def _ctx(self, spec, single_exposure, work_dir=None):
        from pathlib import Path

        from autoreduce import instruments
        from autoreduce.pipeline import _StageContext

        ctx = _StageContext(
            spec=spec,
            adapter=instruments.get("acs_wfc"),
            cache=None,
            out_dir=Path("."),
            work_dir=work_dir or Path("."),
        )
        ctx.record["drizzle"] = {"single_exposure_branch": single_exposure}
        return ctx

    def _spec(self, **overrides):
        from autoreduce.target import TargetSpec

        return TargetSpec(name="lens", ra=2.0, dec=-0.1, **overrides)

    def test_auto_default_uses_science_mosaic_and_records_why(self):
        from autoreduce.pipeline import _star_pass_image

        ctx = self._ctx(self._spec(), single_exposure=False)
        sci, header = np.zeros((5, 5)), object()
        star_sci, star_header, prov = _star_pass_image(ctx, sci, header)
        assert star_sci is sci and star_header is header
        assert prov["star_source_pass"] == "science"
        assert "no_cr" in prov["star_source_reason"]

    def test_single_exposure_branch_is_already_least_rejected(self):
        from autoreduce.pipeline import _star_pass_image

        ctx = self._ctx(self._spec(psf_star_pass="no_cr"), single_exposure=True)
        sci, header = np.zeros((5, 5)), object()
        star_sci, _, prov = _star_pass_image(ctx, sci, header)
        # "no_cr" on the single-exposure branch never drizzles again: the
        # science mosaic had no CR rejection to begin with.
        assert star_sci is sci
        assert prov["star_source_pass"] == "science"
        assert "single-exposure" in prov["star_source_reason"]

    def test_deepcr_route_records_per_frame_reason(self):
        from autoreduce.pipeline import _star_pass_image

        ctx = self._ctx(self._spec(cr_method="deepcr"), single_exposure=False)
        _, _, prov = _star_pass_image(ctx, np.zeros((5, 5)), object())
        assert prov["star_source_pass"] == "science"
        assert "deepcr" in prov["star_source_reason"]

    def test_no_cr_opt_in_builds_the_dedicated_pass(self, monkeypatch, tmp_path):
        from astropy.io import fits

        from autoreduce import pipeline as pipeline_mod
        from autoreduce.pipeline import _star_pass_image

        star_path = tmp_path / "lens_f814w_starpass_sci.fits"
        fits.PrimaryHDU(np.full((4, 4), 7.0, dtype=np.float32)).writeto(star_path)
        calls = {}

        def fake_star_pass(exposures, spec, adapter, output_dir):
            calls["output_dir"] = output_dir
            return star_path, {"resetbits": 0, "driz_cr": False}

        monkeypatch.setattr(
            pipeline_mod.combine_mod, "combine_star_pass", fake_star_pass
        )
        ctx = self._ctx(
            self._spec(psf_star_pass="no_cr"),
            single_exposure=False,
            work_dir=tmp_path,
        )
        science = np.zeros((5, 5))
        star_sci, star_header, prov = _star_pass_image(ctx, science, object())
        assert calls["output_dir"] == tmp_path
        assert star_sci is not science
        assert star_sci[0, 0] == pytest.approx(7.0)
        assert prov["star_source_pass"] == "no_cr_drizzle"
        assert prov["star_pass_kwargs"]["resetbits"] == 0


class TestCrDialFailFast:
    """reduce_target rejects unsupported dial/instrument combinations
    before any download (issues #61/#62)."""

    def _reduce(self, tmp_path, **spec_overrides):
        from autoreduce.pipeline import reduce_target
        from autoreduce.target import TargetSpec

        spec = TargetSpec(name="x", ra=2.0, dec=-0.1, **spec_overrides)
        return reduce_target(
            spec, cache_root=tmp_path / "cache", output_root=tmp_path / "out"
        )

    def test_deepcr_needs_the_astrodrizzle_backend(self, tmp_path):
        with pytest.raises(ValueError, match="astrodrizzle"):
            self._reduce(tmp_path, instrument="nircam_lw", cr_method="deepcr")

    def test_deepcr_needs_a_registered_model(self, tmp_path):
        # wfc3_ir is astrodrizzle-combined but has no deepCR model — its
        # cosmic rays are already per-frame flagged by calwf3 ramp fitting.
        with pytest.raises(ValueError, match="deepCR model"):
            self._reduce(tmp_path, instrument="wfc3_ir", cr_method="deepcr")

    def test_no_cr_star_pass_needs_the_astrodrizzle_backend(self, tmp_path):
        with pytest.raises(ValueError, match="psf_star_pass"):
            self._reduce(tmp_path, instrument="nircam_lw", psf_star_pass="no_cr")

    def test_no_cr_star_pass_conflicts_with_psf_from_frames(self, tmp_path):
        # psf_from_frames never finds stars on the mosaic; a silent no-op of
        # an explicitly requested second drizzle would hide the mistake.
        with pytest.raises(ValueError, match="psf_star_pass"):
            self._reduce(
                tmp_path, psf_star_pass="no_cr", psf_from_frames=True
            )
