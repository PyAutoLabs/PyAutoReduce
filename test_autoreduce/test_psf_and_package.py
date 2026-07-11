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
from autoreduce.psf.starred_epsf import StarredUnavailableError, build_starred_epsf


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
    """The optional Tier-1b STARRED back-end (PyAutoReduce#35) is a defined
    interface that fails loudly until the spike wires it — never a silent
    fall-through to the photutils Tier-1 ePSF."""

    def test_backend_is_a_loud_hard_stop_until_wired(self):
        # Loud whether or not the optional GPL/JAX `starred` extra is installed:
        # the install message (extra absent) or the not-wired message (extra
        # present, spike pending) — both StarredUnavailableError, never silent.
        with pytest.raises(StarredUnavailableError):
            build_starred_epsf(np.zeros((100, 100)), None, (21, 21), (61, 61))

    def test_missing_optional_extra_names_the_install(self):
        try:
            import starred  # noqa: F401
        except ImportError:
            with pytest.raises(StarredUnavailableError, match="not installed"):
                build_starred_epsf(np.zeros((10, 10)), None, (21, 21), (61, 61))
        else:
            pytest.skip("starred installed; install-guard path not exercised")


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
