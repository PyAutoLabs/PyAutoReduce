import numpy as np
import pytest

from autoreduce.noise.rms import (
    assert_finite_within,
    casertano_r,
    empirical_background_rms,
    noise_map_from,
)


class TestCasertanoR:
    def test_shift_and_add_limit(self):
        # p=1, s=1 is shift-and-add: r = 2/3, R = 1.5
        assert casertano_r(1.0, 1.0) == pytest.approx(1.5)

    def test_spike_value(self):
        # The value the SLACS parity study validated against legacy noise.
        assert casertano_r(0.8, 1.0) == pytest.approx(1.364, abs=1e-3)

    def test_interlacing_limit(self):
        # p -> 0 at fixed s: no flux sharing, R -> 1.
        assert casertano_r(1e-6, 1.0) == pytest.approx(1.0, abs=1e-3)

    def test_smaller_pixfrac_reduces_correlation(self):
        assert casertano_r(0.6, 1.0) < casertano_r(0.8, 1.0) < casertano_r(1.0, 1.0)

    def test_branch_s_less_than_p(self):
        # Finer output grid than the drop uses the (s/p) branch: more
        # correlation than shift-and-add, R -> inf as s -> 0.
        r_fine = casertano_r(1.0, 0.5)
        assert r_fine > casertano_r(1.0, 1.0)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            casertano_r(0.0, 1.0)
        with pytest.raises(ValueError):
            casertano_r(1.5, 1.0)
        with pytest.raises(ValueError):
            casertano_r(0.8, 0.0)


class TestNoiseMapFrom:
    def test_background_only(self):
        sci = np.zeros((4, 4))
        wht = np.full((4, 4), 25.0)
        noise = noise_map_from(sci, wht, exptime=100.0)
        assert noise == pytest.approx(np.full((4, 4), 0.2))

    def test_poisson_term_adds_in_quadrature(self):
        sci = np.full((2, 2), 9.0)  # cps
        wht = np.full((2, 2), 4.0)
        noise = noise_map_from(sci, wht, exptime=1.0)
        assert noise == pytest.approx(np.full((2, 2), np.sqrt(9.0 + 0.25)))

    def test_negative_sky_pixels_floor_poisson_at_zero(self):
        sci = np.array([[-5.0]])
        wht = np.array([[4.0]])
        noise = noise_map_from(sci, wht, exptime=1.0)
        assert noise == pytest.approx(np.array([[0.5]]))

    def test_correlated_factor_scales_linearly(self):
        sci = np.full((2, 2), 1.0)
        wht = np.full((2, 2), 1.0)
        base = noise_map_from(sci, wht, exptime=1.0)
        scaled = noise_map_from(sci, wht, exptime=1.0, correlated_noise_factor=1.364)
        assert scaled == pytest.approx(1.364 * base)

    def test_zero_weight_becomes_nan_not_patched(self):
        sci = np.zeros((2, 2))
        wht = np.array([[1.0, 0.0], [1.0, -1.0]])
        noise = noise_map_from(sci, wht, exptime=1.0)
        assert np.isnan(noise[0, 1]) and np.isnan(noise[1, 1])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            noise_map_from(np.zeros((2, 2)), np.zeros((3, 3)), exptime=1.0)

    def test_bad_exptime_raises(self):
        with pytest.raises(ValueError):
            noise_map_from(np.zeros((2, 2)), np.ones((2, 2)), exptime=0.0)

    def test_sub_unity_correlated_factor_raises(self):
        with pytest.raises(ValueError):
            noise_map_from(
                np.zeros((2, 2)), np.ones((2, 2)), exptime=1.0,
                correlated_noise_factor=0.9,
            )


class TestAssertFiniteWithin:
    def test_clean_map_passes(self):
        assert_finite_within(np.ones((3, 3)), "test")

    def test_nan_inside_cutout_fails_loudly(self):
        noise = np.ones((3, 3))
        noise[1, 1] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            assert_finite_within(noise, "test")

    def test_zero_noise_fails_loudly(self):
        noise = np.ones((3, 3))
        noise[0, 0] = 0.0
        with pytest.raises(ValueError):
            assert_finite_within(noise, "test")


def test_empirical_background_rms_recovers_sigma():
    rng = np.random.default_rng(0)
    sci = rng.normal(0.0, 0.02, size=(200, 200))
    assert empirical_background_rms(sci) == pytest.approx(0.02, rel=0.05)
