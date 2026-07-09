"""Keck ground stages: adapters, calibrate, sky, registration (numpy/astropy)."""

import numpy as np
import pytest

from autoreduce import instruments
from autoreduce.align.registration import offsets_to_reference, phase_offset
from autoreduce.calibrate import build_calibrations, calibrate_frame
from autoreduce.instruments.nirc2 import (
    DISTORTION_EPOCH_BOUNDARY_MJD,
    NIRC2_DETECTOR,
    distortion_solution_for_mjd,
)
from autoreduce.sky import running_sky_subtract


class TestNirc2Adapters:
    def test_registered(self):
        narrow = instruments.get("nirc2_narrow")
        wide = instruments.get("nirc2_wide")
        assert narrow.archive == "koa"
        assert narrow.observatory == "keck"
        assert narrow.combine_backend == "nirc2_native"
        assert narrow.native_scale == pytest.approx(0.009942)
        assert wide.native_scale == pytest.approx(0.039686)

    def test_space_adapters_unaffected(self):
        assert instruments.get("acs_wfc").archive == "mast"
        assert instruments.get("nircam_sw").archive == "mast"

    def test_distortion_epoch_routing(self):
        assert distortion_solution_for_mjd(56000.0) == "yelda2010"
        assert (
            distortion_solution_for_mjd(DISTORTION_EPOCH_BOUNDARY_MJD + 1)
            == "service2016"
        )
        with pytest.raises(ValueError):
            distortion_solution_for_mjd(0.0)

    def test_sharp_scale_convention(self):
        # 10 mas narrow / 40 mas wide (Chen et al. 2019).
        assert instruments.get("nirc2_narrow").recommended_final_scale == 0.010
        assert instruments.get("nirc2_wide").recommended_final_scale == 0.040


class TestCalibrate:
    def _calib(self, shape=(32, 32)):
        rng = np.random.default_rng(1)
        darks = [np.full(shape, 10.0) + rng.normal(0, 0.1, shape) for _ in range(3)]
        flat_on = [np.full(shape, 2000.0) + rng.normal(0, 5.0, shape) for _ in range(3)]
        flat_off = [np.full(shape, 500.0) + rng.normal(0, 5.0, shape) for _ in range(3)]
        return darks, flat_on, flat_off

    def test_flat_normalised_and_lamp_off_subtracted(self):
        darks, flat_on, flat_off = self._calib()
        calib = build_calibrations(darks, flat_on, flat_off)
        assert np.median(calib.master_flat) == pytest.approx(1.0, abs=1e-6)
        assert calib.provenance["n_flat_off_frames"] == 3

    def test_dead_and_hot_pixels_flagged(self):
        darks, flat_on, flat_off = self._calib()
        for f in flat_on:
            f[5, 5] = 500.0  # dead: no lamp response above the off level
        for d in darks:
            d[7, 7] = 500.0  # hot
        calib = build_calibrations(darks, flat_on, flat_off)
        assert calib.bad_pixel_mask[5, 5]
        assert calib.bad_pixel_mask[7, 7]
        assert calib.provenance["n_bad_pixels"] >= 2

    def test_calibrate_frame_units_and_bads(self):
        darks, flat_on, flat_off = self._calib()
        for f in flat_on:
            f[5, 5] = 500.0
        calib = build_calibrations(darks, flat_on, flat_off)
        raw = np.full((32, 32), 100.0)  # per-coadd DN
        out = calibrate_frame(raw, calib, gain_e_per_dn=4.0, coadds=6)
        # (100 - 10 dark) DN * 4 e-/DN * 6 coadds, flat ~ 1.
        finite = out[np.isfinite(out)]
        assert np.median(finite) == pytest.approx(90.0 * 4.0 * 6.0, rel=0.01)
        assert np.isnan(out[5, 5])

    def test_no_darks_is_allowed_and_recorded(self):
        _, flat_on, flat_off = self._calib()
        calib = build_calibrations([], flat_on, flat_off)
        assert calib.master_dark is None
        assert calib.provenance["dark_subtraction"] is False
        raw = np.full((32, 32), 100.0)
        out = calibrate_frame(raw, calib, gain_e_per_dn=4.0, coadds=1)
        assert np.median(out[np.isfinite(out)]) == pytest.approx(400.0, rel=0.01)

    def test_garbage_flat_fails_loudly(self):
        with pytest.raises(ValueError, match="non-positive median"):
            build_calibrations([], [np.zeros((8, 8))], None)


class TestRunningSky:
    def _frames(self, n=8, shape=(48, 48), source=True):
        rng = np.random.default_rng(2)
        frames = []
        for i in range(n):
            sky = 1000.0 + 50.0 * i  # K'-band sky drifting in time
            frame = sky + rng.normal(0, 3.0, shape)
            if source:
                frame[20:28, 20:28] += 500.0  # the target, same dither spot
            frames.append(frame)
        return frames

    def test_removes_time_varying_sky(self):
        subtracted, prov = running_sky_subtract(self._frames(), window=4)
        for sub in subtracted:
            background = np.concatenate([sub[:10].ravel(), sub[-10:].ravel()])
            assert abs(np.median(background)) < 5.0
        assert len(prov["sky_levels_e"]) == 8
        assert prov["sky_levels_e"][-1] > prov["sky_levels_e"][0]

    def test_source_survives_subtraction(self):
        subtracted, _ = running_sky_subtract(self._frames(), window=4)
        # An un-dithered source is the worst case for in-field sky (it sits
        # in every window frame); the object mask is what protects it.
        core = subtracted[4][22:26, 22:26]
        assert np.median(core) > 400.0

    def test_single_frame_rejected(self):
        with pytest.raises(ValueError, match=">= 2 frames"):
            running_sky_subtract(self._frames(n=1), window=4)


class TestRegistration:
    def _frame(self, shape=(64, 64), at=(32.0, 32.0)):
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        return np.exp(-(((yy - at[0]) ** 2 + (xx - at[1]) ** 2) / (2 * 2.0**2)))

    def test_recovers_integer_shift(self):
        # Convention (what nirc2_combine subtracts in the pixmap):
        # offset = source position in frame minus position in reference.
        ref = self._frame()
        shifted = self._frame(at=(35.0, 30.0))
        dy, dx = phase_offset(ref, shifted)
        assert dy == pytest.approx(3.0, abs=0.05)
        assert dx == pytest.approx(-2.0, abs=0.05)

    def test_subpixel_shift(self):
        ref = self._frame()
        shifted = self._frame(at=(32.6, 31.7))
        dy, dx = phase_offset(ref, shifted)
        assert dy == pytest.approx(0.6, abs=0.15)
        assert dx == pytest.approx(-0.3, abs=0.15)

    def test_first_frame_is_reference(self):
        frames = [self._frame(), self._frame(at=(30.0, 34.0))]
        offsets = offsets_to_reference(frames)
        assert offsets[0] == (0.0, 0.0)

    def test_nan_tolerant(self):
        ref = self._frame()
        shifted = self._frame(at=(36.0, 28.0))
        shifted[0:4, 0:4] = np.nan
        dy, dx = phase_offset(ref, shifted)
        assert np.isfinite(dy) and np.isfinite(dx)

    def test_gain_read_noise_constants_sane(self):
        # The blank-sky closure validates these on real data; here just pin
        # the adapter-owned values the noise model uses.
        assert NIRC2_DETECTOR.gain_e_per_dn == 4.0
        assert NIRC2_DETECTOR.read_noise_e_cds == 38.0
