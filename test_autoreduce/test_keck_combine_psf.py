"""
NIRC2 native combine + tier-A PSF on synthetic prepared frames.

Uses the standalone ``drizzle`` resampler (a light dependency, not the
drizzlepac stack); skipped cleanly where it is absent.
"""

import numpy as np
import pytest

drizzle_pkg = pytest.importorskip("drizzle")

from astropy.io import fits  # noqa: E402

from autoreduce import instruments  # noqa: E402
from autoreduce.drizzle import nirc2_combine  # noqa: E402
from autoreduce.psf.nirc2_star import build_candidates, group_epochs  # noqa: E402
from autoreduce.target import TargetSpec  # noqa: E402

SHAPE = (64, 64)
NARROW = instruments.get("nirc2_narrow")


def _write_distortion(tmp_path, shape=SHAPE):
    """Identity distortion lookup tables (zero shifts)."""
    paths = []
    for axis in "XY":
        p = tmp_path / f"dist_{axis}.fits"
        fits.PrimaryHDU(np.zeros(shape, dtype=np.float32)).writeto(p)
        paths.append(p)
    return paths


def _write_frame(tmp_path, name, data, dist_paths, itime=10.0, coadds=6,
                 sky_e=1000.0, mjd=59000.0):
    header = fits.Header()
    header["ITIME"] = itime
    header["COADDS"] = coadds
    header["MJD-OBS"] = mjd
    header["SKYLEV"] = sky_e
    header["DISTX"] = str(dist_paths[0])
    header["DISTY"] = str(dist_paths[1])
    path = tmp_path / name
    fits.PrimaryHDU(data.astype(np.float32), header=header).writeto(path)
    return path


def _gaussian(shape, at, flux=6.0e4, sigma=1.8):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    g = np.exp(-(((yy - at[0]) ** 2 + (xx - at[1]) ** 2) / (2 * sigma**2)))
    return flux * g / g.sum()


def _spec(**overrides):
    kwargs = dict(
        name="synthetic",
        ra=150.0,
        dec=2.0,
        instrument="nirc2_narrow",
        filter_name="Kp",
        final_scale=NARROW.native_scale,  # scale ratio 1 for flux checks
        final_pixfrac=1.0,
        cutout_shape=(41, 41),
        psf_shape=(11, 11),
        psf_full_shape=(21, 21),
    )
    kwargs.update(overrides)
    return TargetSpec(**kwargs)


class TestNirc2Combine:
    def _combine_dithered(self, tmp_path, n=4):
        dist = _write_distortion(tmp_path)
        rng = np.random.default_rng(3)
        paths = []
        offsets = [(0.0, 0.0), (3.0, -2.0), (-4.0, 5.0), (6.0, 6.0)][:n]
        for i, (dy, dx) in enumerate(offsets):
            data = _gaussian(SHAPE, (32.0 + dy, 32.0 + dx)) + rng.normal(
                0, 2.0, SHAPE
            )
            paths.append(
                _write_frame(tmp_path, f"f{i}.fits", data, dist,
                             mjd=59000.0 + i * 1e-3)
            )
        out = tmp_path / "out"
        out.mkdir()
        return nirc2_combine.combine(paths, _spec(), NARROW, out)

    def test_flux_conserved_and_exptime_summed(self, tmp_path):
        sci_path, wht_path, prov = self._combine_dithered(tmp_path)
        sci = fits.getdata(sci_path).astype(float)
        header = fits.getheader(sci_path)
        # 4 frames x (10 s x 6 coadds); mosaic in e-/s.
        assert header["EXPTIME"] == pytest.approx(240.0)
        assert header["BUNIT"] == "ELECTRONS/S"
        # Total source rate: flux 6e4 e- per frame over 60 s = 1000 e-/s.
        assert sci.sum() == pytest.approx(1000.0, rel=0.05)
        assert prov["total_exptime"] == pytest.approx(240.0)
        assert len(prov["registration_offsets_native_pix"]) == 4

    def test_registration_stacks_the_source(self, tmp_path):
        sci_path, _, _ = self._combine_dithered(tmp_path)
        sci = fits.getdata(sci_path).astype(float)
        # A mis-registered stack smears the source; the peak rate of the
        # aligned stack must reach most of the single-frame peak rate.
        single_peak = _gaussian(SHAPE, (32.0, 32.0)).max() / 60.0
        assert sci.max() > 0.7 * single_peak

    def test_noise_closure_on_blank_frames(self, tmp_path):
        from autoreduce.noise.rms import noise_map_from

        from autoreduce.instruments.nirc2 import NIRC2_DETECTOR

        dist = _write_distortion(tmp_path)
        rng = np.random.default_rng(4)
        sky_e, itime, coadds = 1000.0, 10.0, 6
        # Frames whose scatter matches their own background budget — for
        # narrow-camera K' the read-noise term (RN^2 x coadds) dominates the
        # sky, which is physically the NIRC2 regime. The target is present:
        # registration is defined by the field, never blank noise.
        t_frame = itime * coadds
        var_e = (
            sky_e
            + NIRC2_DETECTOR.dark_e_per_s * t_frame
            + NIRC2_DETECTOR.read_noise_e_cds**2 * coadds
        )
        paths = [
            _write_frame(
                tmp_path,
                f"b{i}.fits",
                _gaussian(SHAPE, (32.0, 32.0))
                + rng.normal(0, np.sqrt(var_e), SHAPE),
                dist,
                itime=itime,
                coadds=coadds,
                sky_e=sky_e,
                mjd=59000.0 + i * 1e-3,
            )
            for i in range(4)
        ]
        out = tmp_path / "out"
        out.mkdir()
        spec = _spec()
        sci_path, wht_path, prov = nirc2_combine.combine(paths, spec, NARROW, out)
        sci = fits.getdata(sci_path).astype(float)
        wht = fits.getdata(wht_path).astype(float)
        noise = noise_map_from(
            sci, wht, exptime=240.0,
            correlated_noise_factor=prov["correlated_noise_factor"],
        )
        # Interior blank annulus: inside coverage, outside the source core.
        yy, xx = np.mgrid[0 : sci.shape[0], 0 : sci.shape[1]]
        blank = (
            (np.hypot(yy - sci.shape[0] / 2, xx - sci.shape[1] / 2) > 12)
            & (yy > 10) & (yy < sci.shape[0] - 10)
            & (xx > 10) & (xx < sci.shape[1] - 10)
        )
        empirical = np.std(sci[blank])
        predicted = np.median(noise[blank])
        # Drizzle correlates neighbours (the empirical mosaic std understates
        # independent-pixel noise by ~1/R, which the factor R in `predicted`
        # compensates) — the closure bounds the accounting within a factor
        # ~2, catching unit errors (gain, coadds, cps) which show as x6-x40.
        assert 0.5 < predicted / empirical < 3.0

    def test_wide_camera_fails_loudly(self, tmp_path):
        dist = _write_distortion(tmp_path)
        path = _write_frame(tmp_path, "w.fits", np.ones(SHAPE), dist)
        with pytest.raises(NotImplementedError, match="narrow camera only"):
            nirc2_combine.combine(
                [path],
                _spec(instrument="nirc2_wide", final_scale=0.039686),
                instruments.get("nirc2_wide"),
                tmp_path,
            )

    def test_distortion_shape_mismatch_fails(self, tmp_path):
        dist = _write_distortion(tmp_path, shape=(32, 32))
        path = _write_frame(tmp_path, "m.fits", np.ones(SHAPE), dist)
        with pytest.raises(ValueError, match="do not match the frame"):
            nirc2_combine.combine([path], _spec(), NARROW, tmp_path)


class TestTierAPsf:
    def test_group_epochs(self):
        mjds = [59000.0, 59000.001, 59000.002, 59000.5, 59000.501]
        groups = group_epochs(mjds)
        assert [sorted(g) for g in groups] == [[0, 1, 2], [3, 4]]

    def test_candidates_built_and_provisional(self, tmp_path):
        dist = _write_distortion(tmp_path)
        rng = np.random.default_rng(5)
        paths, mjds = [], []
        # Two epochs, two frames each; second epoch sharper (higher Strehl).
        for i, (mjd, sigma) in enumerate(
            [(59000.0, 2.5), (59000.001, 2.5), (59000.5, 1.2), (59000.501, 1.2)]
        ):
            data = _gaussian(SHAPE, (32.0, 32.0), sigma=sigma) + rng.normal(
                0, 0.5, SHAPE
            )
            paths.append(
                _write_frame(tmp_path, f"s{i}.fits", data, dist, mjd=mjd)
            )
            mjds.append(mjd)
        psf, psf_full, candidates, diag = build_candidates(
            paths, mjds, _spec(), NARROW, tmp_path / "work"
        )
        assert diag["psf_provisional"] is True
        assert diag["n_candidates"] == 2
        assert diag["selected_epoch"] == 1  # the sharper epoch wins
        assert psf.shape == (11, 11)
        assert psf_full.shape == (21, 21)
        assert psf.sum() == pytest.approx(1.0)
        assert all(c.sum() == pytest.approx(1.0) for c in candidates)
        assert (
            diag["candidates"][1]["fwhm_arcsec"]
            < diag["candidates"][0]["fwhm_arcsec"]
        )
