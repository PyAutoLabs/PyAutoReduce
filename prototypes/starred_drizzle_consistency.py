"""
STARRED Tier-1b — rigorous + adversarial drizzle-consistency test (#35).

The feasibility spike showed STARRED reconstructs a clean PSF on a real field,
but a real field gives no ground truth. The open design question is sharper
than "does it look right": when STARRED's super-sampled PSF is resampled onto
the mosaic grid and delivered as an odd kernel, is it *drizzle-consistent* — no
sub-pixel centroid shift, no broadening, flux preserved? This tests that with a
known truth, then tries to BREAK it.

Method (per scenario): (1) known continuous PSF on a super-sampled grid
(elliptical Moffat + off-centre bump so the starlet channel must work);
(2) synthesise N stars the way a drizzled mosaic samples the PSF — random
sub-pixel phase shift -> box-average to the data grid -> flux -> noise;
(3) STARRED `build_psf` -> recovered super-sampled PSF; (4) resample to the
mosaic grid and deliver a centred odd kernel (`_deliver`); (5) measure
recovered-vs-truth: residual RMS, centroid offset (px), size ratio, flux error.

Delivery `_deliver` = `Downsample` (route a) + **centroid-preserving crop**: the
naive even-super -> odd-kernel crop injects a ~0.5 px centre offset (see the
`geometry_only` check, which reports naive-vs-centred), so production must crop
around the measured centroid and sub-pixel-recentre. That fix is baked in here.

Scenarios:
  baseline               well-sampled, dithered, high SNR, N=25 (should pass)
  no_dither              all stars at ONE sub-pixel phase (Nyquist degeneracy)
  undersampled           FWHM ~1.3 px (box-rebin vs true pixel-integral diverges)
  undersampled_nodither  the theoretical worst case: undersampled AND no dither
  few_lowsnr             N=4, 12x background noise (robustness / failure)
  geometry_only          NO STARRED: naive vs centred crop of a centred truth

Runs entirely in the starred venv:
  ~/venv/starred/bin/python prototypes/starred_drizzle_consistency.py
"""

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "prototypes" / "output" / "starred_drizzle"
S = 2
DATA = 32
SUP = DATA * S
KERNEL = (21, 21)


def _moffat(shape, x0, y0, fwhm, beta=2.5, q=0.85, pa=0.5):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    dx, dy = xx - x0, yy - y0
    c, s = np.cos(pa), np.sin(pa)
    xr, yr = c * dx + s * dy, -s * dx + c * dy
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    r2 = (xr**2 + (yr / q) ** 2) / alpha**2
    return (1 + r2) ** (-beta)


def make_truth_ss(fwhm_data=2.6):
    cy = cx = SUP / 2 - 0.5
    f = fwhm_data * S
    psf = _moffat((SUP, SUP), cx, cy, f)
    psf += 0.06 * _moffat((SUP, SUP), cx + 2.2 * S, cy + 1.1 * S, f * 0.7, beta=3.0, q=1.0)
    return psf / psf.sum()


def _downsample(img, factor):
    from starred.utils.generic_utils import Downsample

    return np.asarray(Downsample(np.asarray(img), factor=factor))


def synth_stars(truth_ss, n, snr_scale, phase_mode, rng):
    from scipy.ndimage import shift as nd_shift

    imgs, noisemaps = [], []
    bkg_sigma = 2e-4 * snr_scale
    fixed = rng.uniform(-S / 2, S / 2, size=2)  # one shared phase for no_dither
    for _ in range(n):
        sy, sx = fixed if phase_mode == "fixed" else rng.uniform(-S / 2, S / 2, size=2)
        shifted = nd_shift(truth_ss, (sy, sx), order=3, mode="constant")
        star = _downsample(shifted, S)
        star = star / star.sum()
        flux = rng.uniform(3e3, 1.5e4)
        signal = flux * star
        var = bkg_sigma**2 * flux**2 + np.clip(signal, 0, None)
        noise = np.sqrt(var)
        imgs.append(signal + rng.normal(0, 1, signal.shape) * noise)
        noisemaps.append(noise)
    return np.array(imgs), np.array(noisemaps)


def _centroid(p):
    """Flux-weighted centroid (row, col), absolute pixel coords."""
    pc = np.clip(p, 0, None)
    ny, nx = pc.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    t = pc.sum()
    return (yy * pc).sum() / t, (xx * pc).sum() / t


def _com_offset(p):
    """Centroid offset from the geometric centre, in px (scalar)."""
    cy, cx = _centroid(p)
    ny, nx = p.shape
    return float(np.hypot(cy - (ny - 1) / 2, cx - (nx - 1) / 2))


def _size(p):
    """Second-moment size sigma-equivalent (px) — continuous, unlike a
    half-max-radius FWHM which quantises to the pixel grid."""
    pc = np.clip(p, 0, None)
    ny, nx = pc.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    t = pc.sum()
    cy, cx = (yy * pc).sum() / t, (xx * pc).sum() / t
    ixx = ((xx - cx) ** 2 * pc).sum() / t
    iyy = ((yy - cy) ** 2 * pc).sum() / t
    return float(np.sqrt(0.5 * (ixx + iyy)))


def _crop_naive(p, shape):
    """The mis-centred delivery: crop around the array-centre pixel."""
    ny, nx = p.shape
    cy, cx = ny // 2, nx // 2
    hy, hx = shape[0] // 2, shape[1] // 2
    cut = p[cy - hy : cy + hy + 1, cx - hx : cx + hx + 1].astype(float)
    return cut / cut.sum()


def _deliver(p, shape):
    """Centroid-preserving delivery (the production-correct route a): crop
    around the measured centroid, then sub-pixel-recentre so the centre of
    light lands on the central pixel of the odd kernel."""
    from scipy.ndimage import shift as nd_shift

    cy, cx = _centroid(p)
    icy, icx = int(round(cy)), int(round(cx))
    hy, hx = shape[0] // 2, shape[1] // 2
    cut = p[icy - hy : icy + hy + 1, icx - hx : icx + hx + 1].astype(float)
    cut = nd_shift(cut, (-(cy - icy), -(cx - icx)), order=3, mode="constant")
    cut = np.clip(cut, 0, None)
    return cut / cut.sum()


def _metrics(rec, truth):
    resid = rec - truth
    return {
        "resid_rms": float(np.sqrt(np.mean(resid**2))),
        "resid_frac_of_peak": float(np.max(np.abs(resid)) / truth.max()),
        "centroid_shift_px": float(
            np.hypot(*(np.subtract(_centroid(rec), _centroid(truth))))
        ),
        "size_ratio": _size(rec) / _size(truth),
        "flux_ratio": float(rec.sum() / truth.sum()),
    }


def geometry_check():
    """No STARRED: does route-a resample of a *centred* truth deliver a centred
    kernel? Reports the naive crop (the ~0.5 px trap) vs the centroid-preserving
    delivery (the fix)."""
    ds = _downsample(make_truth_ss(), S)
    return {
        "naive_crop_centroid_px": _com_offset(_crop_naive(ds, KERNEL)),
        "centred_delivery_centroid_px": _com_offset(_deliver(ds, KERNEL)),
        "note": "resample+crop only, no reconstruction",
    }


SCENARIOS = [
    ("baseline", dict(fwhm=2.6, n=25, snr=1.0, phase="dither")),
    ("no_dither", dict(fwhm=2.6, n=25, snr=1.0, phase="fixed")),
    ("undersampled", dict(fwhm=1.3, n=25, snr=1.0, phase="dither")),
    ("undersampled_nodither", dict(fwhm=1.3, n=25, snr=1.0, phase="fixed")),
    ("few_lowsnr", dict(fwhm=2.6, n=4, snr=12.0, phase="dither")),
]


def run_scenario(cfg):
    from starred.procedures.psf_routines import build_psf

    rng = np.random.default_rng(0)
    truth_ss = make_truth_ss(cfg["fwhm"])
    imgs, noisemaps = synth_stars(truth_ss, cfg["n"], cfg["snr"], cfg["phase"], rng)
    result = build_psf(
        image=imgs, noisemap=noisemaps, subsampling_factor=S,
        n_iter_analytic=40, n_iter_adabelief=600, adjust_sky=True,
    )
    rec_ss = np.asarray(result["full_psf"], dtype=float)
    truth = _deliver(_downsample(truth_ss, S), KERNEL)
    rec = _deliver(_downsample(rec_ss, S), KERNEL)
    return _metrics(rec, truth), truth, rec


def main():
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    g = geometry_check()
    results = {"geometry_only": g}
    print(f"geometry_only  naive_crop={g['naive_crop_centroid_px']:.3f}px  "
          f"centred_delivery={g['centred_delivery_centroid_px']:.3f}px")
    panels = []
    for name, cfg in SCENARIOS:
        m, truth, rec = run_scenario(cfg)
        results[name] = m
        panels.append((name, rec - truth, truth.max()))
        print(f"{name:22s} resid_rms={m['resid_rms']:.2e} "
              f"centroid={m['centroid_shift_px']:.3f}px size={m['size_ratio']:.3f} "
              f"flux={m['flux_ratio']:.3f} peakresid={m['resid_frac_of_peak']:.2%}")
    (OUT / "adversarial_metrics.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(1, len(panels), figsize=(2.7 * len(panels), 3.0))
    for a, (name, resid, vmax) in zip(ax, panels):
        im = a.imshow(resid, origin="lower", cmap="RdBu_r", vmin=-0.05 * vmax, vmax=0.05 * vmax)
        a.set_title(f"{name}\nresid (±5% peak)", fontsize=8)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("STARRED Tier-1b drizzle-consistency — recovered − truth (ground truth)")
    fig.tight_layout()
    fig.savefig(OUT / "adversarial.png", dpi=110)
    print("saved", OUT / "adversarial.png")


if __name__ == "__main__":
    main()
