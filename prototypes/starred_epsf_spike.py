"""
STARRED Tier-1b ePSF feasibility spike (PyAutoReduce#35).

Resolves the one open technical question the design left: is STARRED's
super-sampled PSF-from-field-stars a viable, *drizzle-consistent* alternative to
the photutils Tier-1 ePSF? Runs on an already-reduced, star-rich mosaic
(COSMOS-Web ring F150W: 96 stars used by Tier-1, 0.03"/px) so the only new
variable is the PSF back-end, and Tier-1's `psf.fits` is the comparison baseline
— no download, no new reduction.

Two modes, one per venv (this mirrors the production GPL/JAX isolation — the
pipeline selects stars with photutils, then hands cutouts to the STARRED
back-end that lives behind an optional, separately-installed extra):

  # star selection + cutout extraction — PyAuto venv, from the repo root
  PYTHONPATH=. ~/venv/PyAuto/bin/python prototypes/starred_epsf_spike.py extract

  # STARRED reconstruction + downsample-to-mosaic-grid — starred venv
  ~/venv/starred/bin/python prototypes/starred_epsf_spike.py reconstruct

  # compare STARRED-Tier1b vs photutils-Tier1 on a residual metric (any venv)
  ~/venv/PyAuto/bin/python prototypes/starred_epsf_spike.py compare

Handoff file: prototypes/output/starred_spike/spike.npz (+ compare.png/json).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Reduced outputs are gitignored and live only in the canonical checkout, not
# in the task worktree — resolve the field there, worktree-first.
_FIELD_NAME = "cosmos_web_ring_f115w"  # richest clean-point-source count of the reduced fields
_CANDIDATES = [
    ROOT / "scripts" / "output" / _FIELD_NAME,
    Path.home() / "Code/PyAutoLabs/PyAutoReduce/scripts/output" / _FIELD_NAME,
]
FIELD = next((p for p in _CANDIDATES if (p / "data.fits").exists()), _CANDIDATES[0])
OUT = ROOT / "prototypes" / "output" / "starred_spike"
NPZ = OUT / "spike.npz"

N_STARS = 20        # brightest clean stars — plenty for STARRED (HST example: 6)
CUTOUT = 32         # even stamp; STARRED reconstructs on this grid
SUBSAMPLE = 2       # super-sampling factor of the reconstructed PSF
PSF_SHAPE = (21, 21)


def _crop_norm(psf, shape):
    """Centre-crop to odd `shape` and unit-normalise (mirrors epsf.normalise_kernel)."""
    ny, nx = psf.shape
    cy, cx = ny // 2, nx // 2
    hy, hx = shape[0] // 2, shape[1] // 2
    cut = psf[cy - hy : cy + hy + 1, cx - hx : cx + hx + 1].astype(np.float64)
    return cut / cut.sum()


def extract():
    """Select field stars + extract (N, CUTOUT, CUTOUT) data/noise cutouts."""
    from astropy.io import fits

    from autoreduce.psf.stars import StarSelection, find_stars

    sci = fits.getdata(FIELD / "data.fits").astype(float)
    noise = fits.getdata(FIELD / "noise_map.fits").astype(float)
    tier1 = fits.getdata(FIELD / "psf.fits").astype(float)
    ny, nx = sci.shape
    target_xy = (nx / 2.0, ny / 2.0)  # ring at mosaic centre — excluded as a "star"

    # Spike-relaxed cuts: these fields are extragalactic (a star/compact-source
    # mix), so this yields a mechanics-validation ePSF, not a science-grade one.
    selection = StarSelection(
        detection_sigma=3.0,
        fwhm_pix=2.5,
        sharp_range=(0.3, 1.0),
        round_limit=0.4,
        min_separation_pix=18.0,
        edge_margin_pix=20,
        exclusion_radius_pix=40.0,
    )
    sources = find_stars(sci, selection, target_xy, peak_max=None)
    assert sources is not None and len(sources) >= 6, "field not star-rich enough"
    order = np.argsort(np.asarray(sources["flux"]))[::-1]  # brightest first

    h = CUTOUT // 2
    cut_sci, cut_noise = [], []
    for i in order:
        x = int(round(float(sources["xcentroid"][i])))
        y = int(round(float(sources["ycentroid"][i])))
        if x - h < 0 or y - h < 0 or x + h > nx or y + h > ny:
            continue
        s = sci[y - h : y + h, x - h : x + h]
        n = noise[y - h : y + h, x - h : x + h]
        if not (np.isfinite(s).all() and np.isfinite(n).all() and (n > 0).all()):
            continue
        cut_sci.append(s)
        cut_noise.append(n)
        if len(cut_sci) >= N_STARS:
            break

    cut_sci = np.array(cut_sci)
    cut_noise = np.array(cut_noise)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        NPZ,
        cutouts=cut_sci,
        noisemaps=cut_noise,
        tier1_psf=tier1,
        subsample=SUBSAMPLE,
    )
    print(f"extract: {len(cut_sci)} star cutouts {cut_sci.shape} -> {NPZ}")


def reconstruct():
    """Run STARRED build_psf, then downsample the full PSF to the mosaic grid."""
    from starred.procedures.psf_routines import build_psf
    from starred.utils.generic_utils import Downsample

    d = np.load(NPZ)
    cutouts, noisemaps, sub = d["cutouts"], d["noisemaps"], int(d["subsample"])

    result = build_psf(
        image=np.asarray(cutouts, dtype=float),
        noisemap=np.asarray(noisemaps, dtype=float),
        subsampling_factor=sub,
        n_iter_analytic=40,
        n_iter_adabelief=500,   # trimmed for the spike; production would raise
        adjust_sky=True,
    )
    print("reconstruct: result keys =", sorted(result))
    full = np.asarray(result["full_psf"], dtype=float)        # observed PSF, super-sampled
    narrow = np.asarray(result["narrow_psf"], dtype=float)
    print(f"  full_psf (super-sampled) {full.shape}, narrow {narrow.shape}, sub={sub}")

    # Drizzle-consistency, candidate route (a): block-rebin the super-sampled
    # observed PSF down by the subsampling factor onto the mosaic pixel grid.
    on_grid = np.asarray(Downsample(full, factor=sub), dtype=float)
    starred_psf = _crop_norm(on_grid, PSF_SHAPE)
    np.save(OUT / "starred_full_supersampled.npy", full)
    np.save(OUT / "starred_psf_mosaicgrid.npy", starred_psf)
    print(f"  downsampled->mosaic grid {on_grid.shape}, cropped {starred_psf.shape} -> saved")


def compare():
    """Residual metric + overlay of STARRED-Tier1b vs photutils-Tier1."""
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(NPZ)
    tier1 = _crop_norm(np.asarray(d["tier1_psf"], dtype=float), PSF_SHAPE)
    starred = np.load(OUT / "starred_psf_mosaicgrid.npy")

    def fwhm(p):
        ny, nx = p.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        r = np.hypot(yy - ny // 2, xx - nx // 2)
        above = r[p >= p.max() / 2.0]
        return float(2 * above.max()) if above.size else float("nan")

    resid = starred - tier1
    metrics = {
        "resid_rms": float(np.sqrt(np.mean(resid**2))),
        "resid_max_abs": float(np.max(np.abs(resid))),
        "tier1_fwhm_pix": fwhm(tier1),
        "starred_fwhm_pix": fwhm(starred),
        "peak_ratio_starred_over_tier1": float(starred.max() / tier1.max()),
    }
    (OUT / "compare.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a, img, t in [
        (ax[0], tier1, "photutils Tier-1"),
        (ax[1], starred, "STARRED Tier-1b"),
        (ax[2], resid, "residual"),
    ]:
        im = a.imshow(img, origin="lower")
        a.set_title(t)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"{_FIELD_NAME} ePSF — resid RMS {metrics['resid_rms']:.2e}")
    fig.tight_layout()
    fig.savefig(OUT / "compare.png", dpi=110)
    print("compare:", json.dumps(metrics, indent=2))
    print("saved", OUT / "compare.png")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "extract"
    {"extract": extract, "reconstruct": reconstruct, "compare": compare}[mode]()
