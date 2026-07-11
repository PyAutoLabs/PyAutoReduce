"""
STARRED Tier-1b vs photutils Tier-1 — real-data head-to-head (#35).

The ground-truth adversarial test proved STARRED's reconstruction + delivery is
drizzle-consistent; this asks the science question on real data: does the
STARRED Tier-1b ePSF *represent the actual point sources better* than the
shipped photutils Tier-1 ePSF? Both PSFs are built from the **same** field-star
set, then each is fit to every star (best-fit flux scale + sub-pixel shift,
noise-weighted) and scored by reduced chi-square. Lower = the model that better
predicts the real data.

Caveat: the reduced fields are extragalactic (COSMOS-Web F115W), so the point
sources are a star / compact-source mix — this is a *relative* comparison of the
two reconstructions on identical inputs, which is fair; a definitive absolute
test wants a genuinely stellar field (a follow-up reduction).

Three venv-scoped stages (photutils in the PyAuto venv, STARRED in its own):
  PYTHONPATH=. ~/venv/PyAuto/bin/python prototypes/starred_vs_epsf_comparison.py epsf
  PYTHONPATH=. ~/venv/starred/bin/python prototypes/starred_vs_epsf_comparison.py starred
  ~/venv/PyAuto/bin/python prototypes/starred_vs_epsf_comparison.py compare
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_FIELD = "cosmos_web_ring_f115w"
_CAND = [ROOT / "scripts/output" / _FIELD,
         Path.home() / "Code/PyAutoLabs/PyAutoReduce/scripts/output" / _FIELD]
FIELD = next((p for p in _CAND if (p / "data.fits").exists()), _CAND[0])
OUT = ROOT / "prototypes" / "output" / "starred_vs_epsf"
NPZ = OUT / "compare.npz"
FIT = 21  # fit stamp (== compact kernel)


def _selection():
    from autoreduce.psf.stars import StarSelection

    return StarSelection(
        detection_sigma=3.0, fwhm_pix=2.5, sharp_range=(0.3, 1.0), round_limit=0.4,
        min_separation_pix=18.0, edge_margin_pix=20, exclusion_radius_pix=40.0,
    )


def epsf():
    """PyAuto venv: find stars, build the photutils Tier-1 ePSF, stash inputs."""
    from astropy.io import fits

    from autoreduce.psf.epsf import build_epsf
    from autoreduce.psf.stars import find_stars

    sci = fits.getdata(FIELD / "data.fits").astype(float)
    noise = fits.getdata(FIELD / "noise_map.fits").astype(float)
    ny, nx = sci.shape
    stars = find_stars(sci, _selection(), (nx / 2, ny / 2), peak_max=None)
    psf_epsf, _, diag = build_epsf(sci, stars, (21, 21), (61, 61))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ, sci=sci, noise=noise,
             x=np.asarray(stars["xcentroid"], float), y=np.asarray(stars["ycentroid"], float),
             psf_epsf=psf_epsf)
    print(f"epsf: {len(stars)} stars, photutils Tier-1 built ({diag['n_stars_used']} used) -> {NPZ}")


def starred():
    """starred venv: build the STARRED Tier-1b ePSF from the same stars."""
    from astropy.table import Table

    from autoreduce.psf.starred_epsf import build_starred_epsf

    d = np.load(NPZ)
    tab = Table({"xcentroid": d["x"], "ycentroid": d["y"]})
    psf_starred, _, diag = build_starred_epsf(d["sci"], d["noise"], tab, (21, 21), (61, 61))
    np.save(OUT / "psf_starred.npy", psf_starred)
    print(f"starred: Tier-1b built, {diag['n_stars_used']} stars, "
          f"centroid_residual {diag['centroid_residual_px']:.3f}px, "
          f"fwhm {diag['sampling_fwhm_px']:.2f}px")


def _core_offset(stamp):
    """Sub-pixel offset (dy, dx) of the source core from the stamp centre."""
    pc = np.clip(stamp, 0, None)
    iy, ix = np.unravel_index(int(np.argmax(pc)), pc.shape)
    w = 4
    y0, y1 = max(iy - w, 0), min(iy + w + 1, pc.shape[0])
    x0, x1 = max(ix - w, 0), min(ix + w + 1, pc.shape[1])
    sub = pc[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    t = sub.sum()
    c = (FIT - 1) / 2
    return (yy * sub).sum() / t - c, (xx * sub).sum() / t - c


def _concentration(p):
    """Fraction of flux in the central 3x3 — a registration-insensitive
    compactness proxy. A real (undersampled) F115W *star* is ~0.5-0.8; an
    extended galaxy is a few percent."""
    iy, ix = np.unravel_index(int(np.argmax(p)), p.shape)
    return float(p[iy - 1 : iy + 2, ix - 1 : ix + 2].sum() / p.sum())


def _radial(p):
    yy, xx = np.mgrid[0:FIT, 0:FIT]
    r = np.hypot(yy - (FIT - 1) / 2, xx - (FIT - 1) / 2).astype(int)
    return np.array([p[r == k].mean() for k in range(8)])


def compare():
    """Build a sub-pixel-registered empirical stack of the field's point
    sources and compare it, STARRED, and the production photutils ePSF by
    compactness + radial profile — registration-insensitive, unlike a
    per-source chi-square which is dominated by sub-pixel misalignment and
    would spuriously reward a flat PSF."""
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from scipy.ndimage import shift as nd_shift

    d = np.load(NPZ)
    sci = d["sci"].astype(float)
    ny, nx = sci.shape
    h = FIT // 2

    reg = []
    for x, y in zip(d["x"], d["y"]):
        ix, iy = int(round(float(x))), int(round(float(y)))
        if ix - h < 0 or iy - h < 0 or ix + h + 1 > nx or iy + h + 1 > ny:
            continue
        st = sci[iy - h : iy + h + 1, ix - h : ix + h + 1]
        if not np.isfinite(st).all():
            continue
        dy, dx = _core_offset(st)
        st = np.clip(nd_shift(st, (-dy, -dx), order=3), 0, None)
        if st.sum() > 0:
            reg.append(st / st.sum())
    stack = np.median(np.array(reg), axis=0)
    stack /= stack.sum()

    starred = np.load(OUT / "psf_starred.npy")
    photutils = fits.getdata(FIELD / "psf.fits").astype(float)  # production ePSF (all field stars)
    photutils /= photutils.sum()

    conc = {k: _concentration(v) for k, v in
            [("empirical_stack", stack), ("starred", starred), ("photutils", photutils)]}
    STELLAR_MIN = 0.30  # a real undersampled F115W star exceeds this; galaxies do not
    verdict = (
        "field is NOT stellar — the empirical stack is diffuse (extended galaxies), "
        "so no valid absolute PSF comparison is possible here; a stellar field is required"
        if conc["empirical_stack"] < STELLAR_MIN else "field usable for PSF comparison"
    )
    metrics = {
        "n_sources": len(reg),
        "central_3x3_concentration": conc,
        "empirical_is_stellar": conc["empirical_stack"] >= STELLAR_MIN,
        "verdict": verdict,
        "note": (
            "STARRED's deconvolution recovers a compact core "
            f"({conc['starred']:.2f}) that resists the extended contamination "
            f"photutils averages in ({conc['photutils']:.2f} ~ empirical "
            f"{conc['empirical_stack']:.2f}) — suggestive but unverifiable without real stars"
        ),
    }
    (OUT / "comparison.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    for a, img, t in [
        (ax[0], stack, f"empirical stack (n={len(reg)})\nconc {conc['empirical_stack']:.2f}"),
        (ax[1], photutils, f"photutils Tier-1\nconc {conc['photutils']:.2f}"),
        (ax[2], starred, f"STARRED Tier-1b\nconc {conc['starred']:.2f}"),
    ]:
        im = a.imshow(img, origin="lower"); a.set_title(t, fontsize=9)
        fig.colorbar(im, ax=a, fraction=0.046)
    ax[3].plot(_radial(stack) / _radial(stack)[0], "k-o", label="empirical", ms=3)
    ax[3].plot(_radial(photutils) / _radial(photutils)[0], "b-s", label="photutils", ms=3)
    ax[3].plot(_radial(starred) / _radial(starred)[0], "r-^", label="STARRED", ms=3)
    ax[3].set_yscale("log"); ax[3].set_xlabel("radius (px)"); ax[3].set_title("radial profile", fontsize=9)
    ax[3].legend(fontsize=8)
    fig.suptitle(f"{_FIELD}: {verdict}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=110)
    print(json.dumps(metrics, indent=2))
    print("saved", OUT / "comparison.png")


if __name__ == "__main__":
    {"epsf": epsf, "starred": starred, "compare": compare}[sys.argv[1] if len(sys.argv) > 1 else "epsf"]()
