"""
STARRED Tier-1b vs photutils Tier-1 on a stellar JWST field (#39, JWST leg).

Same head-to-head as the WFC3 leg (#37), now on JWST/NIRCam SW F150W stars:
M92 (NGC 6341), program 1334 (reduce_m92_jwst.py). SW is *undersampled*
(0.03"/pix, PSF FWHM ~1.7 px), the regime the #35 adversarial test flagged for
STARRED broadening — so this is the hard JWST case, and the SW band family the
extragalactic COSMOS-Web F115W could not test (galaxies, not stars).

JWST mosaics are in surface-brightness units (MJy/sr), so there is no full-well
saturation cut (peak_max=None) — the JWST branch of find_stars. Otherwise
identical: concentration + radial profile vs a sub-pixel-registered empirical
star-stack (the true PSF).

  PYTHONPATH=. ~/venv/PyAuto/bin/python  prototypes/starred_vs_epsf_m92.py epsf
  PYTHONPATH=. ~/venv/starred/bin/python prototypes/starred_vs_epsf_m92.py starred
  ~/venv/PyAuto/bin/python               prototypes/starred_vs_epsf_m92.py compare
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_FIELD = os.environ.get("M92_FIELD", "m92_f150w")  # m92_f150w (SW) | m92_f277w (LW)
_CAND = [
    ROOT / "scripts/output" / _FIELD,
    Path.home() / "Code/PyAutoLabs/PyAutoReduce/scripts/output" / _FIELD,
]
FIELD = next((p for p in _CAND if (p / "data.fits").exists()), _CAND[0])
OUT = ROOT / "prototypes" / "output" / f"starred_vs_epsf_{_FIELD}"
NPZ = OUT / "compare.npz"
FIT = 21
PEAK_MAX = (
    None  # JWST surface-brightness units -> no full-well cut (find_stars JWST branch)
)


def epsf():
    from astropy.io import fits

    from autoreduce.psf.epsf import build_epsf
    from autoreduce.psf.stars import StarSelection, find_stars

    sci = fits.getdata(FIELD / "data.fits").astype(float)
    noise = fits.getdata(FIELD / "noise_map.fits").astype(float)
    ny, nx = sci.shape
    # M92 SW is very crowded; relax the isolation radius to 13 px (0.4") to get
    # a workable sample (mild blending is exactly where STARRED's deconvolution
    # should beat photutils' star-averaging).
    sel = StarSelection(
        detection_sigma=12.0,
        sharp_range=(0.3, 1.0),
        round_limit=0.4,
        min_separation_pix=float(
            os.environ.get("M92_MINSEP", "13")
        ),  # SW 13px / LW 18px
        exclusion_radius_pix=0.0,
    )
    stars = find_stars(sci, sel, (nx / 2, ny / 2), peak_max=PEAK_MAX)
    try:
        build_epsf(sci, stars, (21, 21), (61, 61))
        psf_full_ok = True
    except Exception:
        psf_full_ok = False
    psf_epsf, _, diag = build_epsf(sci, stars, (21, 21), (21, 21))
    print(
        f"epsf: photutils psf_full(61x61) build {'OK' if psf_full_ok else 'FAILED (non-positive flux)'}"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        NPZ,
        sci=sci,
        noise=noise,
        x=np.asarray(stars["xcentroid"], float),
        y=np.asarray(stars["ycentroid"], float),
        psf_epsf=psf_epsf,
    )
    print(
        f"epsf: {len(stars)} stars, photutils Tier-1 built ({diag['n_stars_used']} used) -> {NPZ}"
    )


def starred():
    from astropy.table import Table

    from autoreduce.psf.starred_epsf import build_starred_epsf

    d = np.load(NPZ)
    tab = Table({"xcentroid": d["x"], "ycentroid": d["y"]})
    psf_starred, _, diag = build_starred_epsf(
        d["sci"], d["noise"], tab, (21, 21), (61, 61)
    )
    np.save(OUT / "psf_starred.npy", psf_starred)
    print(
        f"starred: Tier-1b built, {diag['n_stars_used']} stars, "
        f"centroid_residual {diag['centroid_residual_px']:.3f}px, fwhm {diag['sampling_fwhm_px']:.2f}px, "
        f"undersampled={diag['undersampled']}"
    )


def _core_offset(stamp):
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
    iy, ix = np.unravel_index(int(np.argmax(p)), p.shape)
    return float(p[iy - 1 : iy + 2, ix - 1 : ix + 2].sum() / p.sum())


def _radial(p):
    yy, xx = np.mgrid[0:FIT, 0:FIT]
    r = np.hypot(yy - (FIT - 1) / 2, xx - (FIT - 1) / 2).astype(int)
    return np.array([p[r == k].mean() for k in range(8)])


def compare():
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    photutils = d["psf_epsf"] / d["psf_epsf"].sum()

    conc = {
        k: _concentration(v)
        for k, v in [
            ("empirical_stack", stack),
            ("starred", starred),
            ("photutils", photutils),
        ]
    }
    emp = _radial(stack) / _radial(stack)[0]

    def rms_vs_emp(p):
        return float(np.sqrt(np.mean((_radial(p) / _radial(p)[0] - emp) ** 2)))

    metrics = {
        "n_stars": len(reg),
        "central_3x3_concentration": conc,
        "empirical_is_stellar": conc["empirical_stack"] >= 0.30,
        "radial_rms_vs_empirical": {
            "starred": rms_vs_emp(starred),
            "photutils": rms_vs_emp(photutils),
        },
        "winner": (
            "starred" if rms_vs_emp(starred) < rms_vs_emp(photutils) else "photutils"
        ),
    }
    (OUT / "comparison.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    for a, img, t in [
        (
            ax[0],
            stack,
            f"empirical stack (n={len(reg)})\nconc {conc['empirical_stack']:.2f}",
        ),
        (
            ax[1],
            photutils,
            f"photutils Tier-1\nconc {conc['photutils']:.2f}  rms {rms_vs_emp(photutils):.3f}",
        ),
        (
            ax[2],
            starred,
            f"STARRED Tier-1b\nconc {conc['starred']:.2f}  rms {rms_vs_emp(starred):.3f}",
        ),
    ]:
        im = a.imshow(img, origin="lower")
        a.set_title(t, fontsize=9)
        fig.colorbar(im, ax=a, fraction=0.046)
    ax[3].plot(emp, "k-o", label="empirical", ms=3)
    ax[3].plot(
        _radial(photutils) / _radial(photutils)[0], "b-s", label="photutils", ms=3
    )
    ax[3].plot(_radial(starred) / _radial(starred)[0], "r-^", label="STARRED", ms=3)
    ax[3].set_yscale("log")
    ax[3].set_xlabel("radius (px)")
    ax[3].set_title("radial profile", fontsize=9)
    ax[3].legend(fontsize=8)
    fig.suptitle(
        f"{_FIELD} (STELLAR): closer-to-empirical wins → {metrics['winner']}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=110)
    print(json.dumps(metrics, indent=2))
    print("saved", OUT / "comparison.png")


if __name__ == "__main__":
    {"epsf": epsf, "starred": starred, "compare": compare}[
        sys.argv[1] if len(sys.argv) > 1 else "epsf"
    ]()
