"""
STARRED Tier-1b vs photutils Tier-1 on a genuinely STELLAR field (#37, WFC3 leg).

The F115W comparison (#35) was inconclusive because the field's "point sources"
were extended galaxies. This runs the same head-to-head on real STARS: Omega Cen
(NGC 5139) WFC3/UVIS F606W (reduce_omegacen_wfc3.py). With real stars the
empirical sub-pixel-registered stack IS the PSF, so concentration + radial
profile are meaningful truth references.

Strict stellar star-selection (point-like sharp/round cuts) + a saturation cut
(60s F606W saturates the brightest Omega Cen stars). Three venv-scoped stages:
  PYTHONPATH=. ~/venv/PyAuto/bin/python  prototypes/starred_vs_epsf_omegacen.py epsf
  PYTHONPATH=. ~/venv/starred/bin/python prototypes/starred_vs_epsf_omegacen.py starred
  ~/venv/PyAuto/bin/python               prototypes/starred_vs_epsf_omegacen.py compare
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_FIELD = "omegacen_f606w"
_CAND = [
    ROOT / "scripts/output" / _FIELD,
    Path.home() / "Code/PyAutoLabs/PyAutoReduce/scripts/output" / _FIELD,
]
FIELD = next((p for p in _CAND if (p / "data.fits").exists()), _CAND[0])
OUT = ROOT / "prototypes" / "output" / "starred_vs_epsf_omegacen"
NPZ = OUT / "compare.npz"
FIT = 21
# 60s F606W saturation ceiling (e-/s): saturation_fraction * saturation_dn / exptime.
PEAK_MAX = 0.7 * 63_000.0 / 60.0


def epsf():
    from astropy.io import fits

    from autoreduce.psf.epsf import build_epsf
    from autoreduce.psf.stars import StarSelection, find_stars

    sci = fits.getdata(FIELD / "data.fits").astype(float)
    noise = fits.getdata(FIELD / "noise_map.fits").astype(float)
    ny, nx = sci.shape
    # Strict, stellar-appropriate cuts (defaults are tuned for point sources);
    # the target is the cluster field itself, so no lens/target exclusion.
    stars = find_stars(
        sci,
        StarSelection(exclusion_radius_pix=0.0),
        (nx / 2, ny / 2),
        peak_max=PEAK_MAX,
    )
    # FINDING: photutils build_epsf with the shipped 61x61 psf_full raises
    # "non-positive total flux" on this real WFC3/F606W field (negative ePSF
    # wings from background over-subtraction) — the Tier-1 path cannot deliver
    # the extended kernel here. STARRED delivers both (see starred()). The
    # compact 21x21 core builds fine and is what the comparison scores.
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
        f"epsf: {len(stars)} stars selected, photutils Tier-1 built ({diag['n_stars_used']} used) -> {NPZ}"
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
