"""
STARRED vs photutils per-frame ePSF on a real exposure (#41 validation).

Drives the actual frame back-ends (build_frame_epsf / build_starred_frame_epsf)
on a single calibrated exposure and scores each native-pixel ePSF against a
sub-pixel-registered empirical stack of the frame's own stars — the per-frame
analogue of the mosaic comparison (#37/#39). Configured by env:

  FRAME_PATH  calibrated frame FITS (SCI/ERR/DQ MEF)
  FRAME_INST  adapter key (wfc3_uvis | nircam_sw | ...)
  FRAME_RA / FRAME_DEC  field coords (target-exclusion anchor; a corner is fine)
  FRAME_EXT   SCI extver (default 1)

  PYTHONPATH=. ~/venv/PyAuto/bin/python  prototypes/starred_vs_epsf_frame.py epsf
  PYTHONPATH=. ~/venv/starred/bin/python prototypes/starred_vs_epsf_frame.py starred
  ~/venv/PyAuto/bin/python               prototypes/starred_vs_epsf_frame.py compare
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAME = os.environ["FRAME_PATH"]
INST = os.environ.get("FRAME_INST", "wfc3_uvis")
RA = float(os.environ.get("FRAME_RA", "201.69283"))
DEC = float(os.environ.get("FRAME_DEC", "-47.47906"))
EXT = int(os.environ.get("FRAME_EXT", "1"))
TAG = os.environ.get("FRAME_TAG", Path(FRAME).stem)
OUT = ROOT / "prototypes" / "output" / f"starred_vs_epsf_frame_{TAG}"
FIT = 21


def _spec():
    from autoreduce.target import TargetSpec

    return TargetSpec(
        name=TAG, ra=RA, dec=DEC, instrument=INST, filter_name="X", frame_products=True
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


def epsf():
    from astropy.io import fits
    from scipy.ndimage import shift as nd_shift

    from autoreduce import instruments
    from autoreduce.psf.frame_epsf import _prepare_frame, build_frame_epsf

    ad = instruments.get(INST)
    with fits.open(FRAME) as hdul:
        work, err, found, *_ = _prepare_frame(hdul, EXT, _spec(), ad)
        psf_e, _, diag = build_frame_epsf(hdul, EXT, _spec(), ad)
    # empirical stack of the frame's own stars (the native-frame truth PSF)
    h = FIT // 2
    ny, nx = work.shape
    reg = []
    for x, y in zip(
        np.asarray(found["xcentroid"], float), np.asarray(found["ycentroid"], float)
    ):
        ix, iy = int(round(x)), int(round(y))
        if ix - h < 0 or iy - h < 0 or ix + h + 1 > nx or iy + h + 1 > ny:
            continue
        st = work[iy - h : iy + h + 1, ix - h : ix + h + 1]
        if not np.isfinite(st).all():
            continue
        dy, dx = _core_offset(st)
        st = np.clip(nd_shift(st, (-dy, -dx), order=3), 0, None)
        if st.sum() > 0:
            reg.append(st / st.sum())
    stack = np.median(np.array(reg), axis=0)
    stack /= stack.sum()
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "empirical.npy", stack)
    np.save(OUT / "psf_epsf.npy", psf_e / psf_e.sum())
    print(f"epsf: {len(found)} stars, photutils method={diag['method']}, "
          f"empirical stack n={len(reg)} -> {OUT}")


def starred():
    from astropy.io import fits

    from autoreduce import instruments
    from autoreduce.psf.starred_epsf import build_starred_frame_epsf

    ad = instruments.get(INST)
    with fits.open(FRAME) as hdul:
        psf_s, _, diag = build_starred_frame_epsf(hdul, EXT, _spec(), ad)
    np.save(OUT / "psf_starred.npy", psf_s)
    print(f"starred: method={diag['method']}, n_stars={diag.get('n_stars_used')}, "
          f"fwhm={diag.get('sampling_fwhm_px')}, undersampled={diag.get('undersampled')}")


def compare():
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stack = np.load(OUT / "empirical.npy")
    photutils = np.load(OUT / "psf_epsf.npy")
    starred = np.load(OUT / "psf_starred.npy")
    conc = {k: _concentration(v) for k, v in
            [("empirical", stack), ("starred", starred), ("photutils", photutils)]}
    emp = _radial(stack) / _radial(stack)[0]

    def rms(p):
        return float(np.sqrt(np.mean((_radial(p) / _radial(p)[0] - emp) ** 2)))

    metrics = {
        "central_3x3_concentration": conc,
        "radial_rms_vs_empirical": {"starred": rms(starred), "photutils": rms(photutils)},
        "winner": "starred" if rms(starred) < rms(photutils) else "photutils",
    }
    (OUT / "comparison.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    for a, img, t in [
        (ax[0], stack, f"empirical (frame stars)\nconc {conc['empirical']:.2f}"),
        (ax[1], photutils, f"photutils frame\nconc {conc['photutils']:.2f} rms {rms(photutils):.3f}"),
        (ax[2], starred, f"STARRED frame\nconc {conc['starred']:.2f} rms {rms(starred):.3f}"),
    ]:
        im = a.imshow(img, origin="lower"); a.set_title(t, fontsize=9)
        fig.colorbar(im, ax=a, fraction=0.046)
    ax[3].plot(emp, "k-o", label="empirical", ms=3)
    ax[3].plot(_radial(photutils) / _radial(photutils)[0], "b-s", label="photutils", ms=3)
    ax[3].plot(_radial(starred) / _radial(starred)[0], "r-^", label="STARRED", ms=3)
    ax[3].set_yscale("log"); ax[3].set_xlabel("radius (px)")
    ax[3].set_title("radial profile", fontsize=9); ax[3].legend(fontsize=8)
    fig.suptitle(f"{TAG} per-frame ePSF ({INST}): closer-to-empirical wins → {metrics['winner']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=110)
    print(json.dumps(metrics, indent=2))
    print("saved", OUT / "comparison.png")


if __name__ == "__main__":
    {"epsf": epsf, "starred": starred, "compare": compare}[sys.argv[1] if len(sys.argv) > 1 else "epsf"]()
