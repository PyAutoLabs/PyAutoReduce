"""
Injection-recovery spike (issue #46): the inject stage's real-data
validation on the slacs0008 ACS/F814W field (docs/design/simulate.md,
phase 1).

Injects a Sersic-like synthetic blob offset from the target into the real
_flc exposures, runs the full unmodified pipeline (astrodrizzle, driz_cr,
noise, psf, package), then measures the recovered flux in the packaged
cutout against the injected truth — parity-appendix style.

  phase 1  build the input image (circular Sersic n=1, e-/s pixels) and
           write it under prototypes/output/inject_recovery/.
  phase 2  reduce twice with a shared cache: once clean, once injected
           (same spec otherwise), so the recovery measurement is a
           difference image of two identically-processed datasets.
  phase 3  report: injected flux vs (injected - clean) cutout sum inside
           a 3" aperture at the injection position, plus the noise-map
           quadrature prediction; JSON + printout.

Run:  ~/venv/PyAuto/bin/python prototypes/inject_recovery_slacs.py \
          [--cache prototypes/cache] [--offset-arcsec 4.0] [--flux-cps 30.0]
drizzlepac + archive access required; unit tests never import this.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from autoreduce.pipeline import reduce_target
from autoreduce.target import TargetSpec

# SDSS J0008-0004 — the phase-1/parity anchor (hst_acs_pipeline.md,
# prototypes/slacs_f814w_spike.py).
NAME = "slacs0008-0004"
RA, DEC = 2.012333, -0.068944


def build_input(out_dir: Path, flux_cps: float, r_eff_arcsec: float = 0.4,
                pixel_scale: float = 0.025, shape=(121, 121)) -> Path:
    from astropy.io import fits

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    cy, cx = shape[0] // 2, shape[1] // 2
    r = np.hypot(yy - cy, xx - cx) * pixel_scale
    img = np.exp(-1.678 * r / r_eff_arcsec)  # Sersic n=1
    img = flux_cps * img / img.sum()
    path = out_dir / "input_sersic.fits"
    fits.PrimaryHDU(img.astype(np.float32)).writeto(path, overwrite=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="prototypes/cache")
    parser.add_argument("--offset-arcsec", type=float, default=4.0)
    parser.add_argument("--flux-cps", type=float, default=30.0)
    args = parser.parse_args()

    out_root = Path("prototypes/output/inject_recovery")
    out_root.mkdir(parents=True, exist_ok=True)
    input_path = build_input(out_root, args.flux_cps)
    inject_ra = RA + args.offset_arcsec / 3600.0
    common = dict(name=NAME, ra=RA, dec=DEC, cutout_shape=(281, 281))

    print("== phase 2a: clean reduction ==")
    clean = reduce_target(
        TargetSpec(**common), Path(args.cache), out_root / "clean"
    )
    print("== phase 2b: injected reduction ==")
    injected = reduce_target(
        TargetSpec(
            **common,
            inject_image=str(input_path),
            inject_pixel_scale=0.025,
            inject_position=(inject_ra, DEC),
        ),
        Path(args.cache),
        out_root / "injected",
    )

    from astropy.io import fits
    from astropy.wcs import WCS

    data_clean = fits.getdata(out_root / "clean" / NAME / "data.fits")
    hdr = fits.getheader(out_root / "injected" / NAME / "data.fits")
    data_inj = fits.getdata(out_root / "injected" / NAME / "data.fits")
    noise_inj = fits.getdata(out_root / "injected" / NAME / "noise_map.fits")

    diff = data_inj - data_clean
    xy = WCS(hdr).world_to_pixel_values(inject_ra, DEC)
    yy, xx = np.mgrid[0 : diff.shape[0], 0 : diff.shape[1]]
    pixel_scale = injected["package"]["pixel_scale"]
    aperture = np.hypot(yy - xy[1], xx - xy[0]) * pixel_scale <= 3.0
    recovered = float(diff[aperture].sum())
    noise_pred = float(np.sqrt((noise_inj[aperture] ** 2).sum()))

    report = {
        "injected_flux_cps": args.flux_cps,
        "recovered_flux_cps_3arcsec": recovered,
        "recovery_ratio": recovered / args.flux_cps,
        "aperture_noise_cps": noise_pred,
        "total_injected_e": injected["inject"]["total_injected_e"],
        "n_frames": len(injected["inject"]["frames"]),
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    ok = abs(report["recovery_ratio"] - 1.0) < max(
        0.05, 3.0 * noise_pred / args.flux_cps
    )
    print("RECOVERY", "OK" if ok else "DISCREPANT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
