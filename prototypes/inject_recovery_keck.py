"""
Keck injection-recovery spike (issue #54): phase 2b's real-data
validation on the B1938+666 anchor (docs/design/keck_ao.md), including
the re-registration check the design promised — measured offsets with
and without injection must agree (consistently-placed injected content
reinforces, never biases, the phase-correlation registration).

Requires a spec YAML with the B1938+666 KOA identifiers (koa_science_ids
etc.), a tier-A PSF candidate FITS to use as inject_psf, and KOA archive
access. Same clean-vs-injected difference-image shape as the HST/JWST
spikes; flux in e-/s.

Run:  ~/venv/PyAuto/bin/python prototypes/inject_recovery_keck.py \
          --spec <b1938_spec.yaml> --inject-psf <psf_candidate.fits> \
          [--cache prototypes/cache_inject_keck] [--offset-arcsec 1.0] \
          [--flux-eps 200.0]
Unit tests never import this.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from autoreduce.pipeline import reduce_target
from autoreduce.target import TargetSpec


def build_input(out_dir: Path, flux_eps: float, fwhm_arcsec: float = 0.08,
                pixel_scale: float = 0.005, shape=(81, 81)) -> Path:
    from astropy.io import fits

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    cy, cx = shape[0] // 2, shape[1] // 2
    sigma = fwhm_arcsec / 2.3548 / pixel_scale
    img = np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / sigma**2)
    img = flux_eps * img / img.sum()
    path = out_dir / "input_gaussian_eps.fits"
    fits.PrimaryHDU(img.astype(np.float32)).writeto(path, overwrite=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="B1938+666 TargetSpec YAML")
    parser.add_argument("--inject-psf", required=True)
    parser.add_argument("--cache", default="prototypes/cache_inject_keck")
    parser.add_argument("--offset-arcsec", type=float, default=1.0)
    parser.add_argument("--flux-eps", type=float, default=200.0)
    args = parser.parse_args()

    args.cache = str(Path(args.cache).resolve())
    out_root = Path("prototypes/output/inject_recovery_keck").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    base = TargetSpec.from_yaml(args.spec)
    input_path = build_input(out_root, args.flux_eps)

    print("== clean reduction ==")
    clean = reduce_target(base, Path(args.cache), out_root / "clean")
    print("== injected reduction ==")
    injected = reduce_target(
        replace(
            base,
            inject_image=str(input_path),
            inject_pixel_scale=0.005,
            inject_position=(
                base.ra + args.offset_arcsec / 3600.0, base.dec
            ),
            inject_psf=str(Path(args.inject_psf).resolve()),
        ),
        Path(args.cache),
        out_root / "injected",
    )

    from astropy.io import fits

    name = base.name
    data_clean = fits.getdata(out_root / "clean" / name / "data.fits")
    data_inj = fits.getdata(out_root / "injected" / name / "data.fits")
    noise_inj = fits.getdata(out_root / "injected" / name / "noise_map.fits")

    diff = data_inj - data_clean
    # Find the injected source empirically (placement is offset-based).
    peak = np.unravel_index(np.nanargmax(diff), diff.shape)
    yy, xx = np.mgrid[0 : diff.shape[0], 0 : diff.shape[1]]
    pixel_scale = injected["package"]["pixel_scale"]
    aperture = np.hypot(yy - peak[0], xx - peak[1]) * pixel_scale <= 0.5
    recovered = float(np.nansum(diff[aperture]))
    noise_pred = float(np.sqrt(np.nansum(noise_inj[aperture] ** 2)))

    # Re-registration check: the two runs' measured offsets must agree.
    off_clean = clean["drizzle"].get("registration_offsets_native_pix")
    off_inj = injected["drizzle"].get("registration_offsets_native_pix")
    max_shift = (
        float(
            np.max(np.abs(np.asarray(off_clean) - np.asarray(off_inj)))
        )
        if off_clean is not None and off_inj is not None
        else None
    )

    report = {
        "injected_flux_eps": args.flux_eps,
        "recovered_flux_eps_0p5arcsec": recovered,
        "recovery_ratio": recovered / args.flux_eps,
        "aperture_noise_eps": noise_pred,
        "registration_max_offset_shift_pix": max_shift,
        "total_injected_e": injected["inject"]["total_injected_e"],
        "n_frames": len(injected["inject"]["frames"]),
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    ok = abs(report["recovery_ratio"] - 1.0) < max(
        0.05, 3.0 * noise_pred / args.flux_eps
    ) and (max_shift is None or max_shift < 0.1)
    print("RECOVERY", "OK" if ok else "DISCREPANT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
