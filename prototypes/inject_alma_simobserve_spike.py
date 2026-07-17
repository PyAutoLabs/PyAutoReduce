"""
simobserve spike (issue #56): phase 3's real-casatasks validation — a
Sersic-like Jy/pixel input through simobserve into the full visibility
branch, ending at the `al.Interferometer.from_fits` product triplet.

Checks: products exist and are finite; n_visibilities > 0; the recovered
total flux (real part of the shortest-baseline visibilities approximates
the total source flux for a compact source) is within tolerance of the
input; provenance carries the inject block.

Run:  ~/venv/PyAuto/bin/python prototypes/inject_alma_simobserve_spike.py \
          [--flux-jy 0.02] [--totaltime-s 600]
casatasks + casatools required; unit tests never import this.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from autoreduce.pipeline import reduce_target
from autoreduce.target import TargetSpec

RA, DEC = 137.0, 2.1  # arbitrary southern-sky-visible field


def build_input(out_dir: Path, flux_jy: float, r_eff_arcsec: float = 0.15,
                pixel_scale: float = 0.02, shape=(129, 129)) -> Path:
    from astropy.io import fits

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    cy, cx = shape[0] // 2, shape[1] // 2
    r = np.hypot(yy - cy, xx - cx) * pixel_scale
    img = np.exp(-1.678 * r / r_eff_arcsec)
    img = flux_jy * img / img.sum()
    path = out_dir / "input_sersic_jy.fits"
    fits.PrimaryHDU(img.astype(np.float32)).writeto(path, overwrite=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flux-jy", type=float, default=0.02)
    parser.add_argument("--totaltime-s", type=float, default=600.0)
    args = parser.parse_args()

    out_root = Path("prototypes/output/inject_alma_simobserve").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    input_path = build_input(out_root, args.flux_jy)

    record = reduce_target(
        TargetSpec(
            name="alma_sim_spike",
            ra=RA,
            dec=DEC,
            instrument="alma",
            inject_image=str(input_path),
            inject_pixel_scale=0.02,
            alma_sim_totaltime_s=args.totaltime_s,
        ),
        out_root / "cache",
        out_root,
    )

    from astropy.io import fits

    prod_dir = out_root / "alma_sim_spike"
    vis = fits.getdata(prod_dir / "visibilities.fits")
    uv = fits.getdata(prod_dir / "uv_wavelengths.fits")
    noise = fits.getdata(prod_dir / "noise_map.fits")

    # Compact source: the real part at the shortest baselines approaches
    # the total flux.
    uv_dist = np.hypot(uv[:, 0], uv[:, 1])
    short = uv_dist < np.percentile(uv_dist, 5)
    recovered = float(np.mean(vis[short, 0]))

    report = {
        "input_flux_jy": args.flux_jy,
        "short_baseline_mean_real_jy": recovered,
        "recovery_ratio": recovered / args.flux_jy,
        "n_visibilities": int(vis.shape[0]),
        "noise_finite_fraction": float(np.isfinite(noise).mean()),
        "thermal_noise": record["inject"]["thermal_noise"],
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    ok = (
        report["n_visibilities"] > 0
        and report["noise_finite_fraction"] == 1.0
        and abs(report["recovery_ratio"] - 1.0) < 0.15
    )
    print("SIMOBSERVE", "OK" if ok else "DISCREPANT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
