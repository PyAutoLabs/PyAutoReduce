"""
JWST injection-recovery spike (issue #52): phase 2a's real-data
validation on the COSMOS-Web ring field, F150W (docs/design/simulate.md).

Same shape as the HST spike (`inject_recovery_slacs.py`): inject a
Sersic-like blob offset from the target into the real _cal exposures
(input in **Jy per pixel** — the JWST inject contract), reduce clean and
injected with a shared cache, difference the packaged cutouts, compare
recovered flux to truth against the noise-map prediction. Recovered
surface brightness integrates to flux via the mosaic pixel area.

Run:  ~/venv/PyAuto/bin/python prototypes/inject_recovery_jwst.py \
          [--cache prototypes/cache_inject_jwst] [--offset-arcsec 3.0] \
          [--flux-jy 2.0e-6]
jwst stack + archive access required; unit tests never import this.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from autoreduce.pipeline import reduce_target
from autoreduce.target import TargetSpec

# The COSMOS-Web ring anchor (docs/design/jwst.md; program 1727).
NAME = "cosmos_web_ring"
RA, DEC = 150.10048, 1.89301
BAND = "F150W"


def build_input(out_dir: Path, flux_jy: float, r_eff_arcsec: float = 0.3,
                pixel_scale: float = 0.015, shape=(121, 121)) -> Path:
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
    parser.add_argument("--cache", default="prototypes/cache_inject_jwst")
    parser.add_argument("--offset-arcsec", type=float, default=3.0)
    parser.add_argument("--flux-jy", type=float, default=2.0e-6)
    args = parser.parse_args()

    args.cache = str(Path(args.cache).resolve())
    out_root = Path("prototypes/output/inject_recovery_jwst").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    input_path = build_input(out_root, args.flux_jy)
    inject_ra = RA + args.offset_arcsec / 3600.0
    common = dict(
        name=NAME, ra=RA, dec=DEC, instrument="nircam_sw",
        filter_name=BAND, proposal_ids=("1727",),
        cutout_shape=(419, 419), final_scale=0.03, final_pixfrac=1.0,
    )

    print("== phase 2a: clean reduction ==")
    reduce_target(TargetSpec(**common), Path(args.cache), out_root / "clean")
    print("== phase 2b: injected reduction ==")
    injected = reduce_target(
        TargetSpec(
            **common,
            inject_image=str(input_path),
            inject_pixel_scale=0.015,
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

    diff = data_inj - data_clean  # MJy/sr
    xy = WCS(hdr).world_to_pixel_values(inject_ra, DEC)
    yy, xx = np.mgrid[0 : diff.shape[0], 0 : diff.shape[1]]
    pixel_scale = injected["package"]["pixel_scale"]
    aperture = np.hypot(yy - xy[1], xx - xy[0]) * pixel_scale <= 2.0
    # Mosaic pixel solid angle (sr): (scale/206265)^2.
    omega = (pixel_scale / 206265.0) ** 2
    recovered_jy = float(diff[aperture].sum()) * omega * 1e6
    noise_jy = float(np.sqrt((noise_inj[aperture] ** 2).sum())) * omega * 1e6

    report = {
        "injected_flux_jy": args.flux_jy,
        "recovered_flux_jy_2arcsec": recovered_jy,
        "recovery_ratio": recovered_jy / args.flux_jy,
        "aperture_noise_jy": noise_jy,
        "total_injected_e": injected["inject"]["total_injected_e"],
        "n_frames": len(injected["inject"]["frames"]),
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    ok = abs(report["recovery_ratio"] - 1.0) < max(
        0.05, 3.0 * noise_jy / args.flux_jy
    )
    print("RECOVERY", "OK" if ok else "DISCREPANT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
