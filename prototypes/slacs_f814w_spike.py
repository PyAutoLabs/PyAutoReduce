"""
SLACS F814W spike — de-risk the HST/ACS pipeline design end-to-end on one lens.

Stages (each idempotent, cached under prototypes/cache/, outputs under
prototypes/output/):

  1. acquire  — query MAST for ACS/WFC F814W exposures at the target, download
                ``_flc`` files only.
  2. drizzle  — AstroDrizzle to a 0.05"/pix north-up mosaic with an IVM weight
                map (design doc stage-3 parameters).
  3. noise    — RMS map: background term 1/sqrt(WHT) + Poisson source term
                (design doc stage 4; correlated-noise factor reported, not yet
                applied — a parity question).
  4. psf      — count usable field stars; build a photutils ePSF if enough
                exist (tier 1), else report that tier 2 (TinyTim+focus) is
                required for this field.
  5. compare  — 281x281 cutout at the target vs the legacy modeling dataset
                (data ratio, noise ratio) to answer the units and
                correlated-noise questions empirically.

Run inside the repo's spike venv:

    .venv/bin/python prototypes/slacs_f814w_spike.py [--stage all]

This is a prototype: it prints findings for the design doc's parity appendix;
it is not pipeline code.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

PROTO_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROTO_DIR / "cache"
OUTPUT_DIR = PROTO_DIR / "output"

# SDSS J0008-0004 (slacs0008-0004): 00h08m02.96s -00d04m08.2s
TARGET = {
    "name": "slacs0008-0004",
    "ra": 2.012333,
    "dec": -0.068944,
}
LEGACY_DIR = Path("/mnt/c/Users/Jammy/Science/subhalo/dataset/slacs") / TARGET["name"]

FINAL_SCALE = 0.05  # arcsec / pix, SLACS convention
CUTOUT_SHAPE = (281, 281)


def stage_acquire():
    """Query MAST and download the ACS/WFC F814W ``_flc`` exposures."""
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Observations

    coord = SkyCoord(TARGET["ra"], TARGET["dec"], unit="deg")
    obs = Observations.query_criteria(
        coordinates=coord,
        radius="0.5 arcmin",
        obs_collection="HST",
        instrument_name="ACS/WFC",
        filters="F814W",
        dataproduct_type="image",
    )
    print(f"[acquire] {len(obs)} ACS/WFC F814W observations at {TARGET['name']}")
    for row in obs:
        print(
            f"[acquire]   {row['obs_id']}  prog={row['proposal_id']}  "
            f"t_exp={row['t_exptime']:.0f}s"
        )
    if len(obs) == 0:
        sys.exit("[acquire] no observations found — check coordinates/filter")

    products = Observations.get_product_list(obs)
    flc = Observations.filter_products(
        products, productSubGroupDescription=["FLC"], mrp_only=False
    )
    print(f"[acquire] downloading {len(flc)} FLC products -> {CACHE_DIR}")
    manifest = Observations.download_products(flc, download_dir=str(CACHE_DIR))
    paths = sorted(
        str(p) for p in Path(CACHE_DIR).rglob("*_flc.fits")
    )
    (CACHE_DIR / "cache_manifest.json").write_text(
        json.dumps({"target": TARGET, "flc_files": paths}, indent=2)
    )
    print(f"[acquire] cached {len(paths)} FLC exposures")
    return paths


def _flc_paths():
    manifest = CACHE_DIR / "cache_manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())["flc_files"]
    return sorted(str(p) for p in Path(CACHE_DIR).rglob("*_flc.fits"))


def stage_drizzle():
    """AstroDrizzle the cached exposures to the design's mosaic grid."""
    from drizzlepac import astrodrizzle

    flc = _flc_paths()
    if not flc:
        sys.exit("[drizzle] no FLC files cached — run --stage acquire first")
    print(f"[drizzle] combining {len(flc)} exposures")

    OUTPUT_DIR.mkdir(exist_ok=True)
    os.chdir(OUTPUT_DIR)  # AstroDrizzle writes into cwd
    astrodrizzle.AstroDrizzle(
        input=flc,
        output=f"{TARGET['name']}_f814w",
        preserve=False,
        build=False,
        clean=True,
        updatewcs=False,
        skymethod="globalmin+match",
        driz_cr=len(flc) > 1,  # CR rejection needs >= 2 exposures
        median=len(flc) > 1,
        blot=len(flc) > 1,
        final_scale=FINAL_SCALE,
        final_rot=0.0,
        final_wht_type="IVM",
        final_pixfrac=0.8,
        final_kernel="square",
        final_units="cps",
    )
    sci = glob.glob(str(OUTPUT_DIR / "*drz_sci.fits")) + glob.glob(
        str(OUTPUT_DIR / "*drc_sci.fits")
    )
    print(f"[drizzle] mosaic written: {sci}")


def _mosaic_files():
    def one(pattern):
        hits = sorted(OUTPUT_DIR.glob(pattern))
        return hits[-1] if hits else None

    sci = one("*_sci.fits")
    wht = one("*_wht.fits")
    if sci is None or wht is None:
        sys.exit("[noise] mosaic not found — run --stage drizzle first")
    return sci, wht


def stage_noise():
    """Design-doc stage 4: sigma = sqrt(Poisson + background) from SCI+WHT."""
    from astropy.io import fits

    sci_path, wht_path = _mosaic_files()
    sci_hdu = fits.open(sci_path)[0]
    sci, hdr = sci_hdu.data, sci_hdu.header
    wht = fits.open(wht_path)[0].data

    exptime = hdr.get("EXPTIME", hdr.get("TEXPTIME"))
    print(f"[noise] EXPTIME={exptime}s  units={hdr.get('BUNIT', 'cps (assumed)')}")

    bad = wht <= 0
    wht_safe = np.where(bad, np.nan, wht)
    var_bkg = 1.0 / wht_safe  # IVM weight: inverse variance of background
    var_src = np.clip(sci, 0, None) / exptime  # cps Poisson: var = N_cps / t
    noise = np.sqrt(var_src + var_bkg)

    # Correlated-noise factor R (DrizzlePac handbook / Casertano+2000),
    # reported for the parity discussion, NOT yet applied.
    p, s = 0.8, FINAL_SCALE / 0.05  # pixfrac, scale ratio vs native 0.05 grid
    # native ACS pixel is 0.05" so s=1 here; formula kept for generality
    r = (p / s) * (1 - s / (3 * p)) if s < p else 1 - p / (3 * s)
    R = 1.0 / r
    print(f"[noise] correlated-noise factor R = {R:.3f} (pixfrac={p}, s={s})")
    print(f"[noise] masked pixels (wht<=0): {bad.sum()}")

    fits.PrimaryHDU(noise.astype(np.float32), header=hdr).writeto(
        OUTPUT_DIR / "noise_map_mosaic.fits", overwrite=True
    )
    print(f"[noise] wrote {OUTPUT_DIR / 'noise_map_mosaic.fits'}")


def stage_psf():
    """Tier-1 feasibility: how many usable ePSF stars does this field have?"""
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    sci_path, _ = _mosaic_files()
    sci = fits.open(sci_path)[0].data
    mean, median, std = sigma_clipped_stats(sci, sigma=3.0)
    finder = DAOStarFinder(fwhm=2.0, threshold=10.0 * std, sharplo=0.4, sharphi=1.0)
    sources = finder(sci - median)
    n = 0 if sources is None else len(sources)
    print(f"[psf] point-like detections at >10 sigma: {n}")
    if n >= 10:
        print("[psf] tier 1 (photutils ePSF) looks feasible for this field")
    else:
        print("[psf] star-poor field — tier 2 (TinyTim + focus) required, as the")
        print("[psf] design anticipates for SLACS snapshot pointings")


def stage_compare():
    """Cut out the lens and compare against the legacy modeling dataset."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D

    sci_path, wht_path = _mosaic_files()
    sci_hdu = fits.open(sci_path)[0]
    noise = fits.open(OUTPUT_DIR / "noise_map_mosaic.fits")[0].data
    wcs = WCS(sci_hdu.header)
    coord = SkyCoord(TARGET["ra"], TARGET["dec"], unit="deg")

    cut_data = Cutout2D(sci_hdu.data, coord, CUTOUT_SHAPE, wcs=wcs)
    cut_noise = Cutout2D(noise, coord, CUTOUT_SHAPE, wcs=wcs)

    fits.PrimaryHDU(
        cut_data.data.astype(np.float32), header=cut_data.wcs.to_header()
    ).writeto(OUTPUT_DIR / "data.fits", overwrite=True)
    fits.PrimaryHDU(
        cut_noise.data.astype(np.float32), header=cut_noise.wcs.to_header()
    ).writeto(OUTPUT_DIR / "noise_map.fits", overwrite=True)
    print(f"[compare] wrote cutouts to {OUTPUT_DIR}")

    legacy_data = fits.open(LEGACY_DIR / "data.fits")[0].data
    legacy_noise = fits.open(LEGACY_DIR / "noise_map.fits")[0].data

    # Legacy cutouts have stripped headers: align by integer-pixel
    # cross-correlation of the data images before comparing.
    from scipy.signal import fftconvolve

    a = cut_data.data - np.nanmedian(cut_data.data)
    b = legacy_data - np.nanmedian(legacy_data)
    corr = fftconvolve(np.nan_to_num(a), np.nan_to_num(b)[::-1, ::-1], mode="same")
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    dy -= a.shape[0] // 2
    dx -= a.shape[1] // 2
    print(f"[compare] cross-correlation offset (new vs legacy): dy={dy}, dx={dx}")

    new_data = np.roll(cut_data.data, (-dy, -dx), axis=(0, 1))
    new_noise = np.roll(cut_noise.data, (-dy, -dx), axis=(0, 1))

    bright = legacy_data > 10 * np.nanmedian(legacy_noise)
    ratio_data = new_data[bright] / legacy_data[bright]
    ratio_noise = new_noise / legacy_noise

    print("[compare] ---- parity findings ----")
    print(
        f"[compare] data ratio (bright px, n={bright.sum()}): "
        f"median={np.nanmedian(ratio_data):.4f}  "
        f"(1.0 => same units/photometry; ~EXPTIME => legacy in e-)"
    )
    print(
        f"[compare] noise ratio: median={np.nanmedian(ratio_noise):.4f}  "
        f"16-84%=[{np.nanpercentile(ratio_noise, 16):.4f}, "
        f"{np.nanpercentile(ratio_noise, 84):.4f}]"
    )
    summary = {
        "offset": [int(dy), int(dx)],
        "data_ratio_median": float(np.nanmedian(ratio_data)),
        "noise_ratio_median": float(np.nanmedian(ratio_noise)),
    }
    (OUTPUT_DIR / "parity_summary.json").write_text(json.dumps(summary, indent=2))


STAGES = {
    "acquire": stage_acquire,
    "drizzle": stage_drizzle,
    "noise": stage_noise,
    "psf": stage_psf,
    "compare": stage_compare,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", default="all", choices=["all", *STAGES], help="stage to run"
    )
    args = parser.parse_args()
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, fn in STAGES.items():
        if args.stage in ("all", name):
            print(f"\n===== {name} =====")
            fn()


if __name__ == "__main__":
    main()
