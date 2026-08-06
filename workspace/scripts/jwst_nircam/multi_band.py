"""
JWST NIRCam: Multi Band
=======================

NIRCam never observes in one filter. Its dichroic splits the beam so the short-wavelength
(SW) and long-wavelength (LW) channels expose *simultaneously*, and surveys like
COSMOS-Web deliver every field in four bands: F115W and F150W (SW) plus F277W and F444W
(LW). For strong lensing this is a gift — the lens galaxy and the lensed source usually
have very different colours, so multi-band imaging separates their light far better than
any single band can, and **PyAutoLens** can fit all four bands simultaneously.

This script reduces the COSMOS-Web ring in all four bands with **PyAutoReduce**,
assembles the results into a single multi-wavelength dataset folder, and then does the
cross-band audit the parity stance demands: are the internal closures (weight
uniformity, the sky/ERR consistency, the correlated-noise factor) consistent from band
to band?

Each band is a full calwebb_image3 run, so the first pass takes a while — budget an hour
or more on a cold cache. Bands already reduced by `start_here.py` / `step_by_step.py`
(F277W) re-use their caches.

__Contents__

- **Imports:** Import **PyAutoReduce** and the supporting libraries.
- **Paths:** Anchor the cache and output folders to the workspace root.
- **SW vs LW:** Two channels, two adapters, two pixel scales — and the undersampling trade-off between them.
- **Band Specs:** One `TargetSpec` per band, routed through `nircam_adapter_for_filter`.
- **The Four Reductions:** Loop the bands through `reduce_target`.
- **Multi-Wavelength Dataset Layout:** Assemble a `wavebands/<BAND>/` folder tree for multi-band modeling.
- **Cross-Band Consistency:** Compare weight uniformity, R and the sky/ERR closure across bands.
- **Plots:** The ring in four bands, side by side.
- **Modeling the Multi-Band Dataset:** Where the dataset goes next in **PyAutoLens**.
- **Wrap Up:** Where to go next.

__Imports__
"""
from pathlib import Path
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

from autoreduce import TargetSpec, reduce_target
from autoreduce.instruments import nircam_adapter_for_filter

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the
folder containing `scripts/`). **PyAutoReduce** requires absolute paths: its combine step
changes the working directory internally, so relative paths would break.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"  # downloaded exposures + CRDS references (re-used across runs)
OUTPUT_ROOT = WORKSPACE / "output"  # reduced datasets, one folder per target

"""
__SW vs LW__

The channel split is not just plumbing — it changes the character of the data:

- **SW (F115W, F150W):** native 0.031"/pixel, reduced onto a 0.03"/pixel grid. Twice
  the spatial resolution of LW, which for a ~2" Einstein ring means twice the number
  of resolution elements across the arcs — more constraining power on the source and
  the mass model. The catch: below ~2 microns the NIRCam PSF is *undersampled* — its
  FWHM spans less than two native pixels (see the JDox NIRCam imaging pages,
  https://jwst-docs.stsci.edu). Resampling dithered exposures onto the finer 0.03"
  grid partially recovers the undersampled information, but the mosaic PSF is harder
  to characterise (the full story is in `psf.py`).

- **LW (F277W, F444W):** native 0.063"/pixel, reduced onto 0.06"/pixel. Beyond ~2
  microns the telescope is diffraction-limited (Strehl ~0.8) and the PSF is
  well-sampled — cleaner PSF systematics at the price of coarser sampling. Red bands
  also favour the lens galaxy's old stellar population and any dusty, high-redshift
  source emission.

For lens modeling the practical upshot: the SW bands carry the sharpest constraints
but the touchiest PSFs; the LW bands are the robust workhorses. Fitting all four
simultaneously gets you both. Depth is not the differentiator — COSMOS-Web reaches
5-sigma point-source depths of roughly 26.7–28.3 AB across its four filters
(Casey et al. 2023; reduction details in Franco et al.,
https://arxiv.org/abs/2506.03256) — so the choice of which bands to lean on is driven
by resolution, PSF behaviour and the colours of the deflector and source, not by
signal-to-noise alone.

The two channels also mean two cutout shapes for the *same* sky area: at 0.03"/pixel a
~12.5" field is 419 pixels across, at 0.06"/pixel it is 209. These shapes match the
COSMOS-Web ring demo dataset convention, so every band covers the identical footprint.

__Band Specs__

One spec per band. `nircam_adapter_for_filter` does the channel routing — F115W/F150W
land on `nircam_sw`, F277W/F444W on `nircam_lw` — from lookup tables covering the full
NIRCam filter complement (wide, medium and narrow bands; the SW/LW boundary sits at
2.4 microns). A filter name the tables do not recognise raises a `KeyError` immediately,
before any download — misrouting a filter to the wrong channel would silently produce a
mosaic at the wrong pixel scale, so the routing is loud by design.
"""
RA, DEC = 150.10048, 1.89301  # the COSMOS-Web ring (Mercier et al. 2024, arXiv:2309.15986)

BANDS = ("F115W", "F150W", "F277W", "F444W")


def spec_for(band: str) -> TargetSpec:
    adapter = nircam_adapter_for_filter(band)
    shape = (419, 419) if adapter.key == "nircam_sw" else (209, 209)  # same ~12.5" footprint per channel
    return TargetSpec(
        name=f"cosmos_web_ring_{band.lower()}",  # one output folder per band
        ra=RA,  # target right ascension in degrees (J2000)
        dec=DEC,  # target declination in degrees (J2000)
        instrument=adapter.key,  # "nircam_sw" or "nircam_lw", routed from the filter
        filter_name=band,  # the NIRCam filter for this reduction
        proposal_ids=("1727",),  # COSMOS-Web only — other programs would change depth and noise
        final_scale=adapter.recommended_final_scale,  # SW 0.03" / LW 0.06" — the COSMOS-Web convention
        final_pixfrac=1.0,  # full drizzle drop, matching the COSMOS-Web mosaics
        cutout_shape=shape,  # SW (419, 419) / LW (209, 209)
    )


"""
__The Four Reductions__

Loop the bands. Each iteration is a complete pipeline run — acquire, calwebb_image3,
noise, PSF, package — and each band's exposures cache independently (SW and LW are
different files even for simultaneous observations).
"""
records = {}

for band in BANDS:
    spec = spec_for(band)
    print(
        f"""
    [{band}] starting reduction ({spec.instrument}, {spec.final_scale}\"/pixel,
    cutout {spec.cutout_shape}). A cold-cache band downloads its _cal exposures from
    MAST and runs calwebb_image3 — expect tens of minutes per band; warm caches
    (e.g. F277W after start_here.py) are much faster.
    """
    )
    records[band] = reduce_target(spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)
    print(f"[{band}] done -> {OUTPUT_ROOT / spec.name}")

"""
__Multi-Wavelength Dataset Layout__

Each band reduced into its own `output/cosmos_web_ring_<band>/` folder. For multi-band
modeling it is more convenient to gather them under one dataset root with a
`wavebands/<BAND>/` sub-folder per filter — the same layout the COSMOS-Web ring demo
dataset uses. We copy the four modeling products (plus the provenance record) per band;
nothing is modified, so each `reduction.json` stays valid for its band.
"""
dataset_dir = OUTPUT_ROOT / "cosmos_web_ring_multi_band"

PRODUCTS = ("data.fits", "noise_map.fits", "psf.fits", "psf_full.fits", "reduction.json")

for band in BANDS:
    band_src = OUTPUT_ROOT / f"cosmos_web_ring_{band.lower()}"
    band_dst = dataset_dir / "wavebands" / band
    band_dst.mkdir(parents=True, exist_ok=True)
    for product in PRODUCTS:
        shutil.copy2(band_src / product, band_dst / product)

print(f"Multi-wavelength dataset assembled at: {dataset_dir.resolve()}")
for path in sorted(dataset_dir.rglob("*.fits")):
    print("  ", path.relative_to(dataset_dir))

"""
__Cross-Band Consistency__

The parity stance (`start_here.py`) accepted that absolute agreement with the bespoke
COSMOS-Web team mosaics is not the bar — internal consistency is. That claim has a
cross-band leg: whatever global offsets exist between this reduction and the team's
(calibration vintage, background treatment) should behave *consistently* from band to
band, and each band's own closures should hold independently. So we tabulate, per band:

- `n_exposures` — how many exposures the footprint filter admitted.
- `weight_uniformity_cutout` — depth uniformity over the shipped cutout (policy < 0.2).
- `correlated_noise_factor` — the Casertano R; nearly identical for the two channels
  because both use pixfrac 1.0 at scale ratios just under 1.
- `sky_over_err_floor` — the blank-sky vs propagated-ERR closure; each band should sit
  near 1 independently, and a band that strays flags a band-specific problem (residual
  1/f banding is stronger in some filters, for example).
"""
consistency = {}

for band, record in records.items():
    consistency[band] = {
        "n_exposures": record["acquire"]["n_exposures"],
        "pixel_scale": record["package"]["pixel_scale"],
        "weight_uniformity_cutout": record["drizzle"]["weight_uniformity_cutout"],
        "correlated_noise_factor": record["noise"]["correlated_noise_factor"],
        "sky_over_err_floor": record["noise"]["sky_over_err_floor"],
        "psf_n_stars": record["psf"].get("n_stars_used"),
    }

print(json.dumps(consistency, indent=2))

summary_path = dataset_dir / "cross_band_consistency.json"
summary_path.write_text(json.dumps(consistency, indent=2))
print(f"Cross-band consistency table saved to: {summary_path.resolve()}")

"""
Read this table the way a referee would: uniform weights in every band (no band with
ragged coverage over the ring), R values that match the dial settings, and sky/ERR
closures near 1 across the board. If one band misbehaves, that band's `reduction.json`
has the per-stage evidence to chase it down.

__Plots__

The ring in four bands. The colour gradient you see — the arcs brightening relative to
the deflector towards the blue bands — is exactly the lens/source colour separation
that makes multi-band fitting so much more constraining than single-band.
"""
plot_dir = dataset_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for ax, band in zip(axes, BANDS):
    data = fits.getdata(dataset_dir / "wavebands" / band / "data.fits").astype(float)
    scale = np.nanpercentile(data, 99.5)
    ax.imshow(np.arcsinh(data / (0.05 * scale)), origin="lower", cmap="magma")
    ax.set_title(f"{band} ({records[band]['package']['pixel_scale']}\"/pix)")
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plot_path = plot_dir / "multi_band_ring.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)

print(f"Four-band plot saved to: {plot_path.resolve()}")

"""
__Modeling the Multi-Band Dataset__

The `wavebands/` tree above is a ready-made multi-wavelength dataset. In **PyAutoLens**,
each band loads as its own `al.Imaging` (with its own pixel scale and PSF), and the
`autolens_workspace/scripts/multi_dataset/` examples show how to fit them
simultaneously — sharing the mass model across bands while letting the lens and source
light vary with wavelength. That is the analysis COWLS ran on more than 100 lens
candidates in these same four bands (Nightingale et al. 2025,
https://arxiv.org/abs/2503.08777).

Remember the units caveat from `start_here.py` applies per band: every band is in
MJy/sr, so fitted intensities are surface brightnesses, and cross-band flux ratios
(colours) come from integrating model images and converting through each band's pixel
solid angle.
"""
try:
    import autolens as al
except ImportError:
    al = None
    print(
        "PyAutoLens is not installed (pip install autolens), so the loading demo is "
        "skipped — the multi-band dataset above is complete."
    )

if al is not None:
    datasets = {}
    for band in BANDS:
        band_dir = dataset_dir / "wavebands" / band
        band_record = json.loads((band_dir / "reduction.json").read_text())
        datasets[band] = al.Imaging.from_fits(
            data_path=band_dir / "data.fits",
            noise_map_path=band_dir / "noise_map.fits",
            psf_path=band_dir / "psf.fits",
            pixel_scales=band_record["package"]["pixel_scale"],
        )
        print(
            f"[{band}] loaded al.Imaging: shape {datasets[band].data.shape_native}, "
            f"pixel scale {datasets[band].pixel_scales}"
        )

"""
__Wrap Up__

Four bands, four independent pipeline runs, one multi-wavelength dataset — with a
cross-band consistency table certifying that every band's internal closures hold and
that the channel conventions (SW 0.03"/419px, LW 0.06"/209px) cover the same footprint.

The following locations of the workspace are good places to checkout next:

- `autolens_workspace/scripts/multi_dataset/`: simultaneous multi-wavelength lens modeling — the consumer of this dataset.
- `scripts/jwst_nircam/psf.py`: why the SW bands' PSFs need more care than LW — undersampling and the ePSF story.
- `scripts/jwst_nircam/individual.py`: frame products, where the SW undersampling argument becomes decisive.
- `scripts/guides/output_contract.py`: the per-band product contract in detail.

__Env__ (Developer Only)

ENV: network
"""
