"""
Start Here: HST WFC3/UVIS
=========================

This script reduces Hubble Space Telescope WFC3/UVIS imaging of the strong gravitational lens
SDSS J0252+0039 into a modeling-ready dataset — `data.fits`, `noise_map.fits`, `psf.fits` and
`psf_full.fits`, plus a full `reduction.json` provenance record — using **PyAutoReduce**.

WFC3/UVIS shares almost all of its reduction machinery with ACS/WFC: both are CCD channels whose
calibrated exposures are CTE-corrected `_flc` files, and both run through the same AstroDrizzle
combination, noise and PSF stages. This folder therefore teaches only what *changes* for UVIS.
For the full stage-by-stage depth — what drizzle actually does, how the noise map is derived,
how the PSF tiers work — see `hst_acs/start_here.py` and `hst_acs/step_by_step.py`, which this
script cross-references rather than duplicates.

The first run downloads the exposures from MAST and syncs CRDS reference files, so expect it to
take tens of minutes; re-runs hit the local cache and are much faster.

__Contents__

- **The UVIS Channel:** What WFC3/UVIS is and why lens modelers use it.
- **The Anchor: SDSS J0252+0039:** The published Bayer et al. F390W reduction this script reproduces.
- **Imports:** Import the required Python libraries.
- **Paths:** Anchor every path to the workspace root, absolutely.
- **CTE and the flc Product:** UVIS exposures are CTE-corrected before we ever touch them.
- **Drizzle Dials:** Why this reduction drizzles to the native scale with pixfrac 1.0.
- **Target Spec:** Declare the reduction as a frozen `TargetSpec`.
- **The Reduction:** Run the full pipeline with one function call.
- **Provenance:** Walk the returned provenance record's key diagnostics.
- **Noise Validation:** Check the noise map against the published sigma_sky of ~0.002 e-/s.
- **Plots:** Visualize the data, noise map and PSF.
- **Load the Dataset in PyAutoLens:** Load the products as an `al.Imaging` dataset.
- **Wrap Up:** Summary of the script and next steps.

__The UVIS Channel__

WFC3/UVIS is the blue/optical CCD channel of Wide Field Camera 3, with a native plate scale of
0.0396"/pixel — finer than the 0.05"/pixel of ACS/WFC. For strong lensing this matters twice
over: the finer pixels sample the PSF better, and blue filters like F390W isolate the lensed
emission of a star-forming background source while the red, quiescent lens galaxy fades — a
high-contrast view of the arcs that redder bands cannot give.

The authoritative reference for everything the instrument pipeline does to UVIS data is the
WFC3 Data Handbook (https://hst-docs.stsci.edu/wfc3dhb), which documents the `calwf3`
calibration pipeline, the UVIS charge-transfer-efficiency (CTE) correction (its Chapter 6 is
the UVIS CTE reference) and the data products this script consumes. Image combination follows
the DrizzlePac Handbook (https://hst-docs.stsci.edu/drizzpac).

__The Anchor: SDSS J0252+0039__

Every **PyAutoReduce** instrument channel is validated against a published reduction, and for
UVIS that anchor is the Bayer et al. (https://arxiv.org/abs/1803.05952) F390W reduction of the
strong lens SDSS J0252+0039: output at the native 0.0396"/pixel, drizzle pixfrac 1.0, and a
noise map built as sigma = sqrt(N/W + sigma_sky^2) with a measured sky RMS of
sigma_sky ~ 0.002 e-/s (their Section 3.1). This script reduces the same data with the same
dials and closes the loop on their published numbers.

Why was so much care lavished on this particular noise map? Because this dataset underpins
*surface-brightness-anomaly* science: Bayer et al. (arXiv:1803.05952, and the follow-up
arXiv:2302.00480) constrain the dark-matter substructure content of the lens from the power
spectrum of tiny surface-brightness fluctuations in the arcs. Any error in the noise map —
an underestimated sky RMS, unaccounted correlated noise from drizzling — masquerades as
exactly the signal being measured. A reduction pipeline that gets the noise right, and writes
down how it did so, is the difference between a substructure measurement and a systematics
measurement. That standard is what **PyAutoReduce** aims to inherit for every dataset.
"""

"""
__Imports__

**PyAutoReduce** exposes exactly two names: `TargetSpec` (the frozen declaration of the
reduction) and `reduce_target` (the one function that executes it). Everything else in this
script is plotting and inspection of the products it writes.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoreduce import TargetSpec, reduce_target

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the folder
containing `scripts/`). **PyAutoReduce** requires absolute paths: its drizzle step changes the
working directory internally, so relative paths would break.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"      # downloaded exposures + CRDS references (re-used across runs)
OUTPUT_ROOT = WORKSPACE / "output"    # reduced datasets, one folder per target

"""
__CTE and the flc Product__

Like ACS, UVIS is a CCD read out by clocking charge across the detector, and years of radiation
damage have filled the silicon with charge traps that smear faint sources into trails during
readout — charge transfer efficiency (CTE) losses. The standard correction is the pixel-based
model of Anderson & Bedin (2010, PASP 122, 1035;
https://ui.adsabs.harvard.edu/abs/2010PASP..122.1035A), which forward-models the trapping
during readout and iteratively inverts it. For WFC3/UVIS the current implementation is the
v2.0 correction of Anderson et al. (2021, WFC3 ISR 2021-09), run inside `calwf3` as the
standalone-capable `wf3cte` step; the corrected product is the `_flc` file, which is what
**PyAutoReduce** downloads from MAST (never the uncorrected `_flt`).

This is not a cosmetic fix. By ~2021, a faint source far from the readout amplifier on a low
(~20 e-) background loses almost *half* its flux uncorrected — and lensed arcs are exactly
that: faint, extended, often on low backgrounds. Uncorrected CTE would systematically dim and
trail the arcs whose photometry the lens model fits.

UVIS observations also routinely apply a *post-flash*: an LED illumination that deliberately
raises the background to ~12-20 e-, filling the charge traps so faint signal survives readout.
The CTE correction and the post-flash work together — but the post-flash raises the sky level,
and with it the background noise, which is why the noise map below is *measured* from the data
rather than assumed. See `hst_wfc3_uvis/step_by_step.py` for the full calibration walk.

__Drizzle Dials__

The combination step drizzles (Fruchter & Hook 2002, PASP 114, 144;
https://ui.adsabs.harvard.edu/abs/2002PASP..114..144F) the `_flc` exposures onto a common
output grid. Two dials define the output: `final_scale` (the output pixel size) and
`final_pixfrac` (how much each input pixel is shrunk before being dripped onto the grid).

Bayer et al. chose the *native* scale (0.0396"/pixel) with pixfrac 1.0 — plain shift-and-add,
no sub-pixel resampling games. The reasoning is about noise: drizzling to a finer grid, or
shrinking the drops, redistributes each input pixel's noise across multiple output pixels and
*correlates* them. A chi-squared likelihood assumes independent pixels, so correlated noise
must either be minimized or accounted for. At native scale with pixfrac 1.0 the correlation is
as simple as it gets, described by the Casertano et al. (2000, AJ 120, 2747) noise-correlation
factor R = 1.5.

One honest caveat: Bayer et al. do not use the scalar R at all — they propagate the
correlation exactly, by drizzling *blank-sky noise realizations* through the same pipeline and
measuring the result. **PyAutoReduce** applies the scalar R to the per-pixel RMS instead: a
single number that inflates the noise map so its diagonal is correct on average, at the cost
of ignoring the off-diagonal structure. It is the standard DrizzlePac-handbook treatment
(Section 3.4), it is recorded in `reduction.json` on every run, and the comparison below
accounts for it explicitly — but if your science lives in the noise power spectrum, know the
difference between the two approaches.

__Target Spec__

A **PyAutoReduce** reduction is *declared*, not scripted: every dial lives in a frozen
`TargetSpec`, so the pipeline is a pure function of the spec plus the archive, and the same
spec always reproduces the same dataset. Below is the J0252+0039 F390W spec with the published
Bayer et al. dials. Dials left at their defaults (cutout shape, PSF shapes, CR method...) are
documented in `hst_acs/dials.py` and `guides/target_spec.py`.
"""
RA, DEC = 43.188375, 0.666222  # SDSS J0252+0039: 02h52m45.21s +00d39m58.4s

spec = TargetSpec(
    name="j0252+0039_f390w",  # Output folder name under output/.
    ra=RA,                    # Target right ascension in degrees (cutout centre).
    dec=DEC,                  # Target declination in degrees.
    instrument="wfc3_uvis",   # The UVIS adapter: _flc products, iref references, 0.0396"/pix native.
    filter_name="F390W",      # The blue filter of the published substructure analysis.
    final_scale=0.0396,       # Bayer dial: drizzle to the native UVIS scale (no upsampling).
    final_pixfrac=1.0,        # Bayer dial: full drop (shift-and-add) -> simplest noise correlation, R = 1.5.
)

"""
__The Reduction__

One call runs the whole pipeline: MAST query and download of the F390W `_flc` exposures, CRDS
reference sync, alignment on the archive's Gaia-tied WCS, AstroDrizzle combination with
cosmic-ray flagging, noise-map construction, PSF measurement from field stars, and packaging
of the WCS-preserving cutout. The returned dictionary is the full provenance record — the same
content written to `reduction.json` next to the FITS products.
"""
print(
    "Running the J0252+0039 F390W reduction. The first run downloads the "
    "exposures from MAST and syncs CRDS reference files (~GBs; expect tens "
    "of minutes). Re-runs re-use the cache under "
    f"{CACHE_ROOT} and are much faster."
)

record = reduce_target(spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)

out_dir = OUTPUT_ROOT / spec.name

print(f"Products written to: {out_dir}")

"""
__Provenance__

Every run reports its diagnostics, so the dials are auditable per dataset. The ones worth
reading on every reduction:

- `acquire.n_exposures`: how many `_flc` exposures went into the stack.
- `drizzle.weight_uniformity`: the STScI rule-of-thumb statistic RMS/median of the drizzle
  weight map. Values above ~0.2 mean the pixfrac is too small for the dither pattern
  (coverage speckle or holes) — with pixfrac 1.0 this should pass comfortably.
- `noise.correlated_noise_factor`: the Casertano R applied to the noise map — 1.5 for these
  dials (native scale, pixfrac 1.0).
- `psf`: the PSF method, the number of stars used, and which drizzle pass the stars came from.
"""
print(f"n_exposures         : {record['acquire']['n_exposures']}")
print(f"weight_uniformity   : {record['drizzle']['weight_uniformity']}")
print(f"correlated_noise_R  : {record['noise']['correlated_noise_factor']}")
print(f"psf                 : {json.dumps(record['psf'], indent=2)}")

"""
__Noise Validation__

Now the closure check against the published anchor. Bayer et al. measure a blank-sky RMS of
sigma_sky ~ 0.002 e-/s in this reduction. **PyAutoReduce** measures its own sigma-clipped sky
RMS from the mosaic and records it as `noise.empirical_background_rms` — the two should agree,
since they describe the same pixels through the same dials.

The second check reads the noise *map*: in blank sky, the per-pixel noise recipe
sigma = R * sqrt(N/W + 1/W) reduces to approximately R * sigma_sky, so the noise map's faint
floor (its 5th percentile) divided by R * sigma_sky should be close to 1. If that ratio drifts
from unity, the noise map and the measured sky disagree — the exact failure mode that would
corrupt a surface-brightness-anomaly measurement, caught here in one line. (Remember the R
accounting from `__Drizzle Dials__`: our sky RMS is measured *before* R is applied, while the
noise map carries R — the ratio below divides it back out.)
"""
from astropy.io import fits

BAYER_SIGMA_SKY = 0.002  # e-/s, published for the F390W reduction (arXiv:1803.05952).

noise = fits.getdata(out_dir / "noise_map.fits").astype(float)

r_factor = record["noise"]["correlated_noise_factor"]
sky_rms = record["noise"]["empirical_background_rms"]

print(f"empirical sky RMS          : {sky_rms:.5f} e-/s")
print(f"published Bayer sigma_sky  : {BAYER_SIGMA_SKY:.5f} e-/s")
print(f"sky RMS / published        : {sky_rms / BAYER_SIGMA_SKY:.3f}")
print(
    f"noise floor / (R * sky RMS): "
    f"{float(np.nanpercentile(noise, 5)) / (r_factor * sky_rms):.3f}"
)

"""
__Plots__

The three products a lens model consumes, visualized. The data uses arcsinh scaling — linear
near zero so the sky noise is visible, logarithmic on the bright lens galaxy — which is the
standard way to see faint arcs and the sky in one image.
"""
data = fits.getdata(out_dir / "data.fits").astype(float)
psf = fits.getdata(out_dir / "psf.fits").astype(float)

plot_dir = out_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

im0 = axes[0].imshow(np.arcsinh(data / sky_rms), origin="lower", cmap="magma")
axes[0].set_title("data (arcsinh, sky-RMS units)")
fig.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(noise, origin="lower", cmap="viridis")
axes[1].set_title("noise map (e-/s)")
fig.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(np.arcsinh(psf / psf.max() * 1e3), origin="lower", cmap="magma")
axes[2].set_title("psf (arcsinh)")
fig.colorbar(im2, ax=axes[2], fraction=0.046)

fig.suptitle(f"{spec.name}: WFC3/UVIS F390W at 0.0396\"/pix, pixfrac 1.0")
fig.tight_layout()

plot_path = plot_dir / "start_here_products.png"
fig.savefig(plot_path, dpi=120)
plt.close(fig)

print(f"Plot saved to: {plot_path}")

"""
__Load the Dataset in PyAutoLens__

The product set is exactly the input format of **PyAutoLens**: `data.fits` (e-/s, WCS intact),
`noise_map.fits` (matching RMS, correlated-noise corrected), `psf.fits` (21x21, odd,
unit-normalized) and the pixel scale recorded in `reduction.json`. Any pixel the reduction
could not trust carries a noise value of 1e8 with the data zeroed — "masked by noise" — so no
mask surgery is needed on the modeling side.

**PyAutoReduce** deliberately never imports **PyAutoLens** (it stays releasable on its own),
so the load below is guarded: if **PyAutoLens** is not installed the reduction is still
complete and the message tells you how to proceed.
"""
try:
    import autolens as al
except ImportError:
    al = None
    print(
        "PyAutoLens is not installed, so the final loading step is skipped. "
        "Run `pip install autolens` to model this dataset — the reduction "
        "itself is complete and the FITS products are on disk."
    )

if al is not None:
    dataset = al.Imaging.from_fits(
        data_path=out_dir / "data.fits",
        noise_map_path=out_dir / "noise_map.fits",
        psf_path=out_dir / "psf.fits",
        pixel_scales=record["package"]["pixel_scale"],
    )
    print(
        f"Loaded al.Imaging dataset: shape {dataset.data.shape_native} at "
        f"{record['package']['pixel_scale']}\"/pix — ready for lens modeling "
        "(see autolens_workspace/scripts/imaging/start_here.py)."
    )

"""
__Wrap Up__

You have reduced HST WFC3/UVIS imaging of a strong lens end-to-end with the published dials of
its literature anchor, validated the noise map against the published sky RMS, and loaded the
result as a **PyAutoLens** dataset. Along the way you saw what makes UVIS UVIS: the `_flc` CTE
correction, the post-flash noise budget, and the native-scale / pixfrac 1.0 drizzle choice
that keeps the noise correlation simple and honest.

The following locations of the workspace are good places to checkout next:

- `scripts/hst_wfc3_uvis/step_by_step.py`: every UVIS reduction stage in detail, as a delta
  against the ACS reference, including live MAST filter discovery.
- `scripts/hst_wfc3_uvis/psf.py`: the UVIS PSF story on a star-rich field, including the
  STARRED backend.
- `scripts/hst_wfc3_uvis/simulator.py`: inject a synthetic lensed arc into these same frames
  to validate the pipeline end-to-end.
- `scripts/hst_wfc3_ir/start_here.py`: the *other* WFC3 channel — an infrared detector that
  works completely differently.
- `scripts/hst_acs/start_here.py`: the ACS/WFC reference reduction all HST channels build on.
- `scripts/guides/noise_maps.py`: the noise recipe, the Casertano R and why chi-squared cares.

__Env__ (Developer Only)

ENV: network
"""
