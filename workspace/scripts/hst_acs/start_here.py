"""
HST ACS: Start Here
===================

The Hubble Space Telescope's Advanced Camera for Surveys (ACS/WFC) is the reference instrument of
**PyAutoReduce**: the instrument every other pipeline in this workspace is described relative to,
and the one whose products set the quality bar. That bar is the SLACS survey — the ACS/WFC F814W
reductions of Bolton et al. 2008 (SLACS V, https://arxiv.org/abs/0805.1931), which sit behind the
largest and most-modeled sample of galaxy-scale strong lenses in existence. If a reduction
pipeline can reproduce a SLACS dataset from the raw archive, it can be trusted with your lens.

This script runs that reference reduction end to end: it points **PyAutoReduce** at the SLACS lens
SDSS J0008-0004, downloads the calibrated exposures from the MAST archive, drizzles them to the
SLACS convention, constructs the RMS noise-map and empirical PSF, and packages a modeling-ready
dataset that loads directly into **PyAutoLens**. Everything runs on defaults — one frozen spec,
one function call. The first run downloads ~0.5 GB of exposures and reference files and takes
roughly 10-30 minutes depending on your connection; re-runs reuse the cache and take a few
minutes.

If you want to see what each stage does internally, `step_by_step.py` in this folder walks the
same reduction stage by stage; `psf.py`, `individual.py`, `simulator.py` and `dials.py` go deeper
on the PSF, per-exposure products, injection testing and the drizzle dials respectively.

__Contents__

- **Why SLACS Is The Quality Bar:** The reference reduction this pipeline validates against.
- **Imports:** Import **PyAutoReduce** and the other libraries we need.
- **Paths:** Anchor the cache and output locations to the workspace root.
- **Target Spec:** Declare the reduction as a frozen `TargetSpec`.
- **What MAST Serves:** The `_flc` product, CTE correction and CRDS reference files.
- **The Reduction:** One call runs acquire, align, drizzle, noise, PSF and package.
- **The Drizzle Convention:** Why the mosaic is 0.05"/pixel, north-up, in electrons per second.
- **Provenance Walk:** Reading the diagnostics every reduction reports.
- **The Noise Recipe:** The RMS map and the Casertano correlated-noise factor R.
- **The PSF:** The empirical ePSF shipped with the dataset (full story in `psf.py`).
- **Output Products:** The four FITS files plus `reduction.json` on disk.
- **Plots:** Inspect the data, noise-map and PSF with matplotlib.
- **Load In PyAutoLens:** The dataset loads directly via `al.Imaging.from_fits`.
- **Wrap Up:** Where to go next in the workspace.

__Why SLACS Is The Quality Bar__

The Sloan Lens ACS Survey (SLACS; Bolton et al. 2006, SLACS I; Bolton et al. 2008, SLACS V,
https://arxiv.org/abs/0805.1931) imaged ~100 galaxy-scale strong lenses with ACS/WFC, mostly in
the F814W filter. Its reductions — dithered exposures drizzled to a 0.05"/pixel mosaic (SLACS IX,
Auger et al. 2009) — became the de-facto standard dataset for strong lens modeling: the mass
models, source reconstructions and substructure analyses of the following fifteen years were
calibrated on them.

**PyAutoReduce** therefore validates its ACS pipeline against SLACS directly: the target reduced
below, SDSS J0008-0004, is one of the pipeline's two acceptance targets, and its products have
been compared pixel-by-pixel against the legacy SLACS modeling dataset (a data-ratio parity of
~0.96, with the residual ~4-6% global flux offset documented and accepted — the legacy
reduction's exact provenance is unrecoverable, and lens-model inferences are invariant under a
global flux scale). The craft reference for ACS reduction more broadly is the COSMOS pipeline
(Koekemoer et al. 2007, https://arxiv.org/abs/astro-ph/0703095).
"""

"""
__Imports__

**PyAutoReduce** exposes exactly two names: `TargetSpec` (the frozen declaration of what to
reduce) and `reduce_target` (the function that reduces it). Everything else in this script is
standard scientific Python.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from autoreduce import TargetSpec, reduce_target

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the folder
containing `scripts/`). **PyAutoReduce** requires absolute paths: its drizzle step changes the
working directory internally, so relative paths would break.

- `cache/` holds downloaded exposures and CRDS reference files, reused across runs and scripts.
- `output/` receives the reduced datasets, one folder per target.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"      # Downloaded exposures + CRDS references (re-used across runs).
OUTPUT_ROOT = WORKSPACE / "output"    # Reduced datasets, one folder per target.

"""
__Target Spec__

A **PyAutoReduce** reduction is *declared*, not scripted: you build a frozen `TargetSpec` and the
pipeline is a pure function of that spec plus the archive — re-running it reproduces the dataset
deterministically (modulo upstream calibration reference-file updates, which the provenance
record tracks).

For this reference reduction every dial stays at its default, so the spec needs only four
entries: the target's name, its coordinates, and the HST proposal that observed it.
"""
SPEC = TargetSpec(
    name="slacs0008-0004",  # Names the output folder and the cache entry for this target.
    ra=2.012333,  # Target right ascension in degrees (SDSS J0008-0004).
    dec=-0.068944,  # Target declination in degrees.
    proposal_ids=("10886",),  # The SLACS ACS program; filters out exposures from unrelated visits at the same coordinates.
)

"""
The defaults this spec inherits are the SLACS convention, and every one of them is a dial you can
change (see `dials.py` for the trade study):

- `instrument="acs_wfc"`, `filter_name="F814W"` — the SLACS instrument/filter combination.
- `final_scale=0.05`, `final_pixfrac=0.8`, `final_kernel="square"` — the drizzle geometry.
- `cutout_shape=(281, 281)` — a 281x281 pixel cutout, ~14" across at 0.05"/pixel.
- `psf_shape=(21, 21)`, `psf_full_shape=(61, 61)` — the compact and extended PSF kernels.
- `cr_method="driz_cr"`, `psf_star_pass="auto"`, `psf_backend="epsf"` — cosmic-ray and PSF policy.

Because `TargetSpec` is a frozen dataclass, variants are made with `dataclasses.replace(SPEC,
...)` — you will see that idiom throughout this folder.

__What MAST Serves__

The acquire stage queries the Mikulski Archive for Space Telescopes (MAST) via `astroquery.mast`
(https://astroquery.readthedocs.io/en/latest/mast/mast.html) and downloads the `_flc` exposures:
fully calibrated individual exposures — bias, dark, flat and photometric calibration applied by
the `calacs` pipeline — **with the pixel-based CTE correction applied**.

CTE (charge transfer efficiency) degradation matters enormously for lensing: radiation damage to
the CCD traps charge during readout, systematically trailing and dimming faint sources far from
the readout amplifiers — exactly the regime of faint lensed arcs on low sky backgrounds. The
correction is the empirical charge-trap model of Anderson & Bedin 2010 (PASP 122, 1035,
https://ui.adsabs.harvard.edu/abs/2010PASP..122.1035A), which forward-models the readout and
iteratively inverts it. The `_flc` product has this applied; the older `_flt` product does not.
The ACS Data Handbook (https://hst-docs.stsci.edu/acsdhb) documents the full product zoo —
`step_by_step.py` walks it.

MAST's on-the-fly reprocessing keeps `_flc` files current with the best calibration reference
files, so **PyAutoReduce** never re-runs `calacs` itself. It does, however, sync the CRDS
reference files the drizzle stage needs (https://hst-crds.stsci.edu) into the cache on every run
— reference files are revised independently of the exposures, so even a cached re-run re-checks
(cheap when nothing is stale). `sync_references=False` is the explicit offline opt-out for
re-runs over a warm cache.

__The Reduction__

One function call runs the whole pipeline: acquire -> align -> drizzle -> noise -> PSF ->
package. The return value is the provenance record — the same dictionary written to
`reduction.json` next to the products.
"""
print(
    "\n"
    "Starting the reduction of slacs0008-0004.\n"
    "\n"
    "On a cold cache this downloads the ACS exposures (~0.5 GB) and CRDS reference files from\n"
    "MAST, then runs AstroDrizzle — expect ~10-30 minutes on the first run. Re-runs reuse the\n"
    "cache under cache/ and finish in a few minutes.\n"
)

record = reduce_target(
    SPEC,
    cache_root=CACHE_ROOT,  # Exposures + CRDS references live here; warm cache = offline-capable re-runs.
    output_root=OUTPUT_ROOT,  # Products land in output/slacs0008-0004/.
)

print("Reduction complete.")

"""
__The Drizzle Convention__

The exposures were combined with AstroDrizzle (DrizzlePac Handbook,
https://hst-docs.stsci.edu/drizzpac; drizzle algorithm: Fruchter & Hook 2002, PASP 114, 144,
https://ui.adsabs.harvard.edu/abs/2002PASP..114..144F) onto a single mosaic with three
conventions chosen for lens modeling:

- **0.05"/pixel** (`final_scale=0.05`) — the SLACS convention, matching the ACS/WFC native pixel
  and every existing SLACS modeling dataset.

- **North-up** (`final_rot=0`) — a uniform orientation across samples simplifies masks, position
  angles and cross-dataset comparison.

- **Electrons per second** (`final_units='cps'`) — **PyAutoLens** assumes data in e-/s, and the
  exposure time recorded in the provenance keeps the Poisson noise term computable.

The weighting is inverse-variance (`final_wht_type='IVM'`), because the noise-map construction
below needs exactly that weight map. The two genuinely contested dials — `final_pixfrac` (default
0.8, the SLACS value) and `final_kernel` (default `square`) — are first-class configuration, not
buried defaults; `dials.py` is the full trade study.

__Provenance Walk__

Every reduction reports diagnostics so the dial choices are auditable per dataset. They live in
the returned record (== `reduction.json`). Four are worth checking on every reduction:
"""
wht_diag = record["drizzle"]["weight_uniformity"]

print("\n--- provenance walk ---")
print(f"exposures combined       : {record['drizzle']['n_exposures']}")
print(f"weight uniformity        : {wht_diag['wht_rms_over_median']:.3f} "
      f"(limit {wht_diag['limit']}, acceptable={wht_diag['acceptable']})")
print(f"correlated noise factor R: {record['drizzle']['correlated_noise_factor']:.3f}")
print(f"PSF stars used           : {record['psf']['n_stars_used']}")
print(f"PSF star source pass     : {record['psf']['star_source_pass']}")

"""
Reading these:

- **`weight_uniformity`** — the STScI rule-of-thumb statistic RMS/median of the drizzle weight
  map. Values above ~0.2 mean the pixfrac is too small for the dither pattern (coverage
  speckle/holes); the pipeline's validation run measured 0.066 at pixfrac 0.8, comfortably
  uniform.

- **`correlated_noise_factor`** — the Casertano et al. 2000 (AJ 120, 2747) scalar R applied to
  the noise-map (next section). At the SLACS geometry (pixfrac 0.8, output scale = native scale)
  R = 1.364.

- **`n_stars_used` / `star_source_pass`** — how many field stars built the PSF, and which drizzle
  pass they were measured on. `psf.py` explains why that second entry exists.

__The Noise Recipe__

Drizzle does not emit an RMS map, but a lens-model likelihood is only as good as its noise-map —
chi^2 assumes it. **PyAutoReduce** constructs it per the strong-lensing literature (Bayer et al.,
https://arxiv.org/abs/1803.05952, section 3.1, derived on exactly this kind of SLACS-style ACS
data):

    sigma_i = R * sqrt( max(N_i, 0) / t_exp  +  1 / W_i )

- The **background term** `1/W_i` comes from the IVM weight map — read noise, dark current and
  sky in one term, per STScI weight-map semantics (DrizzlePac Handbook section 3.4).
- The **Poisson term** uses the source counts/s and the total exposure time, floored at zero so
  negative sky fluctuations never produce NaNs.
- **R** is the Casertano et al. 2000 correlated-noise correction: drizzling shares each input
  pixel among neighbouring output pixels, so a naive per-pixel RMS underestimates the noise a
  chi^2 over the mosaic actually sees. R (1.364 here) inflates the map to compensate, and is
  recorded in the provenance. `dials.py` shows how R varies with pixfrac.

The pipeline fails loudly — it refuses to package — if the noise-map contains NaN or zero-weight
pixels inside the cutout, rather than silently patching them.
"""
print("\nnoise recipe   :", record["noise"]["recipe"])
print("empirical sky RMS (e-/s):", f"{record['noise']['empirical_background_rms']:.5f}")

"""
__The PSF__

The dataset ships with an empirical PSF built from field stars in the mosaic — an "effective PSF"
(ePSF) in the lineage of Anderson & King 2000 (PASP 112, 1360,
https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A), constructed with `photutils`'s
`EPSFBuilder` (https://photutils.readthedocs.io/en/stable/user_guide/epsf.html).

The one invariant, whatever construction method is used: **the delivered PSF is the drizzled
PSF** — measured at the same kernel, pixfrac, scale and orientation as the science mosaic, so it
describes the blurring actually present in `data.fits`. Never pair a native-frame PSF with a
drizzled image.

PSF accuracy matters more in lens modeling than in most photometric applications: PSF mismatch
produces structured residuals at the lensed arcs that bias mass models and can mimic (or mask)
dark-matter substructure. The full PSF story — star selection, the `psf_star_pass` dial, the
STARRED backend, quality diagnostics and honest limitations — lives in the module docstring and
sections of `psf.py` in this folder. Read it before trusting any PSF at high precision.
"""
print("\nPSF method    :", record["psf"]["method"])
print("PSF FWHM (pix):", f"{record['psf']['fwhm_pix']:.2f}")

"""
__Output Products__

The products on disk are the exact input set **PyAutoLens** loads:

- `data.fits` — the 281x281 drizzled science cutout (~14" at 0.05"/pixel), WCS and units intact
  in the header (a deliberate deviation from the legacy SLACS cutouts, whose headers were
  stripped).
- `noise_map.fits` — the matching per-pixel RMS map, R applied.
- `psf.fits` — the 21x21 compact PSF kernel for fit convolution.
- `psf_full.fits` — the 61x61 extended PSF capturing the wings (for point-source work).
- `reduction.json` — the complete provenance record you walked above, plus a software-version
  envelope.

All kernels are odd-shaped, centred and unit-normalised — the **PyAutoLens** input standards.
"""
out_dir = OUTPUT_ROOT / SPEC.name

print("\n--- products ---")
for product in record["package"]["products"]:
    print(f"  {out_dir / product}")
print(f"  {out_dir / 'reduction.json'}")
print("pixel scale :", record["package"]["pixel_scale"])
print("data units  :", record["package"]["data_units"])

"""
__Plots__

Let's look at what we made. The data is shown with an arcsinh stretch (linear in the noise,
logarithmic on the galaxy), the standard way to see faint lensed arcs next to a bright deflector.
"""
data = fits.getdata(out_dir / "data.fits").astype(float)
noise_map = fits.getdata(out_dir / "noise_map.fits").astype(float)
psf = fits.getdata(out_dir / "psf.fits").astype(float)

plot_dir = out_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

sky_rms = record["noise"]["empirical_background_rms"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(np.arcsinh(data / (3.0 * sky_rms)), origin="lower", cmap="magma")
axes[0].set_title("data.fits (arcsinh stretch)")

# The noise-map is displayed capped just above the sky level: the Poisson term brightens it on
# the galaxy, and any masked pixels (noise = 1e8) would otherwise swamp the colour scale.
axes[1].imshow(
    np.clip(noise_map, 0.0, 5.0 * sky_rms), origin="lower", cmap="viridis"
)
axes[1].set_title("noise_map.fits (capped)")

axes[2].imshow(np.arcsinh(psf / psf.max() * 100.0), origin="lower", cmap="magma")
axes[2].set_title("psf.fits (arcsinh stretch)")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plot_path = plot_dir / "start_here_products.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)

print(f"\nPlot saved to: {plot_path}")

"""
You should see the elliptical deflector galaxy at the centre of the data panel with the faint
lensed arc around it, a noise-map that brightens over the galaxy (the Poisson term at work), and
a sharp, centred PSF.

__Load In PyAutoLens__

The whole point of the output contract: these files load directly into **PyAutoLens** with no
further processing — no unit conversion, no header surgery, no mask file. Bad pixels are already
handled by the masked-by-noise convention (noise = 1e8 where a pixel must not contribute), so any
chi^2 ignores them automatically.

The import is guarded — **PyAutoReduce** deliberately never depends on **PyAutoLens**, so this
workspace works even without it installed.
"""
try:
    import autolens as al
except ImportError:
    al = None
    print(
        "\nPyAutoLens is not installed, so the final loading demonstration is skipped.\n"
        "Install it with `pip install autolens` to model this dataset.\n"
        "The reduction itself is complete — the products above are on disk."
    )

if al is not None:
    reduction = json.loads((out_dir / "reduction.json").read_text())

    dataset = al.Imaging.from_fits(
        data_path=out_dir / "data.fits",
        noise_map_path=out_dir / "noise_map.fits",
        psf_path=out_dir / "psf.fits",
        pixel_scales=reduction["package"]["pixel_scale"],  # 0.05 — read from provenance, never hardcoded.
    )

    print("\nLoaded into PyAutoLens:")
    print(f"  shape       : {dataset.shape_native}")
    print(f"  pixel scale : {dataset.pixel_scales}")
    print(
        "\nYou are ready to model this lens — see autolens_workspace/scripts/imaging/start_here.py."
    )

"""
__Wrap Up__

You have reduced a SLACS lens from the MAST archive to a modeling-ready dataset with a single
declared spec: CTE-corrected exposures, the SLACS drizzle convention (0.05"/pixel, north-up,
e-/s), a chi^2-faithful noise-map with the correlated-noise correction applied, an empirical
drizzled PSF, and a provenance record that makes every choice auditable.

The following locations of the workspace are good places to checkout next:

- `scripts/hst_acs/step_by_step.py`: the same reduction, one stage at a time, with the archive
  anatomy, CRDS, drizzle parameters and noise construction taught in full.
- `scripts/hst_acs/psf.py`: the complete PSF story — selection cuts, star passes, STARRED,
  diagnostics and honest limitations.
- `scripts/hst_acs/dials.py`: the pixfrac / kernel / CR-method trade study, with the literature
  disagreements laid out.
- `scripts/hst_acs/individual.py`: per-exposure frame products — model the undrizzled frames and
  sidestep correlated noise entirely.
- `scripts/hst_acs/simulator.py`: inject a synthetic lensed arc into the real exposures and
  validate the pipeline end to end.
- `scripts/guides/output_contract.py`: the output contract in detail, and how the products map
  onto **PyAutoLens** standards.

__Env__ (Developer Only)

ENV: network
"""
