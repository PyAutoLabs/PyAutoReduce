"""
Start Here: Data Reduction for Lens Modeling
============================================

Every strong-lens model fit begins long before the first likelihood evaluation: it begins with a
data reduction. **PyAutoReduce** turns raw archival telescope data — HST, JWST, Keck adaptive
optics, ALMA — into the modeling-ready datasets that **PyAutoLens** and **PyAutoGalaxy** load,
and this script is the single best first read of the workspace. In about thirty minutes of
reading (and, if you run it, one real HST reduction) you will understand why reduction quality
decides modeling quality, how a reduction is declared rather than scripted, and where to go for
your own instrument.

The script performs a complete, real reduction: it downloads the SLACS strong lens
SDSS J0008-0004's HST ACS/WFC exposures from the MAST archive, reduces them with the default
pipeline, walks the provenance record, plots the products, and loads the finished dataset into
**PyAutoLens**. Running it therefore needs network access and the HST reduction stack
(`pip install "autoreduce[hst]"`); reading it needs neither.

__Contents__

- **Why Reduction Quality Matters:** The two hard requirements lens modeling places on a reduction: an accurate PSF and an honest noise map.
- **A Reduction Is Declared, Not Scripted:** The `TargetSpec` philosophy — a reduction is a pure function of a declared spec plus the archive.
- **Imports:** Import **PyAutoReduce** and the plotting libraries.
- **Paths:** Anchor the cache and output directories to the workspace root with absolute paths.
- **Target Spec:** Declare the SLACS J0008-0004 reduction, dial by dial.
- **The Output Contract:** The four FITS products plus `reduction.json` that every reduction emits.
- **Run The Reduction:** One function call: acquire, align, drizzle, noise, PSF, package.
- **Provenance Walk:** Read the diagnostics out of the returned record — weight uniformity, the correlated-noise factor, star counts.
- **Plotting The Products:** Inspect the data, noise map and PSF with matplotlib.
- **The SLACS Quality Bar:** Validate the reduction against the legacy SLACS dataset — including the documented ~6% flux offset and the ~30%-higher-by-design noise.
- **Load Into PyAutoLens:** The seam between reduction and modeling: `al.Imaging.from_fits`.
- **Where To Go Next:** One-paragraph routing to every instrument folder in the workspace.
- **Wrap Up:** Summary and good places to checkout next.

__Why Reduction Quality Matters__

Lens modeling asks more of a reduction than almost any other analysis, and the demands
concentrate in two products most reductions treat as afterthoughts: the PSF and the noise map.

**The PSF.** A lens model fits the lensed arcs at the pixel level, convolved with the PSF. If the
PSF is wrong, the model leaves structured residuals at the arcs — residuals that can mimic, or
mask, the signal of dark-matter substructure that lens modeling is often trying to detect (see
the discussion of PSF systematics in the HST lensing literature, e.g. Bayer et al.,
https://arxiv.org/abs/1803.05952). Empirical effective PSFs built from stars in the science
frames themselves (Anderson & King 2000, PASP 112, 1360,
https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A) are the modern standard, and a subtle
invariant matters throughout this workspace: the delivered PSF must be processed *identically* to
the science data — drizzled with the same kernel, pixfrac, scale and rotation — or it does not
describe the blurring in the mosaic at all.

**The noise map.** Every chi-squared and every Bayesian evidence **PyAutoLens** computes assumes
the noise map is the per-pixel RMS of *independent* Gaussian noise. Image combination breaks that
assumption: drizzling (Fruchter & Hook 2002, PASP 114, 144,
https://ui.adsabs.harvard.edu/abs/2002PASP..114..144F) shares each input pixel among neighbouring
output pixels, so the per-pixel RMS underestimates the true uncertainty of any larger-scale
structure. The canonical treatment is the scalar correlation factor of Casertano et al. (2000,
AJ 120, 2747), which **PyAutoReduce** applies to every resampled noise map and records in the
provenance. Get this wrong and every posterior width in your lens model is miscalibrated — the
SL2S survey went as far as avoiding resampling entirely to keep its noise uncorrelated (Gavazzi
et al. 2012, https://arxiv.org/abs/1202.3852), the cleanest precedent for how seriously the
lensing literature takes this. The guide `scripts/guides/noise_maps.py` develops the full story.

Everything else a reduction does — cosmic-ray rejection, astrometric alignment, sky subtraction,
cutout packaging — exists in service of those two products arriving accurate and honest.

__A Reduction Is Declared, Not Scripted__

The **PyAutoReduce** public API is deliberately tiny — two names:

```python
from autoreduce import TargetSpec, reduce_target
```

You do not write a reduction script that chains pipeline calls. You *declare* the reduction as a
frozen `TargetSpec` — target name, coordinates, instrument, and every literature-contested dial
(pixel scale, pixfrac, kernel, cosmic-ray method, PSF options) — and call `reduce_target` once.
The pipeline is a pure function of the spec plus the archive: re-running the same spec reproduces
the same dataset, modulo upstream reference-file updates, which the provenance record captures.
This is what makes a reduced sample reproducible — commit the spec (one small YAML per target,
see `dataset/README.md`), never the FITS.

The corollary is the pipeline's failure philosophy: it fails loudly. A NaN in the noise map, a
zero-weight pixel inside the cutout, a star field too poor to build a PSF — these crash the
reduction with an explanatory error rather than shipping a silently-degraded dataset. A product
that reaches `output/` is one the pipeline is prepared to stand behind.

__Imports__

We import the two public **PyAutoReduce** names, plus the standard libraries for reading and
plotting the FITS products afterwards. You'll see these imports in the majority of workspace
examples.
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

Two directories matter:

- `cache/` — downloaded exposures and CRDS reference files, re-used across runs. Delete it and
  the next run simply re-downloads; keep it and re-running a target is fast and can even work
  offline (see `sync_references` in `scripts/guides/target_spec.py`).
- `output/` — the reduced datasets, one folder per target.

Both are gitignored: no FITS is ever committed to this repository.
"""
WORKSPACE = Path(__file__).resolve().parents[1]
CACHE_ROOT = WORKSPACE / "cache"      # Downloaded exposures + CRDS references (re-used across runs).
OUTPUT_ROOT = WORKSPACE / "output"    # Reduced datasets, one folder per target.

print(f"Workspace root: {WORKSPACE}")
print(f"Exposure cache: {CACHE_ROOT}")
print(f"Reduced output: {OUTPUT_ROOT}")

"""
__Target Spec__

Our target is SDSS J0008-0004, a strong lens from the SLACS survey — the survey whose HST ACS
reductions (Bolton et al. 2006, ApJ 638, 703; Bolton et al. 2008, ApJ 682, 964,
https://arxiv.org/abs/0805.1931) are the quality bar the **PyAutoReduce** HST pipeline was
validated against. Its F814W exposures were taken under HST proposal 10886, and pinning the
acquisition to that proposal keeps exposures from neighbouring programmes out of the stack.

Everything else stays at the ACS/WFC defaults, which are themselves the SLACS conventions —
a 0.05"/pixel output grid and drizzle pixfrac 0.8. The spec below is complete: this declaration,
plus the MAST archive, *is* the reduction.
"""
spec = TargetSpec(
    name="slacs0008-0004",     # Names the output folder: output/slacs0008-0004/.
    ra=2.012333,               # Target right ascension, degrees (J2000).
    dec=-0.068944,             # Target declination, degrees (J2000).
    proposal_ids=("10886",),   # Restrict acquisition to the SLACS HST programme's exposures.
    # Defaults doing the work (see scripts/guides/target_spec.py for every dial):
    # instrument="acs_wfc"     ACS/WFC: CTE-corrected _flc exposures, AstroDrizzle combine.
    # filter_name="F814W"      The SLACS modeling band.
    # cutout_shape=(281, 281)  ~14" postage stamp at the output scale.
    # final_scale=0.05         Output pixel scale, arcsec/pix — the SLACS convention.
    # final_pixfrac=0.8        Drizzle drop size — the SLACS convention.
    # cr_method="driz_cr"      STScI-default cosmic-ray rejection at the combine.
    # psf_shape=(21, 21)       Compact PSF for convolution; psf_full_shape=(61, 61) for wings.
)

print(f"Declared reduction: {spec.name} ({spec.instrument}, {spec.filter_name})")

"""
__The Output Contract__

Every imaging reduction emits the same five products into `output/<name>/`, and they are exactly
what `al.Imaging.from_fits` consumes:

- `data.fits` — the science cutout, drizzled to the modeling pixel scale, WCS and units intact
  in the header (electrons/second for HST).
- `noise_map.fits` — the matching per-pixel RMS map, correlated-noise corrected. Pixels the
  pipeline masked carry an effectively infinite noise value (1e8) with the data zeroed, so no
  separate mask file is needed downstream.
- `psf.fits` — a compact (21x21) unit-normalised PSF for fit convolution.
- `psf_full.fits` — an extended (61x61) PSF carrying the wings, for science that needs them.
- `reduction.json` — the full provenance record: which exposures, which dials, every diagnostic,
  and the software versions that produced the dataset.

The guide `scripts/guides/output_contract.py` walks this contract file by file and
`reduction.json` block by block — read it after this script.

__Run The Reduction__

One call runs the whole pipeline: query MAST for the calibrated exposures, download them into the
cache, sync the CRDS reference files, verify the Gaia-tied archive astrometry, drizzle the
exposures onto the output grid with cosmic-ray rejection, construct the noise map, build the PSF
from field stars, and package the cutouts.

The return value is the provenance record — the same content written to `reduction.json`.
"""
print(
    """
    Starting the reduction. On the first run this downloads the SLACS J0008-0004 F814W
    exposures (a few hundred MB) from MAST plus CRDS reference files, then runs the full
    AstroDrizzle combine — expect roughly 10-20 minutes depending on your connection and
    machine. Re-runs re-use the cache and are much faster.
    """
)

record = reduce_target(
    spec,
    cache_root=CACHE_ROOT,     # Exposures + CRDS references cached here across runs.
    output_root=OUTPUT_ROOT,   # Products land in output_root/<spec.name>/.
)

out_dir = OUTPUT_ROOT / spec.name

print(f"Reduction complete: {out_dir}")

"""
__Provenance Walk__

A reduction you cannot audit is a reduction you cannot trust, so every run reports its
diagnostics in the returned record. Let's read the ones that matter most.

First, acquisition: how many exposures went into the stack.
"""
print(f"Exposures combined: {record['acquire']['n_exposures']}")

"""
Next, the drizzle diagnostics. The weight-map uniformity statistic (RMS/median of the drizzle
weight map, an STScI rule of thumb from the DrizzlePac Handbook,
https://hst-docs.stsci.edu/drizzpac) tells you whether the chosen `final_pixfrac` is compatible
with the dither pattern — values above ~0.2 mean the drops are too small for the dithers and the
coverage is speckled.

The correlated-noise factor R is the Casertano et al. (2000) correction applied to the noise
map — for the SLACS dials (pixfrac 0.8 at native scale) it is ~1.36, meaning drizzle correlation
inflates the effective per-pixel noise by 36%.
"""
print(json.dumps(record["drizzle"]["weight_uniformity"], indent=2))

print(f"Correlated-noise factor R: {record['drizzle']['correlated_noise_factor']:.4f}")

"""
The noise block records the recipe used — for HST, the per-pixel RMS is constructed from the
science and weight mosaics as R * sqrt(max(sci, 0)/exptime + 1/wht), the background-plus-Poisson
recipe of Bayer et al. (https://arxiv.org/abs/1803.05952, section 3.1) — plus the empirical
blank-sky RMS of the mosaic, which closes the loop: the constructed noise floor should match the
sky the mosaic actually shows.
"""
print(f"Noise recipe: {record['noise']['recipe']}")
print(f"Empirical background RMS: {record['noise']['empirical_background_rms']:.3e} e-/s")

"""
The PSF block records how the PSF was built: the method (the default is a photutils effective PSF
in the Anderson & King 2000 lineage, built from stars in the mosaic itself), how many stars
survived selection and fitting, and which drizzle pass fed the star finding (`star_source_pass` —
see `scripts/hst_acs/psf.py` for why that is a dial at all).
"""
print(json.dumps(record["psf"], indent=2))

"""
Finally the package block: the pixel scale and data units the modeling stack needs, read straight
from the provenance rather than remembered by you.
"""
print(f"Pixel scale: {record['package']['pixel_scale']} arcsec/pix")
print(f"Data units:  {record['package']['data_units']}")
print(f"Products:    {record['package']['products']}")

"""
__Plotting The Products__

Numbers first, but always look at your data. We plot the science cutout with arcsinh scaling
(which shows the faint arcs and the bright deflector core in one stretch), the noise map, and the
compact PSF on a log stretch, saving each as a PNG under the target's output folder.
"""
plots_dir = out_dir / "plots"
plots_dir.mkdir(exist_ok=True)

data = fits.getdata(out_dir / "data.fits").astype(float)
noise_map = fits.getdata(out_dir / "noise_map.fits").astype(float)
psf = fits.getdata(out_dir / "psf.fits").astype(float)

plt.figure(figsize=(6, 6))
plt.imshow(np.arcsinh(data / np.nanstd(data)), origin="lower", cmap="magma")
plt.colorbar(label="arcsinh(data / rms)")
plt.title(f"{spec.name} data.fits ({spec.filter_name})")
data_png = plots_dir / "data.png"
plt.savefig(data_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {data_png.resolve()}")

plt.figure(figsize=(6, 6))
# Masked pixels carry the 1e8 sentinel; clip the display to the physical range.
plt.imshow(
    np.clip(noise_map, None, np.nanpercentile(noise_map[noise_map < 1.0e7], 99)),
    origin="lower",
    cmap="viridis",
)
plt.colorbar(label="RMS noise (e-/s)")
plt.title(f"{spec.name} noise_map.fits")
noise_png = plots_dir / "noise_map.png"
plt.savefig(noise_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {noise_png.resolve()}")

plt.figure(figsize=(6, 6))
plt.imshow(np.log10(np.clip(psf, 1e-8, None)), origin="lower", cmap="cividis")
plt.colorbar(label="log10(PSF)")
plt.title(f"{spec.name} psf.fits (21x21, sums to {psf.sum():.4f})")
psf_png = plots_dir / "psf.png"
plt.savefig(psf_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {psf_png.resolve()}")

"""
__The SLACS Quality Bar__

How do we know this reduction is *right*? The **PyAutoReduce** HST pipeline was validated against
the legacy SLACS modeling datasets — reductions that have supported fifteen-plus years of
published lens models (Bolton et al. 2008, https://arxiv.org/abs/0805.1931). The validation
principle is worth internalising: *the reduction is correct when the science is invariant under
it* — a lens model fitted to the new reduction should infer the same physics as one fitted to the
legacy data.

Two honest numbers come out of that comparison, both documented rather than hidden:

- **Data parity ~0.94-0.96.** The new mosaics carry ~6% less flux than the legacy cutouts — a
  known, accepted offset between the modern CTE-corrected `_flc` + AstroDrizzle chain and the
  legacy reduction, recorded in the design docs rather than tuned away.

- **Noise ~30% above legacy, by design.** The legacy SLACS noise maps do not carry the Casertano
  correlated-noise factor; ours do (R ~ 1.36 at the SLACS dials). The new noise maps are
  *supposed* to be higher — they are the honest ones (see `scripts/guides/noise_maps.py`).

The comparison machinery is public: `autoreduce.validation.registered_ratios` registers a new
data/noise pair onto a reference pair at sub-pixel precision and reports median bright-pixel
ratios. If you have a trusted previous reduction of your own target, run the same parity check
against it — it is the fastest way to catch a mistake in either.
"""
from autoreduce.validation import registered_ratios  # noqa: E402  (used when you have a reference dataset)

print(
    "Parity idiom (needs a reference dataset):\n"
    "  registered_ratios(new_data, new_noise, ref_data, ref_noise)\n"
    "  -> data_ratio_median ~0.94-0.96 and noise_ratio_median ~1.3 vs legacy SLACS is expected."
)

"""
__Load Into PyAutoLens__

This is the seam the whole workspace exists to serve: the four FITS files plus the pixel scale
from the provenance are exactly the inputs `al.Imaging.from_fits` takes. The committed example
datasets that `autolens_workspace` models — its `imaging/data_preparation` examples spell out the
standards (electrons/second units, centred postage stamp, RMS noise map, odd unit-normalised
PSF) — are files produced by scripts exactly like this one.

The import is guarded: **PyAutoReduce** never depends on the modeling stack, and the reduction
above is complete whether or not **PyAutoLens** is installed.
"""
try:
    import autolens as al
except ImportError:
    al = None
    print(
        "PyAutoLens is not installed (pip install autolens), so the final loading step is "
        "skipped — the reduction itself is complete and the products above are ready for "
        "modeling on any machine that has it."
    )

if al is not None:
    dataset = al.Imaging.from_fits(
        data_path=out_dir / "data.fits",
        noise_map_path=out_dir / "noise_map.fits",
        psf_path=out_dir / "psf.fits",
        pixel_scales=record["package"]["pixel_scale"],  # From provenance, never from memory.
    )
    print(
        f"Loaded {spec.name} into PyAutoLens: shape {dataset.data.shape_native}, "
        f"pixel scale {dataset.pixel_scales}."
    )
    print(
        "Next: model it with autolens_workspace/scripts/imaging/start_here.py — "
        "point its Imaging.from_fits paths at this output folder."
    )

"""
__Where To Go Next__

Each instrument folder follows the same shape — a `start_here.py` running the default pipeline on
that instrument's validation anchor, a `step_by_step.py` teaching what each stage does with the
instrument handbook and literature, and (where the instrument supports them) `psf.py`,
`individual.py` and `simulator.py` deep dives. Route yourself by your data:

- `scripts/hst_acs/` — HST ACS/WFC, the reference pipeline this script just ran. Its
  `step_by_step.py` teaches the AstroDrizzle chain against the ACS Data Handbook
  (https://hst-docs.stsci.edu/acsdhb), and `dials.py` is the drizzle trade study — what
  scale/pixfrac/kernel/CR-method choices cost and what the literature disagrees on.

- `scripts/hst_wfc3_uvis/` — HST WFC3/UVIS, anchored to the Bayer et al.
  (https://arxiv.org/abs/1803.05952) F390W reduction of SDSS J0252+0039 at the native 0.0396"
  scale with pixfrac 1.0. UVIS is where the STARRED super-sampled PSF back-end
  (Millon et al. 2024, https://arxiv.org/abs/2402.08725) earns its keep.

- `scripts/hst_wfc3_ir/` — HST WFC3/IR: no CTE correction exists, cosmic rays are already
  rejected by up-the-ramp fitting, and the undersampled 0.128" pixels drive the dither-and-
  drizzle strategy — including the honest rule that few-dither IR data at sub-native scales
  needs pixfrac 1.0 or a coarser grid.

- `scripts/jwst_nircam/` — JWST NIRCam through the calwebb_image3 pipeline, anchored to the
  COSMOS-Web ring (Mercier et al. 2024, https://arxiv.org/abs/2309.15986). Two deltas from HST
  to internalise: the noise map is *read* from the pipeline's propagated ERR array rather than
  constructed, and the data stays in surface-brightness units of MJy/sr — not electrons/second.
  `multi_band.py` builds the four-band dataset for multi-wavelength modeling.

- `scripts/keck_nirc2/` — Keck NIRC2 laser-guide-star AO, anchored to B1938+666 — the SHARP
  survey's ring (Lagattuta et al. 2012, MNRAS 424, 2800) in which Vegetti et al. (2012, Nature
  481, 341) detected a dark satellite. Ground-based means the workspace's extra stages appear —
  darks/flats, running-sky subtraction, epoch-matched distortion — and the AO PSF is delivered
  as *provisional candidates*, because final PSF selection belongs to the modeling stage
  (Chen et al. 2016, https://arxiv.org/abs/1601.01321).

- `scripts/alma/` — ALMA visibilities: no images at all. The dataset is the calibrated
  visibilities themselves, extracted to the `(N_vis, 2)` triplet `al.Interferometer.from_fits`
  loads, because fitting in the uv-plane keeps the noise independent and the likelihood
  well-defined (Hezaveh et al. 2016, https://arxiv.org/abs/1601.01388; Dye et al. 2018,
  https://arxiv.org/abs/1705.05413).

- `scripts/surveys/` — Legacy Surveys / SDSS / Pan-STARRS cutouts around a target for colour
  context. Deliberately *not* modeling data: no PSF ships, and no noise map beyond what the
  service provides.

- `scripts/guides/` — the cross-instrument deep dives: `output_contract.py` (the products in
  depth), `noise_maps.py` (recipes, the Casertano factor, closure checks — mostly offline) and
  `target_spec.py` (every dial; fully offline).

__Wrap Up__

This script reduced a real SLACS strong lens from the MAST archive to a modeling-ready dataset
with one declared spec and one function call, audited the reduction through its provenance
record, and handed the products to **PyAutoLens**.

The two ideas to carry forward: reduction quality *is* modeling quality (the PSF and the noise
map are where it concentrates), and a reduction should be a declaration you can commit, not a
script you can lose.

The following locations of the workspace are good places to checkout next:

- `scripts/guides/output_contract.py`: the five output products in depth, on the dataset this script just produced.
- `scripts/guides/noise_maps.py`: the noise story — recipes per instrument, the Casertano factor, closure diagnostics.
- `scripts/guides/target_spec.py`: every `TargetSpec` dial, spec YAML round-trips, validation guard rails (runs offline).
- `scripts/hst_acs/step_by_step.py`: what each pipeline stage actually did to the exposures reduced above.
- `autolens_workspace/scripts/imaging/start_here.py`: model the dataset this script produced.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness. This script downloads
exposures from MAST and runs the drizzlepac stack, so it is excluded from offline smoke runs.

ENV: network
"""
