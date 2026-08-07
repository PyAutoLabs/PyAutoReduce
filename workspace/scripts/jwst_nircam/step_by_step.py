"""
JWST NIRCam: Step By Step
=========================

The `start_here.py` example reduced the COSMOS-Web ring in one `reduce_target` call.
This script slows down and teaches what actually happens to a NIRCam exposure on its way
from detector ramps to your `data.fits` — the three stages of the JWST Science
Calibration Pipeline, which of them **PyAutoReduce** re-runs (only the third), and how to
read the evidence of every choice out of `reduction.json` afterwards.

The canonical references are the `jwst` package documentation
(https://jwst-pipeline.readthedocs.io/en/stable/, stage index at
https://jwst-pipeline.readthedocs.io/en/stable/jwst/pipeline/main.html) and the JDox
pipeline pages (https://jwst-docs.stsci.edu/jwst-science-calibration-pipeline). Links to
the specific pages accompany each stage below — this workspace's whole purpose is that
every reduction step points at the document that defines it.

The reduction at the end re-uses the cache from `start_here.py`, so run that first if you
want the fast path.

__Contents__

- **The Three Stages:** The calwebb pipeline architecture and where **PyAutoReduce** enters it.
- **Imports:** Import **PyAutoReduce** and the supporting libraries.
- **Paths:** Anchor the cache and output folders to the workspace root.
- **Stage 1 — Detector1:** Up-the-ramp fitting, jump detection, snowballs, 1/f noise, superbias and reference pixels.
- **Stage 2 — Image2:** WCS assignment, flat fielding and the photometric calibration to MJy/sr.
- **Stage 3 — Image3, As PyAutoReduce Runs It:** Association building, tweakreg, skymatch, outlier detection, and the resample dial mapping.
- **CRDS:** Lazy reference syncing through CRDS_PATH, and context pinning for reproducibility.
- **The Reduction:** Run the ring reduction and collect the record.
- **Reading the Evidence:** The drizzle and noise blocks of reduction.json, stage by stage.
- **The Casertano Factor, Standalone:** Compute R yourself with the public helper.
- **Wrap Up:** Where to go next.

__The Three Stages__

Every JWST image passes through three pipeline stages at STScI before you ever see it:

1. **calwebb_detector1** — detector-level corrections on the raw up-the-ramp readouts
   (`_uncal.fits` -> `_rate.fits`): one file per exposure, units of DN/s.
2. **calwebb_image2** — per-exposure calibration (`_rate.fits` -> `_cal.fits`): WCS,
   flat field, flux calibration to MJy/sr.
3. **calwebb_image3** — ensemble combination (`_cal.fits` -> `_i2d.fits` mosaic):
   alignment, background matching, outlier rejection, drizzle-style resampling.

**PyAutoReduce** enters at level 2: it downloads the `_cal` products from MAST and
re-runs only stage 3. Stages 1 and 2 are therefore *pure STScI defaults* — the archive's
own processing, with the calibration reference files current at retrieval time. This is
the defaults-first principle doing real work: the detector-level corrections are exactly
what every other JWST paper uses, and the lensing-specific decisions all live in the one
stage **PyAutoReduce** controls. You still need to understand stages 1 and 2, though,
because their fingerprints (removed cosmic rays, error budgets, surface-brightness
units) are all over the products you model.

__Imports__
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

from autoreduce import TargetSpec, reduce_target
from autoreduce.instruments import nircam_adapter_for_filter
from autoreduce.noise.rms import casertano_r, empirical_background_rms
from autoreduce.drizzle.diagnostics import weight_uniformity

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
__Stage 1 — Detector1__

Documentation:
https://jwst-pipeline.readthedocs.io/en/stable/jwst/pipeline/calwebb_detector1.html

JWST's near-infrared detectors do not take a single exposure the way a CCD does. Each
pixel is read out *non-destructively* many times as charge accumulates — an
"up-the-ramp" sequence of groups — and the flux is the fitted *slope* of that ramp.
calwebb_detector1 turns the raw ramp cube (`_uncal.fits`) into a slope image
(`_rate.fits`) through a chain of NIR steps: group_scale, dq_init, saturation, ipc,
superbias, refpix, linearity, persistence, dark_current, jump, ramp_fit and gain_scale.
The ones whose consequences you will meet downstream:

- **superbias / refpix** — subtract the fixed bias structure per group, then use the
  4-pixel border of light-insensitive reference pixels around each 2048x2048 sensor to
  track and remove bias drifts during the integration.

- **jump** — detects cosmic-ray hits as discontinuities between consecutive group
  differences (two-point difference statistics). This is why JWST frames need no
  L.A.Cosmic-style CR rejection: a hit corrupts one *group*, not the exposure.

- **ramp_fit** — fits an optimally-weighted (Fixsen et al. 2000, PASP 112, 1350) slope
  to each pixel's ramp, split into segments at flagged jumps. A pixel hit by a cosmic
  ray mid-ramp still yields a valid rate from its clean segments; even pixels that
  saturate late in the ramp yield rates from the early groups. Cosmic rays are thus
  *removed*, not merely flagged — a fact that shapes the DQ policy for frame products
  (see `individual.py`).

- **Snowballs** — the spectacular failure mode of jump detection. Large cosmic-ray
  events (likely secondary particle showers) saturate a core and splash charge over
  hundreds to thousands of pixels
  (https://jwst-docs.stsci.edu/data-artifacts-and-features/snowballs-and-shower-artifacts).
  The jump step's `expand_large_events` option fits enclosing circles around big events
  and expands the flagged region to cover the halo. Early-cycle data (including the
  first COSMOS-Web and CEERS epochs) needed manual snowball masking; modern pipeline
  builds handle most of it automatically.

- **1/f noise** — correlated readout noise from the SIDECAR ASIC readout electronics,
  appearing as faint banding along the slow-read axis, different per amplifier
  (https://jwst-docs.stsci.edu/known-issues/1-f-noise). The community fix — Chris
  Willott's `image1overf.py` (https://github.com/chriswillott/jwst), applied by CEERS
  and many survey teams to the rate files — worked well enough that the algorithm was
  adopted into the official pipeline as the `clean_flicker_noise` step. A community
  correction graduating into the pipeline is the JWST calibration story in miniature,
  and it is why **PyAutoReduce** is comfortable trusting archive defaults: the defaults
  keep absorbing the community's fixes. (The `_cal` files this workspace's anchor
  reductions use predate routine 1/f cleaning — the residual banding is part of the
  parity stance discussed in `start_here.py`.)

__Stage 2 — Image2__

Documentation:
https://jwst-pipeline.readthedocs.io/en/stable/jwst/pipeline/calwebb_image2.html

calwebb_image2 calibrates each slope image individually into a `_cal.fits` file:

- **assign_wcs** — attaches the full distortion-aware WCS (a gwcs object serialised in
  the file, alongside a FITS-approximation SIP solution), tied to the spacecraft
  pointing solution.
- **flat_field** — divides by the CRDS flat for the detector/filter.
- **photom** — converts DN/s to **MJy/sr** using the CRDS photometric calibration
  (the `PHOTMJSR` scale factor recorded in the header). From this point on, the data
  are surface brightnesses; the pixel solid angle `PIXAR_SR` rides in the header so
  fluxes remain recoverable (flux [Jy] = SB [MJy/sr] x 1e6 x PIXAR_SR).

The `_cal` file also carries the per-pixel error budget as separate variance planes —
`VAR_POISSON`, `VAR_RNOISE`, `VAR_FLAT` — with `ERR = sqrt(sum)`. These planes are the
foundation of the JWST noise story: they propagate through resampling, which is why
**PyAutoReduce** reads its noise map rather than constructing one.

__Stage 3 — Image3, As PyAutoReduce Runs It__

Documentation:
https://jwst-pipeline.readthedocs.io/en/stable/jwst/pipeline/calwebb_image3.html

Stage 3 is where **PyAutoReduce** takes over, because the ensemble-combination choices
(output grid, drop size, kernel, orientation) are exactly the ones lens modeling cares
about. What it does, in order:

1. **Association building.** Image3 consumes an ASN file — a JSON manifest declaring
   which `_cal` exposures form one product. **PyAutoReduce** builds this from the
   cached exposures that survived its footprint filter and hands the pipeline the list.

2. **tweakreg** — relative alignment of the exposures via source catalogs, then
   absolute alignment against Gaia DR3 (the default `abs_refcat`). A deep-field
   caveat worth knowing: in fields like COSMOS, few Gaia stars land on a single
   NIRCam pointing, which is why survey teams align to Gaia-*tied* external catalogs
   instead (the COSMOS astrometric frame; CEERS aligns to HST catalogs tied to Gaia).
   **PyAutoReduce** runs tweakreg with defaults — the MAST `_cal` headers already
   carry good a-priori pointing, and residual relative alignment is what matters at
   lens-cutout scale.

3. **skymatch** — measures and equalises background levels between exposures. By
   default it *records* the matched levels (the `BKGLEVEL` header key) rather than
   subtracting them from your data.

4. **outlier_detection** — the stack-based second line of cosmic-ray defence: resample
   all exposures, median them, blot the median back to each frame, and flag deviant
   pixels (residual CRs, snowball residue, hot pixels, persistence). Only overlapping
   dithers can be cleaned this way. The flagged per-exposure files are the `_crf`
   products, which the frame-products mode packages (`individual.py`).

5. **resample** — the drizzle analogue (Fruchter & Hook 2002, PASP 114, 144), combining
   all exposures onto one output grid. This is where your `TargetSpec` dials land.

The dial mapping is one-to-one and recorded in provenance:

| TargetSpec dial      | resample argument | value here                       |
|----------------------|-------------------|----------------------------------|
| `final_scale`        | `pixel_scale`     | 0.06 (LW COSMOS-Web convention)  |
| `final_pixfrac`      | `pixfrac`         | 1.0 (full drop)                  |
| `final_kernel`       | `kernel`          | "square"                         |
| (always)             | `rotation`        | 0.0 — north-up output            |
| (adapter default)    | `weight_type`     | "ivm" — inverse-variance weights |

Two implementation details you would otherwise discover the hard way: **PyAutoReduce**
runs the pipeline with `in_memory=False` (image3 holds every resampled model in RAM
otherwise, and exhausts it on large dither sets), and the multi-extension `_i2d` output
(SCI/ERR/CON/WHT/VAR_*) is normalised into standalone `sci`/`wht`/`err` files so every
downstream stage (noise, PSF, package) stays backend-agnostic — the same three files an
HST AstroDrizzle run produces.

__CRDS__

Every pipeline step above pulls its reference files (flats, distortion maps, photometric
calibrations...) from CRDS (https://jwst-crds.stsci.edu). The `jwst` package syncs
references *lazily*: when a step needs a file it checks the local cache under
`CRDS_PATH` and downloads on miss. **PyAutoReduce** points `CRDS_PATH` into its cache
root, so references download once and persist — they are never evicted, even when
exposure caches are.

Reproducibility note: CRDS is versioned by *context* (a pmap file naming every reference
in force). By default you get the latest operational context, which advances as STScI
delivers new calibrations. For strictly reproducible reruns, pin `CRDS_CONTEXT` in your
environment; either way `reduction.json` records the software versions used. Relatedly,
the `jwst` package itself is pinned at 1.14.0 in this stack — the provenance records the
version, so a future upgrade is a visible, auditable event rather than silent drift.

__The Reduction__

Now run it: the same F277W ring spec as `start_here.py`, so a warm cache makes this a
combine-only rerun.
"""
band = "F277W"
adapter = nircam_adapter_for_filter(band)

spec = TargetSpec(
    name=f"cosmos_web_ring_{band.lower()}",  # same name as start_here.py -> shares its exposure cache
    ra=150.10048,  # the COSMOS-Web ring (Mercier et al. 2024)
    dec=1.89301,
    instrument=adapter.key,  # "nircam_lw"
    filter_name=band,
    proposal_ids=("1727",),  # COSMOS-Web only
    final_scale=adapter.recommended_final_scale,  # 0.06"/pixel — maps to resample pixel_scale
    final_pixfrac=1.0,  # maps to resample pixfrac
    final_kernel="square",  # maps to resample kernel (the default, stated here for the mapping table)
    cutout_shape=(209, 209),  # ~12.5" at 0.06"/pixel
)

print(
    """
    Running calwebb_image3 on the COSMOS-Web ring (F277W).

    With a warm cache from start_here.py this skips the MAST download and runs the
    combine + noise + psf + package stages — expect several minutes. On a cold cache,
    add the download time (a few GB).
    """
)

record = reduce_target(spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)

out_dir = OUTPUT_ROOT / spec.name

"""
__Reading the Evidence__

Every claim made above is checkable in the record. First the `drizzle` block — the
backend, the exact resample kwargs, and the `_crf` bookkeeping from outlier_detection:
"""
drizzle = record["drizzle"]

print("Backend:", drizzle["backend"])  # "jwst_image3"
print("Resample kwargs as run:", json.dumps(drizzle["resample_kwargs"], indent=2))
print("Normalised ERR mosaic:", drizzle["err_path"])
print(f"Outlier-flagged _crf exposures: {len(drizzle['crf_paths'])}")
for path in drizzle["crf_paths"][:3]:
    print("  ", Path(path).name)

"""
The `crf_paths` list is the receipt that outlier_detection ran and saved its
per-exposure flagged products — `individual.py` turns those into modelable native-frame
cutouts. The `err_path` is the resampled ERR mosaic the noise stage read.

Next the `noise` block. The recipe string states the JWST policy in one line — read the
propagated ERR, multiply by R — and the consistency numbers close the loop against the
actual sky in the mosaic:
"""
noise_block = record["noise"]

print("Recipe:", noise_block["recipe"])  # "R * ERR (propagated by calwebb_image3 resample)"
print("R:", noise_block["correlated_noise_factor"])
print("Empirical sky RMS:", noise_block["empirical_sky_rms"])
print("ERR floor (5th percentile, pre-R):", noise_block["err_5th_percentile_pre_R"])
print("sky_over_err_floor:", noise_block["sky_over_err_floor"])

"""
`sky_over_err_floor` near 1 means the pipeline's propagated error budget and the
measured blank-sky fluctuations agree — the reduction is internally consistent. A value
far from 1 would mean the upstream variance planes and the data disagree; that is
investigated (wrong exposures mixed in? background structure?), never absorbed into a
fudge factor.

__The Casertano Factor, Standalone__

R is not a black box — it is a closed-form function of the two resampling dials, from
Casertano et al. (2000, AJ 120, 2747; also the DrizzlePac handbook). You can compute it
yourself with the same public helper the pipeline uses, and check it against the record:
"""
scale_ratio = spec.final_scale / adapter.native_scale  # s: output / native pixel size

R = casertano_r(pixfrac=spec.final_pixfrac, scale_ratio=scale_ratio)

print(f"scale_ratio s = {scale_ratio:.4f}, pixfrac p = {spec.final_pixfrac}")
print(f"casertano_r -> R = {R:.4f}")
print(f"recorded    -> R = {noise_block['correlated_noise_factor']:.4f}")

"""
Note the regime: here s < p (0.06/0.063 ~ 0.95 output pixels per native pixel, with a
full drop), which puts us on the fine-grid branch of the formula where correlation —
and therefore R — grows as the output grid gets finer than the drop. Shift-and-add
(p = 1 at s = 1) gives the textbook R = 1.5; our slightly finer grid gives a little
more. This is the price of resampling: adjacent output pixels share input flux, so
per-pixel errors understate aperture errors by R.

You can also re-derive the two mosaic diagnostics from the normalised products the
combine stage left in the work directory — the same public helpers the pipeline calls:
"""
sci = fits.getdata(out_dir / "data.fits").astype(float)
wht_files = sorted((out_dir / "work").rglob("*_wht.fits"))

print("Empirical background RMS of the cutout:", empirical_background_rms(sci))
if wht_files:
    wht = fits.getdata(wht_files[0]).astype(float)
    print("Weight uniformity of the full mosaic:", weight_uniformity(wht))

"""
Finally, a picture of the evidence: the ERR mosaic's distribution against the sky RMS —
the closure check drawn rather than tabulated.
"""
plot_dir = out_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

noise_map = fits.getdata(out_dir / "noise_map.fits").astype(float)
good = noise_map[np.isfinite(noise_map) & (noise_map < 1.0e7)]

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(good, bins=100, color="steelblue", log=True)
ax.axvline(
    noise_block["correlated_noise_factor"] * noise_block["err_5th_percentile_pre_R"],
    color="k", ls="--", label="R x ERR floor",
)
ax.axvline(
    noise_block["correlated_noise_factor"] * noise_block["empirical_sky_rms"],
    color="crimson", ls=":", label="R x empirical sky RMS",
)
ax.set_xlabel("noise_map value (MJy/sr)")
ax.set_ylabel("pixels")
ax.set_title("Noise-map distribution vs the blank-sky closure")
ax.legend()
fig.tight_layout()
plot_path = plot_dir / "step_by_step_noise_closure.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)

print(f"Noise-closure plot saved to: {plot_path.resolve()}")

"""
__Wrap Up__

You now know what each calwebb stage does to a NIRCam exposure, which one
**PyAutoReduce** re-runs (Image3, with the lensing dials mapped onto resample), and how
to audit every choice from the provenance record: the resample kwargs, the `_crf`
receipts, the read-don't-construct noise recipe, the Casertano R you can recompute
yourself, and the blank-sky closure that certifies internal consistency.

The following locations of the workspace are good places to checkout next:

- `scripts/jwst_nircam/multi_band.py`: run all four COSMOS-Web bands and compare their closures.
- `scripts/jwst_nircam/individual.py`: package the `_crf` exposures as native-frame products.
- `scripts/jwst_nircam/psf.py`: the PSF stage in the same depth as the combine stage here.
- `scripts/guides/noise_maps.py`: the noise recipes across all instruments, and why chi^2 cares.

__Env__ (Developer Only)

ENV: network
"""
