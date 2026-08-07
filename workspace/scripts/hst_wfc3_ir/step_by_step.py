"""
Step By Step: HST WFC3/IR
=========================

This script walks the WFC3/IR reduction stage by stage, teaching what happens to the data at
each step of the pipeline — as a *delta* against the ACS/WFC reference and the UVIS delta
folder beside this one. The drizzle/noise/PSF/packaging machinery is shared across all HST
channels and documented once in `hst_acs/step_by_step.py`; the UVIS-specific CCD story
(CTE, post-flash, `iref`) is in `hst_wfc3_uvis/step_by_step.py`. Here the focus is the parts
of the pipeline the infrared detector rewrites: the calibration chain that turns
non-destructive ramps into count-rate images, cosmic-ray handling split across two stages,
and the coverage economics of few-dither data on a fine output grid.

If you have not read `hst_wfc3_ir/start_here.py`, do that first — it establishes the
detector physics, the 0.065"/pixel drizzle choice and the zero-weight-speckle finding this
script keeps building on.

__Contents__

- **Imports:** Import the required Python libraries.
- **Paths:** Anchor every path to the workspace root, absolutely.
- **Stage 1, Acquisition:** `_flt` products, `iref` references, and no CTE variant to choose.
- **Stage 2, Calibration:** The `calwf3` IR chain — from ramps to count-rate images.
- **Stage 3, Alignment:** Shared with ACS — cross-references only.
- **Stage 4, Drizzle:** Why `driz_cr` still runs on CR-cleaned exposures, and few-dither reality.
- **Stage 5, Noise:** The shared recipe on data that is already e-/s, with the fine-grid R.
- **Stage 6, PSF and Packaging:** Shared with ACS; the 78 ke- saturation delta.
- **The Reduction:** Run the pipeline and produce the evidence to read.
- **Reading the Evidence:** The WHT-uniformity reading and the per-stage provenance audit.
- **Units and Closure:** Confirm the e-/s contract from the headers and close the noise floor.
- **Wrap Up:** Summary of the script and next steps.

__Imports__

Alongside the **PyAutoReduce** entry points we import the adapter registry (to read the IR
channel's constants rather than hard-code them) and the public noise helpers demonstrated
standalone below.
"""
import json
from pathlib import Path

import numpy as np

from autoreduce import TargetSpec, reduce_target
from autoreduce import instruments
from autoreduce.noise.rms import casertano_r, empirical_background_rms

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the folder
containing `scripts/`). **PyAutoReduce** requires absolute paths: its drizzle step changes the
working directory internally, so relative paths would break.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"      # downloaded exposures + CRDS references (re-used across runs)
OUTPUT_ROOT = WORKSPACE / "output"    # reduced datasets, one folder per target

RA, DEC = 43.188375, 0.666222  # SDSS J0252+0039 — the IR anchor field (see start_here.py).

IR = instruments.get("wfc3_ir")

print(f"wfc3_ir adapter: native {IR.native_scale}\"/pix, recommends {IR.recommended_final_scale}\"/pix")

"""
__Stage 1, Acquisition__

The acquire stage is the shared MAST machinery (`hst_acs/step_by_step.py`, Stage 1): query
the archive, filter to direct program observations — dropping the HAP duplicates, exactly as
demonstrated live in `hst_wfc3_uvis/step_by_step.py` — and download the calibrated exposures
into the per-target cache. The IR deltas:

- The calibrated product is the **`_flt`**, full stop. On the CCD channels you choose the
  CTE-corrected `_flc` over the uncorrected `_flt`; on IR *no CTE correction exists* —
  a HgCdTe array does not clock charge across the detector, so there is nothing to correct
  and no `_flc` variant in the archive. One entire class of systematics (and one entire
  decision) is simply absent.
- Reference files come from CRDS (https://hst-crds.stsci.edu) under the WFC3 environment key
  `iref`, shared with UVIS.

__Stage 2, Calibration__

The `calwf3` IR chain (WFC3 Data Handbook, https://hst-docs.stsci.edu/wfc3dhb; Python
tooling at https://wfc3tools.readthedocs.io) is where the infrared detector rewrites the
rules. Each raw IR exposure is a MULTIACCUM sequence — the detector read non-destructively
many times up the ramp — and calibration turns that sequence into one count-rate image:

- **Reference-pixel correction:** the detector's border pixels are blind to light and track
  the electronic bias level read by read; subtracting them removes drifts a CCD would handle
  with overscan.
- **Zero-read subtraction:** the first read is taken as the ramp's zero point and subtracted,
  removing pedestal structure present before integration effectively began.
- **Non-linearity correction:** HgCdTe pixels respond non-linearly as they fill; each read is
  corrected up to the saturation flag (the effective full well is ~78 ke-).
- **Dark subtraction:** a MULTIACCUM dark matched to the same read sequence is subtracted.
- **Ramp fitting (CRCORR):** the pipeline fits a slope — counts per second — to each pixel's
  reads, and here cosmic rays are caught: a CR hit is a *jump* between two reads, glaringly
  inconsistent with a smooth ramp, so the fit detects the discontinuity, splits the ramp at
  it, and estimates the rate from the clean segments.

The output `_flt` is therefore already in **electrons per second**, already largely
CR-cleaned, with an ERR extension and DQ flags — a fundamentally more processed object than
a CCD `_flt`/`_flc`, which is a single destructive read in electrons.

__Stage 3, Alignment__

Identical to the other HST channels — the archive's Gaia-tied WCS is trusted, with a
cross-correlation diagnostic recorded per run. See `hst_acs/step_by_step.py` Stage 3.

__Stage 4, Drizzle__

The combination stage is the shared AstroDrizzle path (https://hst-docs.stsci.edu/drizzpac),
with two IR-flavored points worth understanding:

- **`driz_cr` still runs on multi-exposure stacks.** If the ramp fit already rejected the
  cosmic rays, why keep the median-stack CR flagging? Because ramp fitting is not perfect:
  hits in the final read interval, grazing hits, and pixels whose ramps were too short after
  splitting all leave *residue*. This is the defaults-first policy — the instrument team's
  standard stack processing is kept unless a lensing requirement demands otherwise — with
  the caveat recorded that it should be revisited if IR reductions ever show over-flagging.
- **Few-dither reality.** Everything about the IR drizzle is governed by how many
  sub-pixel-distinct dither positions the program took. Deep IR programs take four or more,
  which is what makes half-native grids work at all. Snapshots (like this anchor) take few —
  and on a fine grid the drops must stay wide (pixfrac 1.0) or coverage develops
  zero-weight speckle and the reduction refuses to package, the finding documented in
  `start_here.py`.

__Stage 5, Noise__

The recipe is the shared one (`guides/noise_maps.py`):

    sigma_i = R * sqrt(N_i / t_exp + 1 / W_i)

with the Poisson term built from the source counts and the total exposure time, and the
background term from the IVM drizzle weights. Nothing in the formula changes for IR — the
mosaic is in e-/s either way (on CCDs the drizzle's `final_units='cps'` conversion made it
so; on IR the `_flt` arrived that way) — but one input changes a lot: **R**. At the
recommended 0.065"/pixel from 0.128" native, the scale ratio s ~ 0.51 is *smaller than the
pixfrac* (1.0), which puts the reduction on the fine-grid branch of the Casertano et al.
(2000, AJ 120, 2747) formula, where correlation grows steeply as the grid refines. Compute
both branches with the public helper:
"""
s = IR.recommended_final_scale / IR.native_scale

print(f"scale ratio s = {s:.3f} (fine-grid branch applies, since s < pixfrac = 1.0)")
print(f"R at s = 1.0 (native-scale CCD convention): {casertano_r(pixfrac=1.0, scale_ratio=1.0):.3f}")
print(f"R at s = {s:.3f} (this IR reduction)        : {casertano_r(pixfrac=1.0, scale_ratio=s):.3f}")

"""
The delivered noise map is inflated by that larger R wholesale — the transparent price of
drizzling to a resolving grid. If a modeling analysis is noise-correlation sensitive (e.g.
the surface-brightness-anomaly science behind the UVIS anchor), the recorded R is the number
to propagate.

__Stage 6, PSF and Packaging__

Both shared with ACS: the PSF tiers, star selection and `psf_star_pass` are documented in
`hst_acs/psf.py` (with the star selection's saturation ceiling using the IR adapter's
~78 ke- effective full well), and the packaging contract — strict cutout, masked-by-noise,
finite-noise guard — in `guides/output_contract.py`. On IR the finite-noise guard is not a
formality; it is the enforcement mechanism of the pixfrac rule, as `start_here.py` showed.

__The Reduction__

Run the anchor reduction — the identical spec to `start_here.py`, so a warm cache makes this
fast and the two scripts share one output folder.
"""
spec = TargetSpec(
    name="j0252+0039_f160w",           # Same name as start_here.py -> shared cache + output.
    ra=RA,                             # Target right ascension in degrees.
    dec=DEC,                           # Target declination in degrees.
    instrument="wfc3_ir",              # IR adapter: _flt (e-/s), no CTE, native 0.128"/pix.
    filter_name="F160W",               # The IR snapshot's filter.
    final_scale=IR.recommended_final_scale,  # 0.065"/pix — half-native (see start_here.py).
    final_pixfrac=1.0,                 # The few-dither pixfrac rule (see start_here.py).
    cutout_shape=(215, 215),           # 14" footprint at 0.065"/pix, inside real coverage.
)

print(
    "Running the J0252+0039 F160W reduction (cached exposures re-used if "
    "start_here.py ran first; otherwise the first run downloads from MAST "
    "and takes tens of minutes)..."
)

record = reduce_target(spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)

out_dir = OUTPUT_ROOT / spec.name

"""
__Reading the Evidence__

The stages are internal to `reduce_target` (a reduction is one declared, reproducible unit),
so the audit runs on the provenance. First the acquisition and instrument blocks — confirm
the `_flt` product path and the exposure count; on a snapshot program expect a *small*
number, which is the root of everything coverage-related above.
"""
provenance = json.loads((out_dir / "reduction.json").read_text())

print(f"instrument block : {json.dumps(provenance['instrument'], indent=2)}")
print(f"n_exposures      : {provenance['acquire']['n_exposures']}")

"""
Now the reading this script has been building to: the **WHT-uniformity diagnostic** on a
few-dither IR mosaic. The statistic is RMS/median of the positive drizzle weights; the STScI
rule of thumb calls values above ~0.2 a sign the pixfrac is too small for the dither
pattern. Interpret the recorded value on a sliding scale:

- A well-dithered ACS program at pixfrac 0.8 measures ~0.07 — smooth, uniform coverage.
- This IR snapshot at pixfrac 1.0 sits *higher* — few dithers genuinely deliver less uniform
  depth, and the structured noise map reflects it honestly.
- The same data at pixfrac 0.8 does not merely score badly: it develops outright zero-weight
  pixels, and fails the finite-noise guard before this diagnostic even gets its say.

The uniformity number, the R factor and the guard together are the full coverage story of an
IR reduction — all three recorded, every run.
"""
print(f"drizzle block    : {json.dumps(provenance['drizzle'], indent=2)}")

"""
The noise block: the recorded `correlated_noise_factor` should match the fine-grid
`casertano_r` computed by hand above, and `empirical_background_rms` is the measured sky RMS
in e-/s. Both helpers are public, so we close the loop on the packaged cutout directly (the
cutout value differs slightly from the recorded full-mosaic value — same estimator,
different footprint):
"""
from astropy.io import fits

data = fits.getdata(out_dir / "data.fits").astype(float)

print(f"noise block                  : {json.dumps(provenance['noise'], indent=2)}")
print(f"R by hand (fine-grid branch) : {casertano_r(pixfrac=1.0, scale_ratio=s):.4f}")
print(f"empirical sky RMS (cutout)   : {empirical_background_rms(data):.5f} e-/s")

"""
Finally the PSF block — tier, star count, star-source pass — and the package block, whose
`pixel_scale` is what a **PyAutoLens** load consumes. Both contracts are the shared ones
(`hst_acs/psf.py`, `guides/output_contract.py`).
"""
print(f"psf block        : {json.dumps(provenance['psf'], indent=2)}")
print(f"package block    : {json.dumps(provenance['package'], indent=2)}")

"""
__Units and Closure__

Two final pieces of evidence, read straight off the products.

First, units. Stage 2 claimed the IR `_flt` arrives in electrons per second and the pipeline
keeps it that way (`final_units='cps'` is a no-op conversion for IR); the packaged header
should therefore carry `BUNIT` in count-rate units, with the total and per-exposure times
preserved alongside. This is worth checking on *every* dataset you hand to a lens model —
a units mistake at this stage silently rescales every flux the model infers.
"""
with fits.open(out_dir / "data.fits") as hdul:
    header = hdul[0].header

print(f"BUNIT   : {header.get('BUNIT', 'missing')}")
print(f"EXPTIME : {header.get('EXPTIME', 'missing')}")
print(f"TEXPTIME: {header.get('TEXPTIME', 'missing')}")

"""
Second, the noise-floor closure — the same check the UVIS anchor validates against its
published sigma_sky, run here as internal validation (no published IR anchor exists for this
field). In blank sky the noise recipe reduces to approximately R * sigma_sky, so the noise
map's faint floor (5th percentile) over R times the measured sky RMS should sit near 1. On a
few-dither IR mosaic expect the agreement to be looser than the UVIS case — the spatially
varying depth that the weight map records genuinely widens the noise distribution — but a
ratio far from unity would mean the noise map and the sky disagree, and that is a reduction
problem, not a statistic to explain away.
"""
noise = fits.getdata(out_dir / "noise_map.fits").astype(float)

r_factor = provenance["noise"]["correlated_noise_factor"]
sky_rms = provenance["noise"]["empirical_background_rms"]

closure = float(np.nanpercentile(noise, 5)) / (r_factor * sky_rms)

print(f"noise floor / (R * sky RMS): {closure:.3f}")

"""
__Wrap Up__

You have now seen the whole IR pipeline at stage resolution: an acquisition with no CTE
decision to make, a calibration chain that fits ramps instead of reading charge — rejecting
cosmic rays per exposure and delivering `_flt` files already in e-/s — a drizzle stage that
keeps `driz_cr` for the ramp fit's residue and prices fine grids in coverage, a noise stage
whose only IR novelty is the larger fine-grid R, and the diagnostics (WHT uniformity, R, the
finite-noise guard) that together tell you whether a few-dither IR dataset is shippable at
your chosen dials.

The following locations of the workspace are good places to checkout next:

- `scripts/hst_wfc3_ir/start_here.py`: the anchor reduction, the drizzle-choice arithmetic
  and the zero-weight-speckle finding in full.
- `scripts/hst_acs/step_by_step.py`: the shared stage-by-stage reference (alignment,
  drizzle, PSF, packaging) this script deltas against.
- `scripts/hst_wfc3_uvis/step_by_step.py`: the CCD sibling — CTE, post-flash and the live
  MAST discovery idiom that works for both channels.
- `scripts/guides/noise_maps.py`: the noise recipe and both Casertano branches, derived.

__Env__ (Developer Only)

ENV: network
"""
