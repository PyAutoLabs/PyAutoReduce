"""
Start Here: HST WFC3/IR
=======================

This script reduces Hubble Space Telescope WFC3/IR imaging of the strong lens SDSS J0252+0039
into a modeling-ready dataset with **PyAutoReduce** — and in doing so introduces a channel
that is *fundamentally different* from every HST CCD you have reduced so far.

WFC3/UVIS and ACS/WFC differ in plate scales and reference files; WFC3/IR differs in physics.
It is an infrared HgCdTe detector, not a CCD: it has no shutter, it reads out non-
destructively while photons accumulate, its cosmic rays are rejected *inside each exposure*
by the calibration pipeline, and no charge-transfer correction exists or is needed. Almost
every CCD instinct — `_flc` files, CR rejection needing multiple exposures, electrons as the
native unit — changes here.

The reduction machinery downstream (AstroDrizzle, the noise recipe, PSF tiers, packaging) is
still the shared HST path, so as in `hst_wfc3_uvis/`, this folder teaches the *deltas* and
defers shared depth to `hst_acs/step_by_step.py`. This script also carries one of the most
instructive findings in the whole workspace: a reduction that **PyAutoReduce** *refused to
ship*, and why that refusal was the pipeline working exactly as designed.

__Contents__

- **A Different Detector:** HgCdTe, no shutter, and up-the-ramp readout.
- **IR Gotchas:** Persistence, blobs and 0.13" undersampling.
- **The Anchor:** The J0252+0039 F160W snapshot — an internal validation target.
- **Imports:** Import the required Python libraries.
- **Paths:** Anchor every path to the workspace root, absolutely.
- **The Drizzle Choice:** Half-native 0.065"/pixel and the fine-grid Casertano branch.
- **The Zero-Weight-Speckle Finding:** The reduction the finite-noise guard refused to ship.
- **Target Spec:** Declare the IR reduction, including the cutout-size reasoning.
- **The Reduction:** Run the full pipeline with one function call.
- **Provenance:** Walk the diagnostics that matter most on few-dither IR data.
- **Plots:** Visualize the data, noise map and PSF.
- **Load the Dataset in PyAutoLens:** Load the products as an `al.Imaging` dataset.
- **Wrap Up:** Summary of the script and next steps.

__A Different Detector__

The WFC3/IR channel is a HgCdTe photodiode array. Three consequences cascade from that (the
WFC3 Data Handbook, https://hst-docs.stsci.edu/wfc3dhb, is the authoritative reference):

- **No shutter, MULTIACCUM readout.** The detector is read *non-destructively* many times
  while charge accumulates — "up the ramp". Each downloaded exposure is really a fitted
  slope: signal per second, from a whole sequence of reads.
- **Cosmic rays are rejected per exposure.** A cosmic-ray hit appears as a discontinuity in
  a pixel's ramp, so the `calwf3` ramp-fitting step (CRCORR) identifies and excludes it
  *within a single exposure*. IR exposures arrive largely CR-cleaned — unlike CCDs, where CR
  rejection needs multiple overlapping exposures at the drizzle stage. (AstroDrizzle's
  `driz_cr` still runs on IR stacks, for the residue the ramp fit misses — see
  `hst_wfc3_ir/step_by_step.py`.)
- **No CTE correction, no `_flc`.** There is no charge transfer across the detector, so
  nothing to correct: the calibrated product is the plain **`_flt`**, and — because the ramp
  fit measures a rate — it is already in **electrons per second**. (CCD `_flc` frames are in
  electrons; the drizzle stage converts. On IR, the units the lens model wants are there from
  the start.)

The effective saturation full well is ~78 ke-, which feeds the PSF star selection's peak cut
just as on the CCD channels.

__IR Gotchas__

Three IR-specific realities to keep in mind when judging any WFC3/IR dataset:

- **Persistence:** a bright source observed earlier — even in a previous program — leaves a
  slowly-decaying afterglow in the pixels it saturated. A ghost arc in your field may be
  someone else's star from two orbits ago.
- **Blobs:** small regions of reduced sensitivity on the channel-select mechanism appear as
  soft dark spots; they are flagged in the DQ arrays and drilled out by the drizzle weights,
  which means *coverage holes* if the dither pattern never moves the target off them.
- **Undersampling:** the native pixels are 0.128" — badly undersampling HST's infrared PSF.
  The remedy is dithering plus drizzle: sub-pixel offsets between exposures let a finer
  output grid recover resolution (Fruchter & Hook 2002, PASP 114, 144;
  https://ui.adsabs.harvard.edu/abs/2002PASP..114..144F). This is why IR programs dither and
  why the drizzle scale choice below is the central dial of any IR reduction.

__The Anchor__

The dataset is the WFC3/IR F160W snapshot of SDSS J0252+0039 — the same lens whose UVIS
F390W reduction is the published Bayer et al. (https://arxiv.org/abs/1803.05952) anchor of
`hst_wfc3_uvis/start_here.py`. Honesty first: unlike the UVIS leg, there is no published IR
reduction of this field to close against, so the IR channel's validation is *internal* —
units, noise closure, weight uniformity and the guard behavior you will meet below. A
snapshot program also means *few dithers*, which is exactly what makes this dataset such an
instructive IR case.
"""

"""
__Imports__

Alongside the two **PyAutoReduce** entry points we import `instruments` — the adapter
registry — to read the IR channel's recommended output scale rather than hard-coding it, and
the public `casertano_r` helper to do the correlated-noise arithmetic by hand.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoreduce import TargetSpec, reduce_target
from autoreduce import instruments
from autoreduce.noise.rms import casertano_r

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the folder
containing `scripts/`). **PyAutoReduce** requires absolute paths: its drizzle step changes the
working directory internally, so relative paths would break.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"      # downloaded exposures + CRDS references (re-used across runs)
OUTPUT_ROOT = WORKSPACE / "output"    # reduced datasets, one folder per target

RA, DEC = 43.188375, 0.666222  # SDSS J0252+0039: 02h52m45.21s +00d39m58.4s

"""
__The Drizzle Choice__

Native 0.128" pixels undersample the PSF, so dithered IR programs conventionally drizzle to a
finer grid — deep-field practice sits in the 0.06-0.08" range. The IR adapter recommends
**0.065"/pixel** (half-native), and we read that recommendation off the adapter itself rather
than hard-coding it:
"""
IR_SCALE = instruments.get("wfc3_ir").recommended_final_scale

print(f"wfc3_ir native scale     : {instruments.get('wfc3_ir').native_scale}\"/pix")
print(f"wfc3_ir recommended scale: {IR_SCALE}\"/pix")

"""
Finer output pixels are not free. The Casertano et al. (2000, AJ 120, 2747) correlated-noise
factor R depends on the pixfrac p and the scale ratio s = output scale / native scale — and
when the output grid is finer than the drizzle drop (s < p), the *fine-grid branch* of the
formula applies and R grows quickly: each shrunken input drop is being spread over several
output pixels, correlating all of them. Compare the two regimes with the public helper —
UVIS at native scale versus IR at half-native:
"""
r_uvis_native = casertano_r(pixfrac=1.0, scale_ratio=1.0)          # UVIS start_here dials.
r_ir_fine = casertano_r(pixfrac=1.0, scale_ratio=IR_SCALE / 0.128)  # IR: s ~ 0.51 < p -> fine-grid branch.

print(f"R at native scale, pixfrac 1.0      : {r_uvis_native:.3f}")
print(f"R at 0.065\" from 0.128\", pixfrac 1.0: {r_ir_fine:.3f}")

"""
R is materially larger on the IR grid — every pixel's noise is inflated by that factor in the
delivered noise map, and the value is recorded per run in `reduction.json`. That is the price
of resolution, paid transparently. If your science prefers uncorrelated noise over sampling,
`final_scale=0.128` at pixfrac 1.0 gets you back to shift-and-add (R = 1.5); the dial is
yours.

__The Zero-Weight-Speckle Finding__

Now the finding this dataset is famous for (in this workspace, anyway). The **PyAutoReduce**
phase-1 default pixfrac is 0.8 — the SLACS-convention value that works beautifully on
well-dithered ACS programs. Reducing *this* snapshot at 0.065"/pixel with pixfrac 0.8 left
230 pixels inside the cutout with **zero drizzle weight**: with only a few dithers, shrunken
drops on a fine grid leave gaps between them that no exposure ever covers — coverage
speckle. Zero weight means infinite noise, and the finite-noise packaging guard *refused to
ship the dataset*.

That refusal is the design working, not failing. A zero-weight pixel has no data; a pipeline
that quietly interpolated over it would hand your lens model pixels that look like data and
are not. The loud failure forces the dial decision back to you, with the diagnostics to make
it: either widen the drops (pixfrac 1.0 closes coverage, at the cost of the larger R above)
or coarsen the grid. The rule of thumb this finding established:

    On WFC3/IR, few-dither data at sub-native output scales needs pixfrac -> 1.0
    (or a coarser final_scale).

The spec below therefore sets `final_pixfrac=1.0`. If you want to see the guard fire — a
genuinely educational crash — rerun with pixfrac 0.8; the reduction refuses to package,
reporting the count of non-finite noise pixels and telling you to fix the coverage rather
than patch the noise map. The WHT-uniformity diagnostic tells the same story numerically
(`hst_wfc3_ir/step_by_step.py` reads it in detail).

__Target Spec__

One more IR-specific choice needs explaining: the cutout shape. The default 281x281 cutout
spans 14" at the ACS 0.05" scale, but at 0.065"/pixel it would span 18.3" of sky — and on
this field that wider footprint reaches a zero-coverage detector-defect blob 8.5" from the
lens, which the packaging guard (rightly) also refused. Matching the ACS sky footprint
instead gives 14" / 0.065 -> **215 pixels**. The lesson generalizes: `cutout_shape` is in
*pixels*, so changing `final_scale` changes the sky footprint — size your cutout to the
coverage your dithers actually deliver.
"""
spec = TargetSpec(
    name="j0252+0039_f160w",  # Output folder name under output/.
    ra=RA,                    # Target right ascension in degrees (cutout centre).
    dec=DEC,                  # Target declination in degrees.
    instrument="wfc3_ir",     # IR adapter: _flt (already e-/s), no CTE, native 0.128"/pix.
    filter_name="F160W",      # The IR snapshot's filter (discovered live from MAST; see below).
    final_scale=IR_SCALE,     # 0.065"/pix — the adapter's half-native recommendation.
    final_pixfrac=1.0,        # Few-dither snapshot -> full drop, or the coverage guard refuses (see above).
    cutout_shape=(215, 215),  # 14" at 0.065"/pix — match the ACS footprint, stay inside real coverage.
)

"""
A note on the filter: this workspace pins F160W for reproducibility, but the integration
script that established this anchor did not — it asked MAST which WFC3/IR filters actually
cover the target and picked from the answer. That live-discovery idiom (query, filter to
direct observations, drop the HAP `"detection"` pseudo-filter) is demonstrated in
`hst_wfc3_uvis/step_by_step.py` and works identically for the IR channel.

__The Reduction__

One call, as always: MAST download of the `_flt` exposures, CRDS (`iref`) reference sync,
alignment, AstroDrizzle onto the fine grid, noise map with the fine-grid R, PSF from field
stars, packaging with the guards you now understand.
"""
print(
    "Running the J0252+0039 F160W IR reduction. The first run downloads the "
    "exposures from MAST and syncs CRDS references (expect tens of minutes); "
    f"re-runs re-use the cache under {CACHE_ROOT}."
)

record = reduce_target(spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)

out_dir = OUTPUT_ROOT / spec.name

print(f"Products written to: {out_dir}")

"""
__Provenance__

On few-dither IR data, two diagnostics deserve first attention:

- `drizzle.weight_uniformity` — RMS/median of the weight map, the numerical form of the
  coverage story above. Below ~0.2 is the acceptable regime; pixfrac 0.8 on this data fails
  the *finite-noise* guard outright, and even passing values on IR snapshots run higher than
  a well-dithered ACS program's.
- `noise.correlated_noise_factor` — should match the fine-grid `casertano_r` computed by
  hand above.

The IR channel has no CTE stage, so unlike UVIS there is no CTE model to audit — one whole
class of systematics simply absent.
"""
print(f"weight_uniformity : {record['drizzle']['weight_uniformity']}")
print(f"correlated_noise_R: {record['noise']['correlated_noise_factor']:.3f} (by hand: {r_ir_fine:.3f})")
print(f"n_exposures       : {record['acquire']['n_exposures']}")
print(f"psf               : {json.dumps(record['psf'], indent=2)}")

"""
__Plots__

The product triplet, arcsinh-scaled. Note the noise map's structure: on a few-dither IR
mosaic the per-pixel depth genuinely varies across the field (each pixel's weight counts the
exposures that covered it), so a spatially-structured noise map here is *information*, not an
artifact.
"""
from astropy.io import fits

data = fits.getdata(out_dir / "data.fits").astype(float)
noise = fits.getdata(out_dir / "noise_map.fits").astype(float)
psf = fits.getdata(out_dir / "psf.fits").astype(float)

sky_rms = record["noise"]["empirical_background_rms"]

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

fig.suptitle(f"{spec.name}: WFC3/IR F160W at {IR_SCALE}\"/pix, pixfrac 1.0")
fig.tight_layout()

plot_path = plot_dir / "start_here_products.png"
fig.savefig(plot_path, dpi=120)
plt.close(fig)

print(f"Plot saved to: {plot_path}")

"""
__Load the Dataset in PyAutoLens__

The IR products obey the same contract as every **PyAutoReduce** dataset — data and matching
RMS noise in e-/s, an odd unit-normalized PSF, pixel scale in the provenance — so the load
is identical. As everywhere, the import is guarded: **PyAutoReduce** never depends on
**PyAutoLens**.
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
        f"{record['package']['pixel_scale']}\"/pix — pair it with the UVIS "
        "F390W dataset for multi-wavelength lens modeling "
        "(see autolens_workspace/scripts/multi_dataset)."
    )

"""
__Wrap Up__

You have reduced data from HST's infrared channel and met everything that makes it
different: up-the-ramp readout with per-exposure CR rejection, `_flt` files already in e-/s,
no CTE stage at all, undersampled pixels that make the drizzle scale the central dial, the
fine-grid Casertano branch that prices that dial in correlated noise — and the zero-weight-
speckle guard refusal that turned a snapshot's sparse dithers into this channel's most
useful lesson: pixfrac 1.0 or a coarser scale, never a holey dataset.

The following locations of the workspace are good places to checkout next:

- `scripts/hst_wfc3_ir/step_by_step.py`: the IR calibration stages in detail — reference
  pixels, ramp fitting, why `driz_cr` still runs, and the WHT-uniformity diagnostic.
- `scripts/hst_wfc3_uvis/start_here.py`: the UVIS half of this same lens, with its published
  noise anchor.
- `scripts/hst_acs/step_by_step.py`: the shared HST machinery both WFC3 channels inherit.
- `scripts/guides/noise_maps.py`: the noise recipe and both Casertano branches in full.

__Env__ (Developer Only)

ENV: network
"""
