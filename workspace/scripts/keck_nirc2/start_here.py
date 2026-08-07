"""
Start Here: Keck NIRC2 AO
=========================

Adaptive optics from the ground can beat Hubble. The sharpest published images of the strong
lens B1938+666 — the system in which Vegetti et al. 2012 (Nature 481, 341) detected a
(1.9 +/- 0.1) x 10^8 solar-mass dark satellite at z = 0.881 — were taken not from space but
from Mauna Kea, with the NIRC2 camera behind the Keck II laser-guide-star adaptive optics
system (the SHARP programme; Lagattuta et al. 2012, MNRAS 424, 2800).

This script runs the **PyAutoReduce** Keck NIRC2 pipeline end to end on exactly that dataset:
it discovers the SHARP B1938+666 K'-band frames in the Keck Observatory Archive, pins the
frame set, reduces it from raw detector counts to a modeling-ready dataset, and closes by
loading the products into **PyAutoLens**.

Expect the first run to take tens of minutes: it downloads the raw science frames, the
night's calibrations and the PSF-star frames from KOA, plus the geometric-distortion lookup
tables, then runs the full ground-based reduction. Re-runs use the warm cache and are much
faster.

__Contents__

- **Why Adaptive Optics:** The atmosphere limits ground-based imaging to ~0.5-1.0" — AO restores the 45-50 mas diffraction limit.
- **Imports:** Import the required Python libraries.
- **Paths:** Anchor every path to the workspace root, because the pipeline requires absolute paths.
- **KOA Discovery:** Query the Keck Observatory Archive for the SHARP B1938+666 NIRC2 frames.
- **Pinning The Frame Set:** KOA has no association tables, so the spec pins the exact KOA identifiers.
- **Target Spec:** Build the frozen `TargetSpec` that declares the whole reduction.
- **The Ground Stages:** What `reduce_target` does to raw NIRC2 frames: calibrate, sky, register, combine.
- **Distortion Epochs:** The NIRC2 distortion solution is epoch-matched: Yelda et al. 2010 before 2015-04-13, Service et al. 2016 after.
- **Run The Reduction:** One function call: `reduce_target`.
- **Provenance:** Walk the returned `reduction.json` record: weights, offsets, sky levels, PSF candidates.
- **Noise Closure:** Check the noise map against the blank-sky scatter of the mosaic itself.
- **PSF Candidates:** Every PSF-star epoch ships as a candidate, and the PSF is provisional by contract.
- **Plots:** Visualize the data, noise map and PSF.
- **Loading In PyAutoLens:** Load the products with `al.Imaging.from_fits`.
- **The Plate-Scale Caveat:** The narrow-camera plate scale in the adapter is under revision — correct it at load time.
- **Open Items:** What this pipeline honestly does not do yet.
- **Wrap Up:** Summary and where to go next.

__Why Adaptive Optics__

A ground-based telescope does not deliver its diffraction limit: atmospheric turbulence
smears every image to the seeing, typically 0.5-1.0" — worse than a 10 cm telescope's
theoretical resolution, no matter how large the mirror. Adaptive optics measures the
atmospheric wavefront in real time and cancels it with a deformable mirror, restoring the
10 m Keck aperture's diffraction limit of ~45-50 mas at K' (2.1 um) — about twice as sharp
as HST in the near-infrared, because resolution scales with aperture.

For lens fields there is rarely a natural star bright enough to measure the wavefront, so
Keck II uses a sodium laser guide star (Wizinowich et al. 2006, PASP 118, 297). The laser
cannot sense image motion, so LGS AO still needs a natural *tip-tilt star* (R ~ 18-19 or
brighter) within about a minute of arc of the target — a hard requirement that decides which
lenses can be observed at all. B1938+666 has a usable tip-tilt star (R ~ 15; Lagattuta et
al. 2012), which is part of why it became the SHARP flagship.

The delivered correction is partial: on typical lens fields the Strehl ratio is ~10-30% and
the PSF FWHM ~60-90 mas (van Dam et al. 2006, PASP 118, 310, report 30-40% at K on bright,
on-axis stars). NIRC2's *narrow* camera samples this PSF at 9.942 mas/pixel — comfortably
Nyquist-sampling the ~50 mas diffraction core, which is exactly why lens programmes use it
(https://www2.keck.hawaii.edu/inst/nirc2/). The scientific payoff is resolution-limited
science: the Vegetti et al. 2012 substructure detection, SHARP's flux-ratio-anomaly studies
(Hsueh et al. 2016, MNRAS-L 463, L51; Hsueh et al. 2017, MNRAS 469, 3713) and AO-based
time-delay cosmography (Chen et al. 2019, MNRAS 490, 1743, arXiv:1907.02533) all rest on
Keck AO imaging reduced with care.

__Imports__

**PyAutoReduce** exposes exactly two names — a frozen `TargetSpec` and the `reduce_target`
function. The KOA query helpers used for discovery live in `autoreduce.acquire.koa`.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoreduce import TargetSpec, reduce_target
from autoreduce.acquire import koa
from autoreduce import instruments

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the
folder containing `scripts/`). **PyAutoReduce** requires absolute paths: its combine step
changes the working directory internally, so relative paths would break.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"      # downloaded KOA frames + distortion tables (re-used across runs)
OUTPUT_ROOT = WORKSPACE / "output"    # reduced datasets, one folder per target

"""
__KOA Discovery__

All Keck data flows through the Keck Observatory Archive (KOA, https://koa.ipac.caltech.edu/),
and **PyAutoReduce** talks to it through PyKOA
(https://github.com/KeckObservatoryArchive/PyKOA), the archive's own Python client — the
KOA analogue of `astroquery.mast`. KOA serves *raw level-0 frames only* for NIRC2: unlike
HST or JWST there are no archive-calibrated products, so every calibration step below is the
pipeline's own job.

The SHARP B1938+666 K' data were taken UT 2010 June 29-30 with the narrow camera
(14,760 s total; Lagattuta et al. 2012). We cone-query KOA around the lens position for
narrow-camera K' object frames. The query is cheap metadata — the frame downloads happen
later, inside `reduce_target`, against the cache.
"""
RA, DEC = 294.60496, 66.81450  # B1938+666 (SHARP I)
FILTER = "Kp"

adapter = instruments.get("nirc2_narrow")

work_dir = OUTPUT_ROOT / "b1938+666" / "koa_queries"
work_dir.mkdir(parents=True, exist_ok=True)

print("Querying KOA for SHARP B1938+666 NIRC2 narrow-camera K' frames (metadata only)...")

science_table = koa.query_science_frames(
    RA,  # target RA (degrees) for the cone query.
    DEC,  # target Dec (degrees).
    adapter,  # the nirc2_narrow adapter selects the camera in the ADQL query.
    FILTER,  # K' — KOA's filter string for the NIRC2 K-prime filter.
    work_dir,  # where the raw query VOTables are written.
    proposal_ids=None,  # a cone query is enough here; a program ID would narrow it further.
    koa_ids=None,  # no pinned ids yet — this IS the discovery pass that finds them.
)

print(f"KOA returned {len(science_table)} object frames at the B1938+666 pointing.")

"""
__Pinning The Frame Set__

KOA has **no association tables**: nothing in the archive says "these 39 frames are one
science stack". An HST reduction can hand MAST an association and trust it; a Keck reduction
must *pin the exact frame set itself*, which is what the `koa_science_ids` and
`koa_psf_star_ids` dials on `TargetSpec` are for. Pinning makes the reduction reproducible —
the same spec always reduces the same frames.

Two wrinkles in the SHARP data, both worth knowing about for your own targets:

1. The cone result contains *two pointing clusters*, and both carry the tip-tilt star's name
   in `targname` (a SHARP convention): the lens dithers themselves, and interleaved visits to
   the PSF/tip-tilt star ~20" away — SHARP's dedicated PSF-star strategy (Lagattuta et al.
   2012). We split them by angular separation from the lens.

2. The pipeline refuses mixed ITIME/COADDS within one science stack (a single calibration
   set must match every frame), so we pin the *modal* setup — for SHARP B1938 that is the
   180 s x 1 science frames, which drops the short acquisition frames. For the PSF star we
   prefer the *shortest* integrations: a K ~ 13-14 star saturates the narrow camera in
   180 s, and the short frames exist precisely to keep its core unsaturated.
"""
from collections import Counter

separation_arcsec = (
    np.hypot(
        (np.asarray(science_table["ra"], float) - RA) * np.cos(np.radians(DEC)),
        np.asarray(science_table["dec"], float) - DEC,
    )
    * 3600.0
)
lens_table = science_table[separation_arcsec < 12.0]
star_table = science_table[separation_arcsec >= 12.0]

setups = Counter((float(r["itime"]), int(r["coadds"])) for r in lens_table)
(modal_itime, modal_coadds), _ = setups.most_common(1)[0]
print(f"Science setups found {dict(setups)}; pinning the modal {modal_itime}s x {modal_coadds}.")

science_ids = tuple(
    str(r["koaid"])
    for r in lens_table
    if float(r["itime"]) == modal_itime and int(r["coadds"]) == modal_coadds
)

star_itimes = np.asarray(star_table["itime"], float)
short = star_itimes <= 60.0
star_rows = star_table[short] if short.any() else star_table
star_ids = tuple(str(k) for k in star_rows["koaid"])[:12]

print(f"Pinned {len(science_ids)} science frames and {len(star_ids)} PSF-star frames.")

"""
__Target Spec__

A **PyAutoReduce** reduction is *declared, not scripted*: everything about it lives in one
frozen `TargetSpec`, and the pipeline is a pure function of the spec plus the archive. The
non-default dials below are the Keck-specific ones.
"""
spec = TargetSpec(
    name="b1938+666",  # output folder name under OUTPUT_ROOT.
    ra=RA,  # target RA (degrees) — the mosaic WCS reference and cutout centre.
    dec=DEC,  # target Dec (degrees).
    instrument="nirc2_narrow",  # the 9.942 mas/pixel camera; nirc2_wide is registered but fails loudly at combine.
    filter_name=FILTER,  # K' — selects the flats and labels the products.
    final_scale=0.010,  # output pixel scale in arcsec: 10 mas, the SHARP convention (Chen et al. 2019).
    final_pixfrac=1.0,  # drizzle drop size; 1.0 (shift-and-add) is robust for the near-native resampling here.
    cutout_shape=(281, 281),  # 2.81" on a side at 10 mas — generous around the ~0.45" Einstein radius.
    koa_science_ids=science_ids,  # the pinned raw science frames (KOA has no association tables).
    koa_psf_star_ids=star_ids or None,  # the pinned PSF-star frames — the tier-A PSF strategy (see psf.py).
    sky_window=9,  # running-sky window: sky structure from the 9 temporally nearest frames.
)

"""
__The Ground Stages__

Space-based archives hand you calibrated exposures; KOA hands you raw detector reads. The
pipeline therefore runs two stages that HST/JWST reductions never see, before anything is
combined. There is no maintained community pipeline to wrap — KAI, the standard NIRC2 DRP
(Lu et al. 2022, https://doi.org/10.5281/zenodo.6677744), is Python 2.7 + IRAF/PyRAF — so
**PyAutoReduce** implements the ground stages natively in numpy/astropy:

- **Calibrate** — each raw frame (stored as the per-coadd average in DN) is converted to
  total electrons (gain 4.0 e-/DN x COADDS), dark-subtracted when the night has matched
  darks, flat-fielded (lamp-on minus lamp-off dome flats, unit median), and bad pixels (hot
  in the dark, dead in the flat) are carried as NaN into zero combine weight. The SHARP
  recipe is flat + sky only — darkless nights are recorded, never silently ignored.

- **Sky** — the defining ground-based NIR step. At K' the sky at Mauna Kea runs
  ~13-13.5 mag/arcsec^2, orders of magnitude brighter per pixel than a lensed arc, and it
  varies on minutes timescales — faster than any calibration. The pipeline estimates a
  *scaled running sky* from the temporally adjacent, object-masked, dithered frames of the
  field itself (see `step_by_step.py` for the mechanics, and Vaduvescu & McCall 2004, PASP,
  astro-ph/0404337, for the method background).

- **Register + Combine** — NIRC2 header pointing is only arcsecond-grade, so relative frame
  offsets are *measured* by phase cross-correlation, then distortion correction,
  registration and the native-to-10-mas rescale enter the `drizzle` resampler as **one
  pixel mapping** — the data are resampled exactly once. Per-frame weights are the inverse
  background variance, so the accumulated weight map is the IVM the shared noise recipe
  expects.

- **Noise + PSF** — the same noise recipe as HST/JWST, `R x sqrt(sci/exptime + 1/wht)`,
  with the Casertano et al. 2000 (AJ 120, 2747) correlated-noise factor R carried over
  unchanged because the resampling engine is identical; and the tier-A PSF-star treatment,
  described below.

__Distortion Epochs__

NIRC2's geometric distortion is corrected with published lookup tables
(https://www2.keck.hawaii.edu/inst/nirc2/dewarp.html), and the right table depends on *when*
your data were taken: the AO bench was realigned on 2015-04-13, so the pipeline
epoch-matches automatically — Yelda et al. 2010 (ApJ 725, 331; ~1 mas accuracy) before the
boundary, Service et al. 2016 (PASP 128, 095004; ~1.1 mas) after. The B1938+666 data are
from 2010, so this reduction uses the Yelda solution; the tables are downloaded once,
checksummed, and recorded in provenance. Frame sets spanning the boundary are rejected
loudly rather than mixed.

__Run The Reduction__

One call runs everything: acquisition (science + calibrations + PSF stars + distortion
tables), calibrate, sky, register, combine, noise, PSF, package.
"""
print(
    f"""
    Starting reduce_target for {spec.name}.

    First run: downloads {len(science_ids)} science frames, the night's calibrations and
    {len(star_ids)} PSF-star frames from KOA (~hundreds of MB), then reduces — expect
    tens of minutes. Re-runs hit the warm cache under {CACHE_ROOT} and skip the downloads.
    """
)

record = reduce_target(spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)

out_dir = OUTPUT_ROOT / spec.name
print(f"Reduction complete. Products in: {out_dir}")

"""
__Provenance__

`reduce_target` returns the full `reduction.json` record — every decision the pipeline made,
so the reduction can be audited (and re-derived) later. The highlights for a Keck run:
"""
print(f"Frames combined:        {record['drizzle']['n_exposures']}")
print(f"Total exposure time:    {record['drizzle']['total_exptime']:.0f} s")
print(f"Distortion epoch:       {record['acquire']['distortion_epoch']}  (Yelda et al. 2010 expected)")
print(f"Casertano R:            {record['noise']['correlated_noise_factor']:.3f}")
print(f"Weight uniformity:      {record['drizzle']['weight_uniformity_cutout']}")
print(f"Calibration:            {record['calibrate']}")
print(f"Per-frame sky levels (e-), first 5: {record['sky']['sky_levels_e'][:5]}")

"""
The measured registration offsets are provenance too — they are the astrometric truth of
this reduction (the headers are not), and the frame-products and injection machinery reuse
them.
"""
offsets = record["drizzle"]["registration_offsets_native_pix"]
print(f"Registration offsets (native pixels), first 5 frames: {offsets[:5]}")

"""
__Noise Closure__

Is the shipped noise map *right*? The pipeline's internal check compares the blank-sky
scatter measured off the mosaic itself against the noise map's background floor. One subtlety
makes the comparison honest: the shipped map is the *decorrelated-equivalent* noise (x R, the
chi^2-correct value for model fitting), while the measurable pixel-to-pixel RMS of a drizzled
mosaic is correlation-*suppressed* by ~1/R (Casertano et al. 2000; Fruchter & Hook 2002). The
apples-to-apples statistic is therefore

    empirical RMS x R^2 / predicted floor  ~  1

This closure is what keeps the detector constants honest rather than trusted: the per-frame
budget (sky + dark + MCDS read noise) predicts 62.0 e- per frame on these MCDS-32 180 s K'
frames versus 62-64 e- measured empirically, and the mosaic-level closure lands at ~0.84 on
B1938 — order-unity, where a unit error (gain, coadds, cps) would miss by x6-x40.
"""
from astropy.io import fits

from autoreduce.noise.rms import empirical_background_rms

data = fits.getdata(out_dir / "data.fits").astype(float)
noise = fits.getdata(out_dir / "noise_map.fits").astype(float)

r_factor = float(record["noise"]["correlated_noise_factor"])
empirical = empirical_background_rms(data)
predicted_floor = float(np.nanmedian(noise[noise < np.nanpercentile(noise, 50)]))
closure = empirical * r_factor**2 / predicted_floor

print(f"Blank-sky closure (empirical x R^2 / predicted): {closure:.2f}  (~1 expected; 0.84 on the reference run)")

"""
__PSF Candidates__

The AO PSF changes with the atmosphere from visit to visit, so no single reduction-time PSF
can be trusted as final. The tier-A treatment reduces each PSF-star *epoch* through the
identical calibrate/sky/combine path as the science and ships **every epoch** as
`psf_candidate_<i>.fits`; `psf.fits` is cut from the sharpest one (a peak-fraction Strehl
proxy), and the provenance carries `psf_provisional: true` — final PSF selection belongs to
lens modeling, by Bayesian evidence over the candidates (the SHARP I practice). The full
story, including the cosmic-ray vetting gates, is in `psf.py` in this folder.
"""
psf_diag = record["psf"]
print(f"PSF method:        {psf_diag['method']}")
print(f"PSF provisional:   {psf_diag['psf_provisional']}  (always true on the Keck path)")
print(f"Candidates:        {psf_diag.get('n_candidates')}, selected epoch {psf_diag.get('selected_epoch')}")
for cand in psf_diag.get("candidates", []):
    print(
        f"  epoch {cand['epoch']}: {cand['n_frames']} frames, "
        f"FWHM {cand['fwhm_arcsec'] * 1000:.0f} mas, peak fraction {cand['peak_fraction']:.3f}"
    )

"""
__Plots__

Now look at what you produced. The data are plotted with arcsinh scaling (linear near zero,
logarithmic at the bright lens galaxy), the noise map linearly, and the PSF with arcsinh to
bring up the faint AO halo around the diffraction core.
"""
psf = fits.getdata(out_dir / "psf.fits").astype(float)

plot_dir = out_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(np.arcsinh(data / np.nanpercentile(data, 90)), origin="lower", cmap="magma")
axes[0].set_title("data.fits (arcsinh, e-/s)")

axes[1].imshow(noise, origin="lower", cmap="viridis", vmax=np.nanpercentile(noise, 99))
axes[1].set_title("noise_map.fits (RMS e-/s)")

axes[2].imshow(np.arcsinh(psf / psf.max() * 100.0), origin="lower", cmap="magma")
axes[2].set_title("psf.fits (arcsinh)")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plot_path = plot_dir / "start_here_products.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Saved product summary plot to: {plot_path.resolve()}")

"""
You should see the B1938+666 Einstein ring — a near-complete ~0.9"-diameter infrared ring
around the lens galaxy. This is the image in which a 10^8 solar-mass dark subhalo is
detectable; everything the reduction did (sky, distortion, one-pass resampling, honest
noise, candidate PSFs) exists to keep that signal trustworthy.

__Loading In PyAutoLens__

The products load directly into **PyAutoLens** — that is the whole point of the output
contract. The import is guarded so this reduction script stands alone if **PyAutoLens** is
not installed.

__The Plate-Scale Caveat__

One honest correction is applied at load time. The `nirc2_narrow` adapter carries the
9.942 mas/pixel plate scale, but the best pre-2015 narrow-camera calibration is
9.952 mas/pixel (Yelda et al. 2010 — the raw headers agree). The mosaic was resampled
assuming 9.942, so one output pixel is truly 10 mas x (9.952 / 9.942) = 10.010 mas. Feeding
the corrected scale to **PyAutoLens** makes the Einstein radius physical without touching
the shipped FITS products. The adapter value is under revision (an epoch-aware fix is a
gated source change); until it lands, apply this correction whenever absolute angular scales
matter.
"""
record_from_disk = json.loads((out_dir / "reduction.json").read_text())

pixel_scale_asbuilt = record_from_disk["package"]["pixel_scale"]  # what the reduction assumed (10 mas).
pixel_scale_true = pixel_scale_asbuilt * (9.952 / 9.942)  # Yelda et al. 2010 plate-scale correction.

print(f"As-built pixel scale: {pixel_scale_asbuilt}\"  ->  corrected: {pixel_scale_true:.6f}\"")

try:
    import autolens as al
except ImportError:
    al = None
    print(
        "PyAutoLens is not installed (pip install autolens), so the loading demonstration "
        "is skipped — the reduction itself is complete and the products are on disk."
    )

if al is not None:
    dataset = al.Imaging.from_fits(
        data_path=out_dir / "data.fits",  # the drizzled K' cutout, e-/s.
        noise_map_path=out_dir / "noise_map.fits",  # matching RMS map (already x R).
        psf_path=out_dir / "psf.fits",  # the sharpest tier-A candidate — provisional, see psf.py.
        pixel_scales=pixel_scale_true,  # the plate-scale-corrected value, NOT the packaged one.
    )
    print(f"Loaded al.Imaging with shape {dataset.data.shape_native} at {pixel_scale_true:.6f}\"/pixel.")

"""
For real modeling of this system the SHARP practice is to fit at 20 mas via 2x2 binning
(Chen et al. 2019's efficiency trick) and to select among the PSF candidates by Bayesian
evidence — see `autolens_workspace` for the modeling side.

__Open Items__

Honesty about what this path does not do yet, so you are never surprised:

- **Wide camera**: the published distortion solutions are narrow-camera only, so
  `nirc2_wide` is registered for spec completeness but *fails loudly at combine*.
- **No cosmic-ray rejection at combine**: the drizzle accumulates without an outlier pass.
  The ~39-frame science stack dilutes CRs by the weight sum and the bad-pixel policy covers
  the cutout, but a median/blot-style rejection is the principled fix and remains open.
  (Single-frame PSF epochs are protected instead by the sharpness vetting — see `psf.py`.)
- **Detector-frame orientation**: the output WCS is TAN at the target with the detector's
  orientation — there is no north-up resampling yet; rotator-angle handling awaits the
  astrometric-parity validation.
- **Plate scale**: as above — corrected at load, adapter fix pending.

__Wrap Up__

You reduced the SHARP B1938+666 Keck AO dataset from raw KOA frames to a modeling-ready
**PyAutoLens** dataset: pinned frame set, native ground-based calibration and running-sky
subtraction, epoch-matched distortion in a single resampling pass, a closure-checked noise
map and a candidate-based provisional PSF.

The following locations of the workspace are good places to checkout next:

- `scripts/keck_nirc2/step_by_step.py`: every ground stage above, demonstrated standalone on the cached frames with the literature.
- `scripts/keck_nirc2/psf.py`: the AO PSF problem end to end — candidates, vetting gates, the provisional contract.
- `scripts/keck_nirc2/simulator.py`: injecting a synthetic lensed source into the real prepared frames to test recovery.
- `scripts/guides/noise_maps.py`: the shared noise recipe, Casertano R and the closure statistics across instruments.
- `scripts/guides/output_contract.py`: the 4-file + `reduction.json` contract every instrument path emits.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness. This script
needs network access (KOA, distortion tables) and the heavy `[keck]` extras, so the smoke
runner skips it.

ENV: network
"""
