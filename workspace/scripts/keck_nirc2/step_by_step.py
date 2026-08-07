"""
Step By Step: Keck NIRC2 AO
===========================

`start_here.py` reduced the SHARP B1938+666 K' dataset with one call to `reduce_target`.
This script opens that call up: every ground-based stage — calibration, sky subtraction,
registration, the single-pass combine, the detector noise budget — demonstrated standalone
on the cached raw frames, with the literature that motivates each step.

Run `start_here.py` first: this script works off the frames it cached under
`cache/b1938+666/` (science, calibrations, PSF stars), so it re-downloads nothing.

The ground stages exist because KOA hands you *raw detector reads*. An HST reduction starts
from `_flc` files that STScI has already bias-corrected, dark-subtracted, flat-fielded and
CR-flagged; a Keck reduction starts from voltages. Everything a space-telescope archive does
for you, this script does in front of you.

__Contents__

- **Imports:** Import the required Python libraries.
- **Paths:** Anchor every path to the workspace root.
- **Raw Frames And KOA:** What a KOA level-0 NIRC2 frame is, and what its header carries.
- **Calibration Matching:** Darks matched to ITIME/COADDS, flats within +/-14 days, and the SHARP flat+sky-only honesty.
- **Building Master Calibrations:** `build_calibrations` — master flat, optional master dark, bad-pixel mask.
- **Calibrating A Frame:** `calibrate_frame` — DN to total electrons, flat-fielded, bad pixels as NaN.
- **Why The Sky Dominates:** The K' sky at Mauna Kea is orders of magnitude brighter per pixel than a lensed arc.
- **Grouping By Time:** `group_by_time_gaps` splits the frame sequence into contiguous runs before any sky is estimated.
- **Running Sky Subtraction:** The scaled running sky, and why it takes two passes.
- **Registration:** `phase_offset` — measured sub-pixel offsets, because header pointing is arcsecond-grade.
- **One Pixmap, One Resampling:** Distortion + registration + rescale as a single drizzle mapping.
- **The Detector Noise Budget:** Gain, MCDS read noise, dark current — and the per-frame budget checked empirically.
- **Casertano R And The Closure:** The correlated-noise factor and the mosaic-level closure statistic.
- **Why Not KAI:** The community pipeline is Python 2.7 + IRAF — why the stages are implemented natively.
- **Wrap Up:** Summary and where to go next.

__Imports__
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from autoreduce import instruments
from autoreduce.calibrate import build_calibrations, calibrate_frame, load_calibration_sets
from autoreduce.sky import group_by_time_gaps, running_sky_subtract
from autoreduce.align.registration import phase_offset
from autoreduce.noise.rms import casertano_r, empirical_background_rms

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the
folder containing `scripts/`). We point at the cache `start_here.py` populated.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"
OUTPUT_ROOT = WORKSPACE / "output"

cache_dir = CACHE_ROOT / "b1938+666"
science_paths = sorted(
    p for p in (cache_dir / "science").rglob("*.fits*") if "download" not in p.name
)
calib_paths = sorted(
    p for p in (cache_dir / "cals").rglob("*.fits*") if "download" not in p.name
)

if not science_paths:
    raise FileNotFoundError(
        f"No cached science frames under {cache_dir / 'science'} — run "
        f"scripts/keck_nirc2/start_here.py first; this script teaches off its cache."
    )

print(f"Cached science frames: {len(science_paths)}; calibration frames: {len(calib_paths)}.")

"""
__Raw Frames And KOA__

Every frame here came from the Keck Observatory Archive (https://koa.ipac.caltech.edu/) via
PyKOA (https://github.com/KeckObservatoryArchive/PyKOA), queried through its TAP/ADQL
service. KOA serves NIRC2 as *raw level-0* frames — the level-1 quick-look products are not
science grade — so the header is the reduction's ground truth. The keywords that drive
everything downstream:

- `ITIME` — integration time per coadd (seconds).
- `COADDS` — number of coadds averaged on chip; the stored pixel value is the *per-coadd
  average* in DN, so total electrons = DN x gain x COADDS.
- `MJD-OBS` — observation time; temporal order defines the running sky and PSF epochs, and
  the date routes the distortion solution (Yelda et al. 2010 before 2015-04-13, Service et
  al. 2016 after).
- `SAMPMODE` / `MULTISAM` — the readout mode; MCDS/Fowler sampling cuts the effective read
  noise (the noise-budget section below).
- `KOAIMTYP` — frame type (`object`, `dark`, `flatlamp`, `flatlampoff`, `domeflat`), which
  is how calibration frames are sorted.
"""
facts = []
for path in science_paths:
    header = fits.getheader(path)
    facts.append(
        {
            "path": path,
            "mjd": float(header["MJD-OBS"]),
            "itime": float(header["ITIME"]),
            "coadds": int(header["COADDS"]),
            "sampmode": int(header.get("SAMPMODE", 2)),
            "multisam": int(header.get("MULTISAM", 1)),
        }
    )
facts.sort(key=lambda f: f["mjd"])

science_itime = facts[0]["itime"]
science_coadds = facts[0]["coadds"]
print(
    f"Science setup: ITIME {science_itime}s x COADDS {science_coadds}, "
    f"SAMPMODE {facts[0]['sampmode']} / MULTISAM {facts[0]['multisam']} (MCDS)."
)

"""
__Calibration Matching__

Ground-based NIR calibration has its own matching rules, and the acquire stage applied them
when it filled the cache:

- **Darks must match ITIME and COADDS.** Dark current and bias structure scale with the
  integration pattern, so a 180 s x 1 science frame needs 180 s x 1 darks — nothing else
  subtracts cleanly. Darks are *optional by design*: the running sky is estimated from
  frames that carry the same dark signal, so sky subtraction removes dark and sky together.
  That is the SHARP recipe — flat + sky only — and the pipeline records a darkless
  calibration honestly (`dark_subtraction: false` in provenance) rather than failing or
  staying silent.

- **Flats are mandatory, within +/-14 days.** NIR flat fields are stable over weeks, and
  SHARP's own science nights routinely carry only darks — so when the night has no flats,
  the nearest flat-bearing night inside a 14-day window is used and recorded. K-band dome
  flats carry a thermal pedestal (the warm dome glows at 2 um), so the master flat is
  *lamp-on minus lamp-off* when off-frames exist: the difference isolates the lamp
  illumination and removes the thermal print-through.

- **Bad pixels come from the calibrations themselves**: hot pixels stand out in the dark,
  dead pixels are unresponsive in the flat. They are *propagated, not interpolated* — NaN in
  the calibrated frame becomes zero weight in the combine, exactly how HST handles
  CR-flagged pixels.
"""
darks, flat_on, flat_off = load_calibration_sets(
    calib_paths,  # every cached calibration frame; sorted by KOAIMTYP internally.
    science_itime=science_itime,  # only darks matching the science ITIME are kept.
    science_coadds=science_coadds,  # ... and the science COADDS.
)
print(f"Calibration sets: {len(darks)} matched darks, {len(flat_on)} lamp-on flats, {len(flat_off)} lamp-off flats.")

"""
__Building Master Calibrations__

`build_calibrations` median-stacks each set into the masters and derives the bad-pixel mask.
Its provenance dict is what `reduce_target` records under `calibrate` in `reduction.json`.
"""
calib = build_calibrations(
    dark_frames=darks,  # median-stacked into the master dark (kept in DN; scaled by gain at use).
    flat_on_frames=flat_on,  # median lamp-on stack.
    flat_off_frames=flat_off,  # median lamp-off stack, subtracted to remove the thermal pedestal.
    hot_sigma=5.0,  # pixels > 5 sigma above the dark median are flagged hot.
    dead_flat_threshold=0.5,  # pixels below 50% response in the unit-median flat are flagged dead.
)
print(f"Master calibrations built: {calib.provenance}")

"""
__Calibrating A Frame__

`calibrate_frame` converts one raw frame to *total electrons*: DN x gain x COADDS, dark
subtracted (when a master dark exists), divided by the unit-median flat, with bad pixels set
to NaN so no downstream stage can use them by accident. Working in total electrons is what
keeps Poisson statistics computable downstream — the noise stage needs real counts.

The gain is an adapter-owned detector constant (4.0 e-/DN), validated by the noise closure
rather than trusted blindly.
"""
adapter = instruments.get("nirc2_narrow")
detector = adapter.ground_detector()

n_demo = min(12, len(facts))
raw_demo = [fits.getdata(f["path"]).astype(np.float64) for f in facts[:n_demo]]

calibrated = [
    calibrate_frame(
        raw,  # the raw per-coadd-average DN array.
        calib,  # the master calibrations built above.
        gain_e_per_dn=detector.gain_e_per_dn,  # 4.0 e-/DN — adapter-owned, closure-validated.
        coadds=science_coadds,  # total e- needs the coadd count back.
    )
    for raw in raw_demo
]

print(
    f"Calibrated {n_demo} frames: median level {np.nanmedian(calibrated[0]):.0f} e- "
    f"(sky pedestal), {int(np.isnan(calibrated[0]).sum())} bad pixels as NaN."
)

"""
__Why The Sky Dominates__

That median level *is the problem this stage exists for*. The K' sky at Mauna Kea runs
~13-13.5 mag/arcsec^2 — orders of magnitude brighter than a lensed arc in every pixel. Below
~2 um the sky is OH airglow, varying on minutes timescales; beyond ~2.2 um thermal emission
takes over. Either way it changes faster than any calibration you could take before or
after, so it must be estimated *from the science sequence itself*: the temporally adjacent,
dithered frames of the same field, with sources masked (the standard craft; see Vaduvescu &
McCall 2004, PASP, astro-ph/0404337, for a methods treatment). Dithering is what makes this
possible — the target lands on different pixels each frame, so a running median over
neighbours sees mostly sky — and it simultaneously averages down bad pixels and flat errors.

__Grouping By Time__

Window adjacency is positional, so the frame set must be *temporally contiguous* before any
running sky is estimated — a window spanning a gap would silently borrow sky from a
different night or visit. `group_by_time_gaps` splits the MJD-sorted sequence at gaps: the
pipeline uses 3600 s for science sequences and 600 s for the interleaved PSF-star visits
(the same 600 s gap that defines PSF epochs — see `psf.py`).
"""
mjds = [f["mjd"] for f in facts[:n_demo]]
groups = group_by_time_gaps(mjds, gap_s=3600.0)
print(f"{n_demo} frames -> {len(groups)} contiguous group(s); sizes {[len(g) for g in groups]}.")

"""
__Running Sky Subtraction__

`running_sky_subtract` implements the *scaled* running sky, in two passes:

1. For each frame, take the `window` temporally nearest other frames (a frame never
   contributes to its own sky), mask their sources, and normalise each to unit median. The
   median of that stack is the sky *structure* — fringes, illumination gradients — free of
   the overall level.

2. Multiply the structure by the frame's *own* masked median — the sky *level*. This scaling
   is the difference between a running sky and a *scaled* running sky: the K' level drifts
   on minutes timescales, so borrowing the level from neighbours biases the frames at the
   edges of the sequence. Structure from neighbours, level from yourself.

The two passes refine the object mask: once the sky pedestal is gone, fainter source wings
emerge in the first-pass residuals, so the mask is rebuilt and the sky re-estimated. The
per-frame sky levels are returned in the provenance — the noise stage's background variance
is built from exactly these numbers.
"""
group = groups[0]
group_frames = [calibrated[i] for i in group]

print(f"Running scaled sky subtraction over {len(group_frames)} frames (window 9, two passes)...")

subtracted, sky_prov = running_sky_subtract(
    group_frames,  # calibrated frames in electrons, NaN bad pixels, temporal order.
    window=9,  # sky structure from the 9 nearest neighbours (the TargetSpec sky_window dial).
    n_sigma=3.0,  # object-mask threshold above the sky, in MAD-sigmas.
)

print(f"Recipe: {sky_prov['recipe']}")
print(f"Per-frame sky levels (e-): {[f'{s:.0f}' for s in sky_prov['sky_levels_e']]}")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
scale = np.nanpercentile(group_frames[0], 99)
axes[0].imshow(np.arcsinh(group_frames[0] / scale), origin="lower", cmap="magma")
axes[0].set_title("calibrated (sky pedestal in)")
axes[1].imshow(np.arcsinh(subtracted[0] / np.nanstd(subtracted[0])), origin="lower", cmap="magma")
axes[1].set_title("sky-subtracted")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plot_dir = OUTPUT_ROOT / "b1938+666" / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)
sky_plot = plot_dir / "step_by_step_sky.png"
plt.savefig(sky_plot, dpi=150)
plt.close()
print(f"Saved sky-subtraction comparison to: {sky_plot.resolve()}")

"""
__Registration__

NIRC2 header pointing is *arcsecond-grade* — at 9.942 mas/pixel, an arcsecond is a hundred
native pixels. The headers tell you roughly where the telescope pointed; they cannot align
frames. Relative offsets are therefore *measured* from the data by phase cross-correlation
(`phase_offset`, pure numpy): the cross-power spectrum of two frames, amplitude-normalised
(whitened) so the correlation peak is sharp and contrast-independent, with a parabolic fit
refining the peak to sub-pixel precision. Bad pixels (NaN) are zero-filled for the transform
only — a fraction of dead pixels does not move the peak.

The measured offsets are the astrometric truth of a Keck reduction, and the pipeline records
them in provenance (`registration_offsets_native_pix`); the injection machinery
(`simulator.py`) reuses the same arithmetic rather than ever trusting the header WCS.
"""
dy, dx = phase_offset(
    subtracted[0],  # the reference frame.
    subtracted[1],  # the frame to align; returns the (dy, dx) that shifts it onto the reference.
)
print(f"Measured dither offset frame 1 -> 0: ({dy:.2f}, {dx:.2f}) native pixels (~{np.hypot(dy, dx) * 9.942:.0f} mas).")

"""
__One Pixmap, One Resampling__

The combine backend (`nirc2_native`) now has three geometric transforms to apply to each
frame: the distortion correction (the Yelda/Service lookup tables), the measured
registration offset, and the native-to-final rescale (9.942 mas -> 10 mas). The crucial
design decision is that they enter the `drizzle` resampler as **one pixel mapping** — each
input pixel is mapped straight to its output location through all three at once, and the
data are resampled exactly once.

Why it matters: every resampling correlates the noise between neighbouring pixels and blurs
the PSF a little. Dewarp-then-shift-then-rescale as three separate interpolations would pay
that price three times, and the correlated-noise bookkeeping would be intractable. One
pixmap means the standard Casertano et al. 2000 (AJ 120, 2747) analysis applies verbatim —
exactly how drizzlepac treats ACS distortion inside AstroDrizzle, with the same resampling
engine (the standalone `drizzle` package is the engine inside drizzlepac and the jwst
pipeline). Distortion correction is where correlated noise enters a NIRC2 reduction; doing
it once, jointly, keeps it accountable.

Each frame's weight in the combine is its *inverse background variance* (sky + dark + read
noise, in cps^2) — so the accumulated weight map is exactly the IVM the shared noise recipe
`R x sqrt(sci/exptime + 1/wht)` expects, and the noise stage needs no Keck-specific branch
at all.

__The Detector Noise Budget__

The weights and the noise map both rest on the per-frame background variance, built from
three adapter-owned detector constants — gain 4.0 e-/DN, CDS read noise 38 e-, dark current
0.1 e-/s — and the frame's own header facts:

    var_bkg [e-^2] = sky_e  +  dark_e_per_s x ITIME x COADDS  +  RN_eff^2 x COADDS

The read noise is *sampling-mode aware*: in MCDS/Fowler-M mode (SAMPMODE 3) the detector
averages MULTISAM read pairs, cutting the read-noise variance by ~1/MULTISAM, so
RN_eff = 38 / sqrt(MULTISAM). For the SHARP B1938 K' frames (MCDS-32, 180 s) the budget
predicts 62.0 e- per frame where plain CDS would predict 72 — and the empirical frame
scatter measures 62-64 e-. That agreement is not luck; it is the check that keeps the
constants honest.
"""
sky_e = float(sky_prov["sky_levels_e"][0])
itime, coadds = science_itime, science_coadds
sampmode, multisam = facts[0]["sampmode"], facts[0]["multisam"]

rn_eff = detector.read_noise_e(sampmode, multisam)  # 38 / sqrt(MULTISAM) for MCDS.
dark_e = detector.dark_e_per_s * itime * coadds
budget_e = np.sqrt(sky_e + dark_e + rn_eff**2 * coadds)

empirical_e = empirical_background_rms(subtracted[0])

print(f"Effective read noise (SAMPMODE {sampmode}, MULTISAM {multisam}): {rn_eff:.1f} e- (CDS would be {detector.read_noise_e_cds:.0f}).")
print(f"Per-frame budget: sqrt({sky_e:.0f} sky + {dark_e:.0f} dark + {rn_eff**2 * coadds:.0f} read) = {budget_e:.1f} e-")
print(f"Empirical per-frame background RMS: {empirical_e:.1f} e-  (reference run: budget 62.0 vs empirical 62-64)")

"""
__Casertano R And The Closure__

Drizzling correlates the noise of neighbouring output pixels, so a per-pixel RMS map that
ignored the correlation would make every chi^2 in your lens fit wrong. The scalar correction
R of Casertano et al. 2000 (AJ 120, 2747; also Fruchter & Hook 2002) depends only on the
drop size (`pixfrac` p) and the output/native scale ratio s. This reduction resamples
9.942 mas -> 10 mas with p = 1.0 — nearly the shift-and-add limit, where R -> 1.5.
"""
scale_ratio = 0.010 / adapter.native_scale  # output / native pixel scale.
r_factor = casertano_r(
    1.0,  # final_pixfrac — the drizzle drop size used by the combine.
    scale_ratio,  # s = final_scale / native_scale ~ 1.006 here.
)
print(f"Casertano R at pixfrac 1.0, s = {scale_ratio:.3f}: {r_factor:.3f} (shift-and-add limit is 1.5).")

"""
The mosaic-level check ties everything in this script together. The shipped noise map is the
decorrelated-equivalent RMS (x R, chi^2-correct); the measurable pixel-to-pixel scatter of
the mosaic is correlation-suppressed by ~1/R. So the honest closure statistic is

    empirical mosaic RMS x R^2 / noise-map background floor  ~  1

It lands at ~0.84 on B1938. Every constant above feeds it: a wrong gain, a dropped COADDS
factor or a cps slip would miss by x6-x40, which is exactly why the constants are described
as closure-validated rather than trusted. `start_here.py` computes it on the shipped
products; the recipe and its cross-instrument siblings live in `scripts/guides/noise_maps.py`.

__Why Not KAI__

The community-standard NIRC2/OSIRIS pipeline is KAI (Lu et al. 2022,
https://doi.org/10.5281/zenodo.6677744; https://keck-datareductionpipelines.github.io/KAI/),
which implements the same conceptual chain — darks, flats, sky, distortion, stacking. It is
also Python 2.7 + IRAF/PyRAF, which cannot be a dependency of a modern stack. Every ground
operation above is a simple array operation, so **PyAutoReduce** implements the stages
natively (numpy/astropy, with the standalone `drizzle` package as the resampler) and
validates them against SHARP's published numbers and the internal closures instead of
against a pipeline that can no longer run.

__Wrap Up__

You walked the full ground-based chain by hand: raw KOA level-0 frames, matched
calibrations, master flat/dark and bad pixels, DN -> electrons, the scaled running sky with
its two passes, measured sub-pixel registration, the one-pixmap combine that pays the
resampling price exactly once, and the closure-validated detector noise budget.

The following locations of the workspace are good places to checkout next:

- `scripts/keck_nirc2/start_here.py`: the same chain as one `reduce_target` call, end to end.
- `scripts/keck_nirc2/psf.py`: the AO PSF problem — epochs, vetting gates, the provisional contract.
- `scripts/keck_nirc2/simulator.py`: source injection into the prepared frames, using the measured offsets.
- `scripts/guides/noise_maps.py`: noise recipes and closure statistics across all instruments.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness. This script
reads the cache populated by `start_here.py` and needs the heavy `[keck]` extras, so the
smoke runner skips it.

ENV: network
"""
