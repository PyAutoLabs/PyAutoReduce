# Keck NIRC2 AO — per-stage deltas vs the HST design

Phase 4. The first ground-based instrument, which forces two new seams — an
**acquire backend** (`InstrumentAdapter.archive`: `koa` | `mast`) and the
**ground pre-combine stages** (`calibrate`, `sky`) that space-based level-2
products make moot — plus a third combine backend, `nirc2_native`.

Scientific reference practice is the SHARP programme: Lagattuta et al. 2012
(SHARP I, the B1938+666 ring); Chen et al. 2016 (SHARP III, the AO-PSF
problem and its in-modelling solution); Chen et al. 2019 (the mature
pipeline statement: flat-field, sky subtraction, distortion correction,
coaddition; 10 mas narrow / 40 mas wide final scales; 2x2 binning to 20 mas
for modelling efficiency); reduction lineage Auger et al. 2011 /
Marshall et al. 2007.

**There is no maintained community pipeline to wrap** — KAI, the standard
NIRC2/OSIRIS DRP, is Python 2.7 + IRAF/PyRAF. The ground stages are
therefore implemented natively (numpy/astropy; every operation is a simple
array op) and validated against SHARP published numbers and the internal
closures, with the ``drizzle`` package (the same resampling engine inside
drizzlepac and the jwst pipeline) doing dewarp + coaddition so the Casertano
correlated-noise factor and the drizzled-PSF invariant carry over unchanged.

| Stage | Delta vs HST |
|-------|--------------|
| spec | `koa_science_ids` pins the exact raw frame set (KOA has no association tables); `koa_psf_star_ids` names the PSF-star frames; `sky_window` sets the running-sky width. Cameras are adapters: `nirc2_narrow` (9.942 mas) / `nirc2_wide` (39.686 mas) |
| acquire | **KOA via PyKOA** (`acquire/koa.py`), not MAST: raw level-0 science frames (level-1 quick-look is not science grade) **plus the night's calibrations** (darks matched to ITIME/COADDS, flats in the science filter) **plus PSF-star frames**. The distortion solution is acquisition too (the CRDS-analogue seam): epoch-matched lookup tables — Yelda et al. 2010 before the 2015-04-13 servicing, Service et al. 2016 after — synced into `references/keck` and recorded with checksums. No footprint filter: NIRC2 observations are pointed and raw-header WCS is approximate; the frame set is pinned by ids/program instead |
| calibrate | **new stage** (`calibrate/nir_frames.py`): DN -> e- (gain x coadds), optional master dark (sky frames carry the dark — the SHARP recipe is flat + sky only; darkless nights are recorded, never silent), flat (lamp-on minus lamp-off, unit median), bad pixels (hot in the dark, dead in the flat) carried as NaN into zero drizzle weight |
| sky | **new stage** (`sky/running.py`), the defining ground-based NIR step: **scaled running sky** — sky *structure* from the unit-median-normalised, object-masked window of temporally adjacent frames; sky *level* from the frame's own masked median (kills the edge-of-sequence bias a drifting K' sky puts on plain running medians); two passes with the object mask rebuilt from first-pass residuals. Per-frame sky levels feed the noise model |
| align | registration is phase cross-correlation (`align/registration.py`, numpy-only) inside the combine — header pointing is arcsecond-grade only |
| combine | `nirc2_native` (`drizzle/nirc2_combine.py`): distortion + registration + native->final rescale enter as **one drizzle pixmap** (exactly how drizzlepac treats ACS distortion), per-frame weights = inverse background variance (sky + dark + RN^2 x coadds, in cps^2), so the accumulated weight map **is** the IVM the shared noise recipe expects. Mosaic in e-/s with total EXPTIME. Final scales: 10 mas narrow / 40 mas wide (SHARP convention); `final_scale` stays the user dial (0.02 reproduces Chen 2019's 2x2 binning) |
| noise | **unchanged recipe** — `noise.rms.noise_map_from` applies verbatim (that was the point of making the weights IVM): R x sqrt(sci/exptime + 1/wht), Casertano R identical because the resampler is identical. Blank-sky closure check as for HST/JWST |
| psf | **redefined contract: the AO PSF is provisional, never final** (`psf_provisional: true` in provenance). Tier A (`psf/nirc2_star.py`, default): PSF-star epochs (MJD-gap grouping) reduced **pipeline-identically** through the same calibrate/sky/combine path — the drizzled-PSF invariant — with **every epoch shipped** as `psf_candidate_<i>.fits`; `psf.fits`/`psf_full.fits` cut from the sharpest (peak-fraction Strehl proxy), because final selection belongs to lens modelling (Bayesian evidence over candidates — SHARP I practice). Tier B: in-field ePSF (existing photutils machinery), still flagged provisional. Tier C (lensed AGN): iterative reconstruction from the quasar images (Chen et al. 2016) / STARRED / PSFr — a *modelling-stage* concern, out of reduction scope |
| package | unchanged `al.Imaging.from_fits` contract; BUNIT e-/s; headers intact |

## Detector constants (adapter-owned, closure-validated)

Gain 4.0 e-/DN, CDS read noise 38 e-, dark ~0.1 e-/s. The effective read
noise is **sampling-mode aware** (`Nirc2Detector.read_noise_e`): SAMPMODE 3
(MCDS/Fowler-M) cuts the variance ~1/MULTISAM. Validated per frame on the
SHARP B1938 K' data (MCDS-32, 180 s): budget 62.0 e- vs empirical 62-64 e-,
where the plain-CDS value would predict 72 — the blank-sky closure is what
keeps these constants honest rather than trusted.

**Closure interpretation** (shared with the parity scripts): the shipped
noise map is the decorrelated-equivalent (x R, the chi^2-correct value),
while the measurable per-pixel RMS of the mosaic is correlation-suppressed
by ~1/R — so the apples-to-apples statistic is `empirical x R^2 /
predicted`, ~1 when the budget is right (0.84 measured on B1938; unit
errors in gain/coadds/cps show as x6-x40).

## Validation anchor — B1938+666 (SHARP I)

K' narrow-camera imaging of the canonical IR Einstein ring (the Vegetti et
al. 2012 substructure-detection system). No legacy PyAuto Keck dataset
exists to diff, so the acceptance bar is the JWST-phase precedent —
**"internally consistent + parity with published SHARP measurements"**:

1. Internal closures: blank-sky RMS vs noise-map prediction; WHT uniformity
   over the cutout; bad-pixel policy.
2. PSF-candidate core FWHM in the ~65-70 mas range SHARP reports.
3. Astrometric parity of ring features against HST imaging of the system.
4. End-to-end PyAutoLens fit reproduces the published Einstein radius within
   statistical errors — science invariance, as for SLACS.

Driver: `prototypes/b1938_keck_spike.py` (KOA query + reduction + checks 1-2);
checks 3-4 complete after the first reduced dataset lands.

## Per-exposure frame products — feasibility (issue #31, 2026-07-10)

Whether the frame-products chain (HST #16/#19/#21, JWST #27/#29) extends to
NIRC2. **Verdict: GO in principle — the strongest science case of the three
observatories — but GATED on the acceptance checks (#13) completing**: the
stack-level pipeline should be accepted (θ_E parity, plate-scale fix)
before a per-frame mode builds on it.

**Implemented (issue #33, 2026-07-10, user-directed go-ahead)** via
`package/keck_frames.py` — with the plate-scale caveat riding every
product (`native_scale_note` in the manifest) until the acceptance task's
epoch-aware fix lands. Corrections/additions from implementation: the
measured offsets were *already* serialized
(`registration_offsets_native_pix` — the note below originally claimed
otherwise); the new provenance additions are the mapping constants
(`origin`, `scale_ratio`, `sci_path`) that make the frame↔mosaic transform
fully reconstructable. The frame-vs-stack outlier pass ships as per-frame
`outlier_mask.fits` (positive >5σ residuals against the robustly-rescaled
resampled mosaic — the mask-generation half of the stack-level CR open
item; the second-pass recombine remains open). Per-frame PSFs are
native-pixel stamps of the temporally nearest **accepted** tier-A star
frame (MJD-matched via `group_epochs`), `psf_provisional` as always;
products convert to e-/s. `psf_from_frames` stays HST/JWST-only — the AO
mosaic PSF is the tier-A epoch design, not a stamp combination.

### Should — the AO case is the strongest one

- **The AO PSF varies frame to frame** (seeing, correction quality) — this
  is THE dominant systematic in AO lens modelling, to which SHARP III
  (Chen et al. 2016) devotes itself. Co-adding marries mismatched PSFs
  under one kernel; per-frame modelling pairs each frame with its
  temporally-nearest PSF epoch, and evidence-based PSF selection (SHARP I
  practice, already the tier-A design here) extends naturally from
  "pick one epoch for the stack" to "match epochs per frame".
- Frame-selection/lucky-imaging heritage: ground-based NIR practice
  already treats frames as individuals worth weighing, not just stacking.
- The B1938+666 SHARP validation dataset and in-flight acceptance fits
  (#13) provide the natural first target.
- Tempering: per-frame SNR is low (single ~60 s AO frames of a faint
  ring); fitting epochs/subsets rather than all ~39 frames individually is
  the realistic granularity. And the stack already dilutes CRs that a
  single frame carries unflagged (see deltas).

### Can — the seam is different but real

Unlike HST/JWST there are no archive-calibrated per-frame files — but the
pipeline's own ground stages already produce **prepared frames on disk in
the work dir** (calibrated to e-, running-sky-subtracted, bad pixels as
NaN) before `nirc2_native` combines them. The packaging seam consumes
those, not `_flc`/`_crf` analogues. Deltas:

1. **Registration inverts.** NIRC2 header WCS is arcsecond-grade; the
   measured `offsets_to_reference` (phase cross-correlation inside the
   combine) ARE the registration truth. The manifest ships offsets (+ the
   sub-pixel accuracy of the correlation) instead of per-frame WCS +
   residuals; today the offsets are computed but **not serialized to
   provenance** — recording them is the first delta, useful to the stack
   provenance regardless of frame products.
2. **Native frames are distorted** — products would live in raw detector
   pixels, with the epoch-matched distortion solution (Yelda/Service
   lookup tables, already synced + checksummed in `references/keck`)
   shipped by reference in the manifest. Across a ~10" science cutout the
   differential distortion is small but must be quantified, not assumed.
3. **Per-frame noise is constructed, not read** — no ERR extensions; the
   same detector model the combine weights use (sky + dark + RN²·coadds
   per frame, MCDS-aware) yields a per-frame noise map: flat background
   variance + source Poisson. Consistent with the ground philosophy —
   every term is already computed per frame for the IVM weights.
4. **Per-frame cosmic rays are the real gap** — no DQ, no ramp fitting,
   no deepCR model for NIRC2. The principled fix is the frame-vs-stack
   outlier pass (resample the combined mosaic back to each frame,
   flag deviants) — which is *also* the standing stack-level open item
   ("Cosmic-ray rejection at combine"); one implementation serves both.
5. **Per-frame PSFs at epoch granularity** — the tier-A candidates are
   already per-epoch, reduced pipeline-identically; per-frame products
   MJD-match each science frame to its nearest PSF epoch and record the
   match + time gap. Finer-than-epoch PSF knowledge does not exist in the
   data; recorded caveat, evidence-based selection stays with modelling.
6. **Packaging hook** — a `frame_products` Keck branch reads the prepared
   frames from the work dir after combine (offsets + weights exist by
   then), cuts target-centred stamps by offset arithmetic (no WCS), and
   writes the same data/noise(+mask) product family with a manifest whose
   registration block is offset-based (schema v2's `source` and
   `sky_subtracted` fields already generalise; sky levels per frame come
   from the running-sky stage records).

### Recommendation

GO, sequenced: (1) finish the acceptance checks (#13) and the plate-scale
fix; (2) implement the frame-vs-stack outlier pass (closes the stack-level
CR open item and unlocks per-frame CR masks); (3) then the Keck frames
branch per the deltas above, validated on B1938. Do not start (3) before
(1)-(2) — the mode would inherit an unaccepted foundation and unflagged
CRs.

## Open items

- **Wide camera**: the published distortion solutions are narrow-camera
  only; `nirc2_native` fails loudly on `nirc2_wide`. A wide solution (or a
  documented identity-with-uncertainty fallback) is its own prompt.
- **Subarrays**: full frames only; the distortion tables are full-frame.
- **Absolute orientation**: the output WCS is TAN at the target with
  detector-frame orientation; rotator-angle (ROTPOSN/INSTANGL) handling and
  north-up resampling await the astrometric-parity numbers.
- **Cosmic-ray rejection at combine**: drizzle accumulates without outlier
  rejection; the 39-frame science stack dilutes CRs by the weight sum and
  the bad-pixel/masked-by-noise policy covers the cutout, but a driz_cr-
  style median/blot pass (or min-combine second pass) is the principled
  fix. Single-frame PSF epochs are protected by the coherence + sharpness
  vetting instead (a CR "PSF" measures below the diffraction floor).
- **B-spline residual background** (Auger-method final step): deferred; the
  10" narrow field is flat at the level the blank-sky closure tests.
- **KOA proprietary data**: anonymous public access only; PI login is a
  PyKOA feature the acquire seam can adopt when needed.
- **Orchestrator dispatch**: pipeline.py currently branches on
  `adapter.archive` / `adapter.observatory` at five points; folding these
  into adapter-declared capabilities (the way `combine_backend` already
  dispatches) is the refactor that keeps a third ground-based instrument
  from touching the orchestrator.
- **Prepared-frame header schema**: the ITIME/COADDS/SAMPMODE/MULTISAM/
  SKYLEV/DISTX/DISTY contract between `_prepare_keck_frames` and
  `nirc2_combine` lives as matching keyword literals; a shared typed
  header schema would break at the write site instead of the read site.
- **Header GAIN cross-check**: detector gain is the adapter constant
  (closure-validated at 4.0); reading the frame's own GAIN keyword as a
  cross-check with provenance would catch a changed electronics setup.
- **FWHM estimators**: tier-A (`nirc2_star`, equivalent-area) and tier-1/B
  (`epsf`, radial-profile) use different definitions; unify before
  cross-tier FWHM comparisons are load-bearing.
