# HST/ACS reduction pipeline — design

Phase-1 design for reducing HST ACS/WFC imaging of strong lenses into
modeling-ready datasets. The governing principle, applied at every stage:
**run the instrument's default pipeline; deviate only where a lensing
calculation requires it, and document every deviation here.**

The quality bar is the SLACS reduction ([Bolton et al. 2008, SLACS
V](https://arxiv.org/abs/0805.1931)): ACS/WFC F814W exposures drizzled to a
0.05″/pix mosaic — the reductions behind the existing modeling datasets
(e.g. the 54-lens subhalo-scan sample). The weak-lensing-grade reference for
ACS reduction craft is the COSMOS pipeline ([Koekemoer et al.
2007](https://arxiv.org/abs/astro-ph/0703095)).

## Output contract

Per target, the pipeline emits the `al.Imaging.from_fits` product set:

| File | Shape (default) | Content |
|------|-----------------|---------|
| `data.fits` | 281×281 (≈14″ at 0.05″/pix) | drizzled science cutout, WCS intact |
| `noise_map.fits` | matches `data.fits` | per-pixel RMS (background + Poisson), correlated-noise corrected |
| `psf.fits` | 21×21, odd | compact PSF kernel for convolution |
| `psf_full.fits` | 61×61, odd | extended PSF (wings) for point-source work |
| `reduction.json` | — | provenance: program/visit/exposure IDs, filter, zero-point, exposure time, pixel scale, drizzle parameters, PSF method + diagnostics, software versions |

The existing SLACS modeling cutouts have **stripped FITS headers** — no WCS,
units or exposure metadata survive. `reduction.json` + intact cutout headers
restore that provenance permanently.

## Stage 0 — target specification

A reduction is declared, not scripted: a small per-target YAML (or a table row
for samples) giving target name, RA/Dec, instrument/filter, proposal ID(s) if
known, cutout size, and any per-target overrides. The pipeline is a pure
function of this spec plus the archive — re-running it reproduces the dataset
bit-for-bit (modulo upstream calibration reference-file updates, which
`reduction.json` records).

## Stage 1 — acquire (astroquery.mast)

**Default behaviour.** Query MAST by coordinates (`astroquery.mast.Observations`),
filter to the requested instrument/filter, and download the calibrated,
CTE-corrected exposures (`_flc.fits`) plus association tables (`_asn.fits`).
No re-calibration with `calacs` — MAST's on-the-fly reprocessing keeps
exposures current with the best reference files.

**Disk-space strategy** (a first-class requirement, not an optimization):

- **Per-target transient cache.** SLACS-like targets are single pointings —
  ~4 exposures ≈ 0.7 GB, needed only while the target reduces. The cache
  directory has a configurable size cap; exposures are evicted
  (oldest-completed-target first) once their products are written. A manifest
  (`cache_manifest.json`) records what was downloaded from where, so eviction
  never loses reproducibility — a re-run re-fetches deterministically.
- **Survey-mosaic targets.** For lenses inside large programs (COSMOS etc.),
  don't download tiles: use the MAST **HAPCut** cutout service
  (`astroquery.mast.Hapcut`) to fetch spatial cutouts of Hubble Advanced
  Products directly. This path skips stages 2–3 (the HAP mosaic is already
  drizzled) and enters at stage 4 with the HAP weight maps — a documented
  trade-off, since HAP drizzle parameters are not ours.
- Whole-sample runs stream: acquire → reduce → package → evict, one target at
  a time, so peak disk usage is one target's exposures regardless of sample
  size.
- **Reference files are acquisition too** (spike finding): AstroDrizzle's IVM
  weighting resolves ACS calibration files through ``jref$``, so the acquire
  stage syncs CRDS best references for the downloaded exposures
  (``crds.bestrefs --sync-references=1 --update-bestrefs``) into the cache and
  exports ``CRDS_PATH``/``jref``. Reference files are shared across targets
  and are the one cache component *not* evicted per target.

**Query hygiene** (spike finding): a plain coordinate query also matches HAP
skycell products, whose member lists re-reference the same exposures many
times over (31 products deduping to 7 files for slacs0008-0004) and pull in
neighbouring-pointing exposures. The acquire stage filters to direct
calibration-level-2 observations and groups by proposal/visit before
downloading; HAP products are used only on the explicit HAPCut path.

## Stage 2 — align

**Default behaviour.** Trust the a-priori WCS solutions MAST now attaches to
exposures (aligned to Gaia where possible). Within a single visit, relative
alignment is normally already adequate for drizzling.

**Deviation trigger, not default.** Run `tweakreg` relative alignment only
when the drizzle's cosmic-ray flagging or the final mosaic shows residual
misregistration (elongated stars, CR-flagged galaxy cores). The pipeline
computes an alignment diagnostic (cross-correlation of single-drizzled
exposures) and records it; sub-0.1-pixel residuals skip TweakReg. Absolute
astrometry only needs to be good enough for the WCS in the cutout header —
lens modeling works in relative arcseconds.

## Stage 3 — drizzle (drizzlepac.AstroDrizzle)

**Default behaviour.** AstroDrizzle with the STScI-recommended flow: sky
matching (`skymethod='globalmin+match'`), cosmic-ray rejection (`driz_cr`,
median-combine baseline), final drizzle of all exposures onto one grid.

**Lensing deviations (each justified):**

| Parameter | Value | Why we deviate |
|-----------|-------|----------------|
| `final_scale` | 0.05″/pix | SLACS convention; matches every existing modeling dataset and the source-plane resolution regime lens models are calibrated in |
| `final_rot` | 0 (north-up) | uniform orientation across samples simplifies masks, position angles, and cross-dataset comparison |
| `final_wht_type` | `IVM` | inverse-variance weights are what stage 4's noise map needs |
| `final_pixfrac` | **user-facing dial**, default 0.8 | the literature genuinely disagrees — SLACS V used no drizzle at all (bilinear rectification, see below), SLACS IX MultiDrizzled with an unstated pixfrac, Bayer et al. used pixfrac 1.0, SLACS's WFPC2 additions 0.6 — so pixfrac (with `final_kernel`) is first-class configuration, not a buried default. Smaller pixfrac *reduces* correlated noise (R at s=1: 1.50 @ p=1.0, 1.364 @ p=0.8, 1.25 @ p=0.6) at the cost of coverage uniformity; the pipeline reports the STScI weight-uniformity diagnostic (WHT RMS/median over the cutout, ≲0.2 required; spike measured 0.066 at p=0.8 for 7 exposures) and the resulting R with every reduction, so the choice is auditable per dataset |
| `final_kernel` | user-facing with `final_pixfrac`, default `square` | SLACS IX used `gaussian`; parity study decides whether matching it matters for the noise/PSF products |
| `final_units` | `cps` (e-/s) | counts/s + `EXPTIME` in provenance keeps the Poisson term computable while matching how the existing datasets are modeled — confirmed against SLACS parity |

Undrizzled artifacts (`_single_sci`, masks) stay in the transient cache; only
the mosaic + weight map proceed.

**The SLACS-V caveat (literature finding):** [Bolton et al.
2008](https://arxiv.org/abs/0805.1931) did **not** drizzle the F814W snapshot
data — "the 'drizzle' re-sampling algorithm … is not well suited to
single-exposure Snapshot data". Their ACSPROC recipe rectifies frames onto the
0.05″ grid by **bilinear interpolation**, masks cosmic rays with **L.A.
Cosmic** (van Dokkum 2001), stacks dithered sets with a further CR-rejection
step, and rectifies a **Tiny Tim** PSF with identical sampling (a clean
precedent for our drizzled-PSF invariant). Consequences: (1) the
single-exposure path needs a non-drizzle branch (rectify + L.A. Cosmic, or
single-image drizzle with `driz_cr` off — decide in phase 1); (2) legacy
parity interpretation depends on which recipe produced a given legacy dataset
(multi-exposure targets like slacs0008-0004 are SLACS-IX-style MultiDrizzle;
true one-exposure snapshots are ACSPROC bilinear, where the noise correlation
structure differs from any drizzle R).

**Open question for the parity study:** whether the existing SLACS
`data.fits` are in e-/s or e-; the study measures this from the data/noise
relation rather than assuming.

## Stage 4 — noise map

The single most lensing-specific stage. Drizzle does not emit an RMS map;
the strong-lensing literature does ([Bayer et al.,
arXiv:1803.05952](https://arxiv.org/abs/1803.05952), §3.1, on exactly this
SLACS-style data):

```
sigma_i = sqrt( N_i / W_i + sigma_sky^2 )        # per pixel i
```

- **Background term** `sigma_sky = 1/sqrt(W_i)` from the IVM weight map —
  read noise, dark current and sky in one term, per STScI weight-map
  semantics.
- **Poisson term** `N_i / W_i` with `N_i` the source photo-electrons in pixel
  i (counts/s × effective exposure time from the weight/exposure maps).
  Negative-fluctuation pixels floor the Poisson term at zero rather than
  producing NaNs — and the pipeline **crashes loudly** on NaN/zero weights
  inside the cutout rather than silently patching them.
- **Correlated-noise correction.** Drizzling correlates neighbouring pixels;
  a naive per-pixel RMS underestimates the noise a lens-model chi^2 sees. Apply
  the DrizzlePac handbook / Casertano et al. (2000) scalar correction R
  (function of pixfrac p and scale ratio s) to the RMS map, and record R in
  `reduction.json`. Whether SLACS applied R is a parity-study question — if
  the legacy noise maps are uncorrected, we ship the *corrected* map and
  document the difference (deviate-when-lensing-requires cuts both ways).

**Validation.** Empirical noise in blank-sky patches of the mosaic must match
the noise-map prediction; the data/noise relation in bright unsaturated pixels
must show the Poisson slope.

## Stage 5 — PSF

Tiered strategy, informed by weak-lensing and AGN/quasar practice. The
non-negotiable invariant, whatever the tier: **the delivered PSF is the
drizzled PSF** — same kernel, pixfrac, scale and orientation as the science
mosaic. Never pair a native-frame PSF with a drizzled image.

- **Tier 1 — empirical ePSF from field stars** (default when the drizzled
  field contains enough usable stars): Anderson & King-style effective-PSF
  construction via `photutils.EPSFBuilder` on star cutouts from the final
  mosaic. Star selection: unsaturated, uncrowded, DQ-clean point sources;
  the count and fit residuals go into `reduction.json`.
- **Tier 2 — TinyTim + focus model** (fallback; SLACS elliptical snapshot
  fields are typically star-poor): model PSFs raytraced per exposure with
  TinyTim, focus ("breathing") estimated by matching whatever stars exist,
  per the COSMOS weak-lensing method ([Rhodes et al.
  2007](https://arxiv.org/abs/astro-ph/0701480) — focus recoverable to
  <1 μm rms from few stars). Per-exposure model PSFs are then **drizzled
  through the same AstroDrizzle footprint** as the science frames, which
  handles the drizzled-PSF invariant exactly. TinyTim's maintenance status is
  a risk; STScI's newer focus-diverse ePSF libraries for ACS are evaluated in
  the spike as a possible replacement.
- **Tier 3 — high-fidelity reconstruction back-ends (optional extra):**
  [STARRED](https://arxiv.org/abs/2402.08725) (wavelet-regularized
  two-channel PSF reconstruction; current state of the art in the TDCOSMO
  pipeline comparisons) and [PSFr](https://github.com/sibirrer/psfr)
  (Birrer's iterative reconstruction developed for lensed quasars). Not
  needed for SLACS-style galaxy-galaxy lenses; becomes the default tier for
  lensed quasars/AGN where the point source itself constrains the PSF.

- **Alternative construction — `TargetSpec.psf_from_frames` (issue #21):**
  the delivered mosaic-grid PSF built by *combining the per-frame tier-1
  ePSFs* instead of measuring stars on the resampled mosaic: each frame's
  native ePSF is convolved with the drizzle drop (`final_pixfrac` box, exact
  fractional-width Fourier convolution), resampled onto the mosaic grid via
  the local frame→mosaic WCS Jacobian at the target position, and
  exposure-time-weighted averaged (`psf/frame_combine.py`). Honours the
  drizzled-PSF invariant by construction — it is the drizzle geometry
  applied to the PSF — while sidestepping mosaic resampling artifacts and
  star scarcity (every frame's full star field contributes). Approximation
  recorded in diagnostics: local-affine geometry + drop convolution;
  sub-pixel output-sampling phases not modelled. Loud when no frame yields
  a tier-1 ePSF — never silently falls back to the mosaic-star ePSF.

Products: `psf.fits` 21×21 for fit convolution; `psf_full.fits` 61×61
capturing wings (diffraction spikes matter for quasar work later). Both odd,
centered, unit-normalized; construction method + diagnostics recorded.

## Stage 6 — package

Cut out `data.fits` and `noise_map.fits` around the target (default 281×281;
configurable), **preserving the cutout WCS and units in the headers** —
deviation from the legacy datasets' stripped headers, deliberately. Write the
PSFs and `reduction.json`. Optionally emit the auxiliary modeling-prep files
(`info.json` skeleton) but leave scientific annotations (positions, extra
galaxies) to the modeling workflow — reduction ends at the dataset.

## Validation — SLACS parity study

End-to-end on 2–3 SLACS lenses (e.g. `slacs0008-0004` plus one well-behaved
and one awkward system):

1. Reduce from MAST with this pipeline.
2. Compare against `/mnt/c/Users/Jammy/Science/subhalo/dataset/slacs/<lens>/`:
   data ratio map (photometric parity + units answer), noise ratio map
   (weight semantics + correlated-noise answer), PSF FWHM/ellipticity
   comparison.
3. Fit both reductions with the standard PyAutoLens SLaM-style model; the
   inferred lens parameters (Einstein radius, slope proxies) must agree
   within statistical errors. This is the acceptance test that matters —
   *the reduction is correct when the science is invariant under it.*
4. Discrepancies are documented in this file's parity appendix (to be added),
   never silently absorbed.

## Parity appendix — spike results (slacs0008-0004, 2026-07-08)

First end-to-end run (`prototypes/slacs_f814w_spike.py`): 7 FLC exposures
(program 10886, EXPTIME 3132s total) → CRDS sync → AstroDrizzle (0.05″/pix,
north-up, IVM, pixfrac 0.8, square kernel, cps) → noise map → 281×281 cutout,
compared against the legacy modeling dataset:

| Quantity | Result | Reading |
|----------|--------|---------|
| registration | integer offset (0, 2) px | legacy cutout centre differs by 0.1″; integer roll only — sub-pixel registration needed for tighter photometry |
| data ratio (bright px, n=1223) | median **0.934** | **units confirmed e-/s** (a legacy-in-electrons mismatch would read ~3132); ~7% global flux deficit to chase (exposure-set and sky-subtraction differences are the suspects — the spike stacked neighbouring-pointing exposures the legacy reduction almost certainly excluded) |
| noise ratio (R *not* applied) | median **0.678** [0.597, 0.714] | our uncorrected noise is ~32% below legacy |
| noise ratio × R (R = 1.364) | **0.924** ≈ data ratio | **legacy noise maps are consistent with the correlated-noise correction being applied** — after applying R, data and noise carry the same ~7% global scale offset, i.e. the noise *recipe* matches |

Conclusions adopted into the design: `final_units='cps'` stands; stage 4
**applies** the Casertano/DrizzlePac factor R as designed. The production
pipeline (proposal-filtered exposures, sub-pixel registration) lands at data
ratio 0.941 / noise ratio 0.925 — the ~6% global flux scale vs the legacy
dataset is **accepted as a documented difference** (decision 2026-07-08):
the legacy reduction's exact provenance (kernel, photometric era,
FLT-vs-FLC calibration) is unrecoverable, both ratios carry the same scale so
the *relative* products are self-consistent, and lens-model inferences are
scale-invariant in the relevant regime. A PyAutoMind research prompt tracks
chasing it (gaussian-kernel re-drizzle + calibration-era check) if it ever
matters.
Also confirmed: tier-1 ePSF is plausible for this field (236 point-like
>10σ detections mosaic-wide, pre-selection), and CRDS reference-file sync +
HAP-skycell query filtering belong to the acquire stage (see above).

**Addendum 2026-07-09 (post HAP-dedupe + usability screen)** — the numbers
above were measured on mosaics that drizzled every exposure **twice** (the
HAP visit-level duplicate-ingestion bug found by the frame-products
validation; a failed 0-second exposure was also being ingested). Doubled IVM
weights suppress the computed noise by √2, which propagates through every
noise row above. Re-measured on the corrected 3-exposure mosaic:
data ratio **0.959** (the flux deficit shrinks to ~4%), noise ratio
**1.309** (previous 0.925 × √2). The corrected map is internally consistent
— blank-sky map/empirical-RMS = 1.45 vs applied R = 1.36 — so stage 4's
recipe stands. The flipped implication is about the *legacy* maps: with the
√2 artifact removed, legacy noise sits ≈ our **uncorrected** IVM noise,
i.e. the legacy maps do **not** appear to carry the correlated-noise
correction after all, and our chi²-faithful maps are ~30% above legacy by
design. Formal re-baseline of the acceptance comparison is tracked in
PyAutoMind (`research/pyautoreduce/acceptance_noise_rebaseline.md`).

## Per-exposure frame products (opt-in packaging mode)

`TargetSpec.frame_products: bool = False` — when on, the pipeline additionally
packages every calibrated `_flc`/`_flt` SCI chip as a modeling-ready
native-pixel product set, for fitting N undrizzled exposures simultaneously
instead of one mosaic (roadmap "Per-exposure frame products"). A packaging
mode over `autoreduce/package/frames.py`, run between the package stage and
the provenance write (after driz_cr DQ flags and any TweakReg WCS refinement
exist, before eviction can delete the cached frames); the mosaic path is
untouched when the flag is off. HST and JWST are supported (the JWST
branch — `_crf` inputs, native MJy/sr, DO_NOT_USE-only masking, manifest
schema v2 — is specified in `jwst.md` §"Per-exposure frame products —
feasibility"); the flag on any other instrument fails fast.

Layout (uniform `_chip<EXTVER>`; the physical `CCDCHIP` is in the manifest):

```
output_root/<name>/frames/
  manifest.json
  <rootname>_chip1/{data.fits, noise_map.fits, dq.fits, cr_mask.fits}
  <rootname>_chip2/{...}
```

Design decisions:

- **Cutout geometry** — the native-pixel shape is derived from the existing
  dials (`cutout_shape * final_scale / native_scale`, odd-forced): same sky
  footprint as the mosaic cutout, no new user dial. Chip coverage is
  re-tested per chip (the acquire footprint filter is per-exposure; ACS
  chip 2 frequently misses the target and is skipped, recorded).
- **Units** — SCI/ERR are converted to e-/s (`ELECTRONS / EXPTIME`; WFC3/IR
  is already e-/s) so every frame and the mosaic share the cps flux scale.
  Each SCI chip's `MDRIZSKY` is subtracted first: `globalmin+match` sky is
  only *virtually* subtracted during drizzle, so the frames would otherwise
  carry the sky pedestal the mosaic lacks.
- **Noise** — the calacs/calwf3-propagated ERR extension (native-pixel
  Poisson + read noise), unit-converted with SCI. No correlated-noise `R`:
  nothing has been resampled.
- **Cosmic rays (deepCR — documented deviation from STScI defaults)** —
  `driz_cr` rejects CRs against a median stack, so its DQ flags exist only
  when several exposures overlap; per-frame modeling needs a mask for every
  frame on its own, single-exposure visits included. deepCR (Zhang & Bloom
  2020) detects CRs on individual exposures; ACS/WFC uses the published
  `ACS-WFC` model and WFC3/UVIS the `WFC3-UVIS` label-free retrain (Chen et
  al. 2024), both shipped in deepCR >= 0.3 (the manifest records the exact
  model + threshold, so datasets remain re-maskable). WFC3/IR skips deepCR
  — calwf3 up-the-ramp fitting already flags IR CRs in DQ.
  Mask-only by contract: deepCR inpainting is never used — bad pixels are
  masked, never fabricated. Optional dependency: `autoreduce[frames]`.
- **Masking policy** — any nonzero DQ bit, deepCR CR pixel, off-chip or
  non-finite/non-positive-ERR pixel is masked-by-noise
  (`noise = MASKED_NOISE_VALUE`, `data = 0`), so each frame's
  `data.fits` + `noise_map.fits` load directly as an imaging dataset and
  masked pixels drop out of any chi^2. The mosaic's isolated-bad-pixel
  policy deliberately does **not** apply — its structured-defect rejection
  would refuse exactly the CR trails this mode exists to capture. Raw
  `dq.fits` (int32) and `cr_mask.fits` (uint8) keep the full bit
  information for consumers wanting a different policy.
- **WCS / registration** — each cutout header carries the frame's SIP WCS
  (`to_header(relax=True)`); the NPOL/D2IM lookup-table distortion is not
  FITS-serializable (~0.1 px residual), so the manifest records
  `target_pixel` — the target projected through the *full* distortion model
  — as the exact per-frame registration anchor. Frame-to-frame mapping is
  `WCS_j^-1 ∘ WCS_i`; no bespoke transform format.
- **Registration accuracy (issue #19)** — every frame's manifest entry
  carries a `registration` block: the astrometric solution behind its WCS
  (`wcsname`/`wcstype` + `RMS_RA`/`RMS_DEC`/`NMATCHES`, which state the
  group's *absolute* catalog alignment — for slacs0008, `FIT_REL_GSC242` at
  ~44 mas), and the *measured relative* residual against the reference frame
  (resample through both shipped WCS, phase-correlate; whitened correlation
  is the CR-hole-robust estimator). Measured on slacs0008: relative
  registration ≲ 0.1 native px, with the measurement itself limited to
  ~0.1–0.3 px where CR-masked pixels bite the source (three estimators —
  whitened, plain correlation, masked centroid — disagree at exactly that
  level). **Modeling stance:** the shifts ship as *information, not
  policy* — the default is treating them as known (residuals sit below the
  scales standard modeling constrains); precision applications free
  per-frame `(dy, dx)` nuisance parameters with Gaussian priors of the
  recorded residual width, which also absorbs the SIP-serialization term.
  `max_registration_residual_px` in `reduction.json` gives the at-a-glance
  verdict per dataset.
- **Caveats recorded in the manifest** — single-exposure reductions carry no
  driz_cr flags (`driz_cr_run: false` + note: the deepCR mask is then the
  only CR rejection); re-runs clear `frames/` first so a smaller exposure
  set leaves no orphan chip directories.

- **Per-frame PSF (tier 1: native ePSF; issue #21)** — each chip dir ships
  `psf.fits` / `psf_full.fits` built from the frame's own full chip on
  native (undrizzled, distorted) pixels: DQ-flagged pixels are NaN-screened
  (detection masks them; stamp extraction rejects windows they touch — in
  multi-exposure visits the driz_cr flags kill cosmic rays masquerading as
  stars; single-exposure visits have only the shape cuts, recorded as
  `cr_screen` in the manifest), the target-exclusion cut uses the
  full-distortion projection, and the saturation cap is formed in the
  frame's native units. **Insufficient stars is a recorded outcome, not a
  hard stop** — a deliberate deviation from the mosaic path's tier-2
  escalation: a single ~500 s frame legitimately may lack the minimum
  usable stars, and its data products remain useful; the manifest `psf`
  block and a loud runtime notice say the frame is not modelable until the
  tier-2 model PSF (TinyTim / focus-diverse grid) lands — that tier stays
  on the roadmap.

## Non-goals (phase 1)

WFC3, other filters, JWST, and Euclid (owned by
`euclid_strong_lens_modeling_pipeline`) — see `roadmap.md`. No GUI, no
database; per-target YAML + filesystem outputs only. (Per-exposure
`_flt`/`_flc` products, a phase-1 non-goal, shipped later as the opt-in
packaging mode above.)
