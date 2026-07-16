# Simulated data through the reduction pipeline — feasibility verdict

Research note for issue #44 (PyAutoMind prompt
`simulated_lens_through_reduction_pipeline.md`, 2026-07-09, extended
2026-07-16). Question: can PyAutoReduce take an arbitrary input image — a
strong lens, or anything else — and emit the dataset that HST / JWST / Keck /
ALMA would have delivered after observing and reducing it? And should it: is
this in scope, or better served by another package (e.g. GalSim)?

**Verdict: in scope, as synthetic-source *injection* into real calibrated
frames (plus CASA `simobserve` for ALMA) — not as raw-instrument simulation,
which stays out of scope.** Rationale and survey below.

## The framing that decides everything

"Simulate what the instrument delivers" hides two different problems:

1. **Simulate the raw instrument files** (HST `_raw` frames with cosmic rays,
   CTE trails and sky; JWST up-the-ramp `uncal` integrations) and run the full
   reduction on them. Maximal fidelity, maximal cost: PyAutoReduce would need
   to *generate* everything the reduction exists to remove.
2. **Inject a synthetic scene into real calibrated frames** (the `_flc`/`_cal`
   /prepared-Keck exposures the pipeline already consumes after `acquire`) and
   run the rest of the reduction unchanged. Cosmic rays, sky, correlated
   noise, bad pixels, dither geometry and PSF wings all come for free —
   they are already in the real frames.

The second framing is the survey literature's standard, and it maps directly
onto PyAutoReduce's stage architecture. That is the whole verdict, really;
the survey below is the evidence.

## Survey: what exists (build vs combine)

- **GalSim** (Rowe et al. 2015) renders galaxy/star scenes — profiles, PSF
  convolution, pixel noise, selected detector effects. By its own scope
  statement it does **not** simulate cosmic rays, persistence, ghosts or
  satellite trails. It is a *scene renderer*, not a raw-instrument simulator;
  combining "GalSim + PyAutoReduce" still leaves the raw-frame gap open.
  For us the input image is arbitrary by requirement, so GalSim is one
  possible upstream among many (a PyAutoLens `simulator.py` output is
  another) — not a dependency.
- **Strong-lensing simulators** — lenstronomy's sim API (Birrer & Amara
  2018), paltas (Wagner-Carena et al.), deeplenstronomy (Morgan et al. 2021),
  SLSim — all emulate at the **final-product level**: PSF-convolve an ideal
  image, add noise following an instrument model, done. None of them exercise
  a reduction pipeline (drizzle correlated noise, CR rejection, per-frame
  registration, ePSF construction). That untouched gap — datasets whose
  systematics come from the *actual reduction path* — is precisely what this
  feature would add, and why no existing package simply replaces it.
- **HST imaging**: there is **no maintained raw-frame simulator** for
  ACS/WFC3 imaging (only grism/spectroscopy simulators, e.g. Wayne, aXeSim).
  Building one — CR morphology, CTE, sky, darks, dither execution — is a
  large, hard-to-validate project with no community baseline to check
  against. Strongest single argument that framing 1 is out of scope.
- **JWST**: MIRAGE generates `uncal` ramps for NIRCam/NIRISS/FGS, but is
  pre-flight code, no longer kept current with instrument models, reference
  files or the calwebb pipeline. Usable externally by a motivated user;
  not a foundation PyAutoReduce should build on. (Roman's stack — romanisim
  for raw, STIPS for post-pipeline products — confirms the industry split
  between the two framings but targets a different mission.)
- **ALMA is the exception where framing 1 is cheap**: CASA `simobserve`
  natively turns a sky-model FITS into a MeasurementSet with thermal +
  atmospheric noise for a chosen array configuration, fully supported by
  NRAO. PyAutoReduce's visibility branch (`alma.md`) already consumes
  calibrated MeasurementSets via casatools — a simulated MS is just another
  input to `extract → assemble → package`.
- **Synthetic-source injection (SSI) is the literature standard for imaging**:
  DES **Balrog** (Suchyta et al. 2016; Everett et al. 2022, Y3; Anbajagane
  et al. 2025, Y6 — 146M injections across 5000 deg²) injects model images
  into real survey frames "containing real sources, as well as the actual
  noise, sky-background, and other systematics" and re-runs the unmodified
  measurement pipeline. Rubin/LSST carries the same pattern as first-class
  pipeline machinery (`source_injection`). SSI is validated methodology, not
  a shortcut we would need to defend.

## Design sketch: the `inject` stage (framing 2)

An opt-in stage between `acquire` and the combine path:

- **Input contract**: a FITS image in surface-brightness units with a pixel
  scale (ideally finer than native) plus a sky position, or "at the target".
  No PyAuto* import — the input is a plain image, preserving the repo's
  never-imports boundary; PyAutoLens simulator outputs qualify but are not
  special.
- **Per exposure**: render the input onto the frame's native pixel grid
  through the frame's own WCS/distortion (the frame↔mosaic transform
  machinery from the per-exposure frame products already does this
  bookkeeping), convolve with that frame's PSF (tier-1 frame ePSFs exist;
  adapter model-PSF fallback otherwise), convert to native units via the
  adapter (cps vs electrons vs MJy/sr), add the *source's own* Poisson noise
  only, and write a modified copy of the calibrated frame.
- **Then run the existing pipeline unchanged** — align, drizzle (real CR
  rejection operating on real CRs), noise, psf, package. The output is the
  standard `al.Imaging.from_fits` product set, with `reduction.json`
  provenance carrying an explicit `injected:` block (never let a
  semi-synthetic dataset masquerade as real).
- **ALMA**: `simobserve` as an acquire-stage alternative for the visibility
  branch (fully synthetic MS), with uv-plane injection into a real MS (add
  the model's FT to real visibilities) as the Balrog-analogue option.

Injection into *real* frames means every simulated dataset needs a real
archival observation to host it. That is a feature, not a bug — the noise,
CRs and PSF are then real by construction — and matches how PyAutoReduce is
used (targets with actual archival coverage). A fully synthetic mode (no
host data) stays out of scope; users who need it can run MIRAGE/`simobserve`
externally and hand PyAutoReduce the result.

## What this is good for

- End-to-end validation: known truth in → modeling-ready dataset out; closes
  the loop on reduction systematics (correlated drizzle noise, ePSF error)
  that final-product-level simulators cannot probe.
- Injection-recovery tests of the pipeline itself (flux conservation through
  drizzle, noise-map calibration) — the same style of acceptance evidence the
  parity appendices use, but with ground truth.
- Training/test sets for lens searches with reduction-real systematics.

## Phasing (each phase = its own PyAutoMind prompt when its predecessor ships)

1. **HST/ACS imaging injection** — **in progress (issue #46)**: the opt-in
   `inject` stage between `acquire` and the combine path. Dials:
   `TargetSpec.inject_image` (plain FITS, e-/s per pixel, not
   PSF-convolved), `inject_pixel_scale`, `inject_position` (default: the
   target), `inject_psf` (default: per-frame tier-1 ePSF), `inject_seed`.
   Real-data validation: injection-recovery on the slacs0008 field
   (`prototypes/inject_recovery_slacs.py` — clean vs injected difference
   image, 3" aperture, parity-style report).
2. **JWST + Keck injection** — split on sizing:
   - **2a JWST — in progress (issue #52)**: `_cal` frames, MJy/sr. The
     input contract is per-adapter (`InstrumentAdapter.inject_units`):
     HST e-/s per pixel; **JWST Jy per pixel**, converted through the
     frame's own `PIXAR_SR` so the injected mean is flux-exact and
     gain-free; the nominal `e_per_dn` sizes only the Poisson width
     (disclosed in provenance). Injected variance enters frame ERR
     before image3 resamples (the JWST noise stage reads propagated
     ERR). Recovery spike: `prototypes/inject_recovery_jwst.py`
     (COSMOS-Web ring, F150W).
   - **2b Keck — prompted** (`draft/feature/pyautoreduce/inject_stage_keck.md`):
     blocked on a registration design decision — raw-header WCS is
     arcsecond-grade, so placement must ride the combine's measured
     offsets (`offsets_to_reference` pre-pass), not `all_world2pix`.
3. **ALMA** — `simobserve` acquire-alternative + optional uv-plane injection.
4. *(deferred, likely never)* raw-frame simulation — revisit only if a
   validated community simulator for HST imaging appears.

## Open questions (for phase-1 planning)

- Input-image units/WCS contract: require a WCS, or pixel-scale + position?
- PSF matching: injected image is convolved with the frame ePSF — does the
  input contract forbid pre-convolved inputs, or detect via header keyword?
- Poisson noise of the injected source: per-frame realisation (correct) —
  seeded how, for reproducibility across re-runs?
- Where injection meets `frame_products`: injected per-frame products should
  carry the injection flag in the manifest too.
- Saturation/nonlinearity: bright injected sources should saturate as real
  ones would — clip at adapter `saturation_dn`, or document as out of scope?

## References

GalSim: Rowe et al. 2015 (A&C 10, 121). Balrog: Suchyta et al. 2016 (MNRAS
457, 786); Everett et al. 2022 (ApJS 258, 15); Anbajagane et al. 2025
(arXiv:2501.05683). lenstronomy: Birrer & Amara 2018. deeplenstronomy:
Morgan et al. 2021 (JOSS, arXiv:2102.02830). paltas: Wagner-Carena et al.
2022 (ascl:2210.029). MIRAGE: STScI JDox "MIRAGE JWST Data Simulator"
(pre-flight caveat). STIPS: STScI 2024 (PASP 136, 124502). CASA
simulations: casadocs "Simulations" (simobserve/sm).
