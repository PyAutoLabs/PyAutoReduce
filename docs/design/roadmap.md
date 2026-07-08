# Roadmap — beyond HST/ACS phase 1

Skeleton only: each section becomes its own design pass (and its own
PyAutoMind prompt) when its predecessor nears shipping. Nothing here is
committed design; it exists so phase-1 code doesn't paint us into a corner.

## Instrument-adapter abstraction (the phase-1 obligation)

The one thing phase 1 must get right for everything below: pipeline stages
(`acquire`/`align`/`drizzle`/`noise`/`psf`/`package`) speak to instruments
only through an adapter in `autoreduce.instruments` that owns detector
geometry, calibrated-product naming (`_flc` vs `_flt` vs `_cal`), units and
gain, recommended combine parameters, and the PSF-model source. ACS/WFC is
adapter #1; nothing outside `instruments/` may mention a detector by name.

## HST/WFC3 (IR + UVIS)

- Same stage graph; new adapter. UVIS is ACS-like (CTE-corrected `_flc`,
  drizzle); IR differs (no CTE correction, up-the-ramp sampling, different
  DQ semantics, 0.13″ native pixels → different `final_scale` choice).
- The [ajshajib/hst-lens](https://github.com/ajshajib/hst-lens) notebooks
  (download / IR / UVIS) serve as a step checklist to audit against — not as
  architecture. Their gaps (no ACS, notebook-form, unclear noise treatment)
  are what this project exists to fix.
- Other ACS filters (F435W, F606W…) are config, not code: the adapter already
  parameterizes filter-dependent zero-points and PSF models.

## JWST (NIRCam first)

- Adapter over the STScI `jwst` pipeline: `_cal` products from stage 2
  (`calwebb_image2`), combination via stage 3 (`calwebb_image3`) whose
  `resample` step is the drizzle analogue — same defaults-first principle.
- Noise: JWST products carry `ERR`/`VAR_POISSON`/`VAR_RNOISE` arrays, so
  stage 4 becomes a *read + resample-consistency check* rather than a
  construction — a good test of the stage abstraction.
- PSF: STPSF (formerly WebbPSF) replaces TinyTim in tier 2; tiers 1 and 3
  carry over unchanged.

## Per-exposure frame products (`_flt`/`_flc` with cosmic rays)

For multi-frame forward modeling (fitting N undrizzled exposures
simultaneously instead of one mosaic): emit per-exposure cutouts, per-frame
noise maps (native-pixel Poisson + read noise — no correlated-noise issue),
per-frame native-pixel PSFs (TinyTim/ePSF, undrizzled), cosmic-ray/DQ masks
from the drizzle rejection stage, and the inter-frame WCS transforms. The
drizzle stage already computes everything needed; this is a packaging mode,
not a new pipeline.

## Other instruments / surveys

- Ground-based (e.g. archival CFHT/Subaru for environment studies) and other
  space missions as need arises — adapters again.
- **Euclid VIS is explicitly out of scope**: Euclid reduction and
  lens-modeling glue live in `euclid_strong_lens_modeling_pipeline`.

## Quality-of-life (unscheduled)

- Sample-level driver: table in → datasets out, streaming acquire→evict.
- PSF-model library caching (focus-diverse ePSFs, STPSF grids) in the size-
  capped cache.
- `verify_install`-style smoke path once the package is released.
