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

## HST/WFC3 (IR + UVIS) — **in progress (phase 2, PyAutoReduce#4)**

- Design deltas live in [`wfc3.md`](wfc3.md); adapters `wfc3_uvis` /
  `wfc3_ir` implemented. UVIS is ACS-like (CTE-corrected `_flc`); IR differs
  (`_flt`, no CTE correction, up-the-ramp CR rejection, 0.128″ native →
  recommended 0.065″ output).
- Other ACS/WFC3 filters (F435W, F606W…) are config, not code: the adapter
  already parameterizes the filter-dependent pieces.

## JWST (NIRCam first) — **in progress (phase 3, PyAutoReduce#6)**

- Design deltas live in [`jwst.md`](jwst.md); adapters `nircam_sw`/`nircam_lw`
  + the combine-backend dispatch (`astrodrizzle` | `jwst_image3`) implemented;
  noise = read propagated ERR × R; validated on the COSMOS-Web ring, four
  bands, against the autolens_assistant demo dataset.
- PSF: STPSF tier-2 back-end still open (tier-1 ePSF carries over).

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
