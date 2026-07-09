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

## Keck NIRC2 AO — **in progress (phase 4, PyAutoReduce#11)**

- Design deltas live in [`keck_ao.md`](keck_ao.md); adapters `nirc2_narrow` /
  `nirc2_wide`, the acquire-backend seam (`archive: koa | mast`, PyKOA), the
  ground pre-combine stages (`calibrate`, `sky`), the `nirc2_native` combine
  backend (distortion-as-drizzle-pixmap), and the provisional-PSF contract
  (tier-A PSF-star candidates) implemented. SHARP-grounded (Lagattuta 2012;
  Chen 2016/2019); validation anchor B1938+666.
- Open: wide-camera distortion, subarrays, MULTISAM read noise, north-up
  resampling (see keck_ao.md open items).

## ALMA interferometry — **in progress (phase 5, PyAutoReduce#14)**

- Design lives in [`alma.md`](alma.md); the first visibility-domain product
  family: calibrated measurement sets → the `al.Interferometer.from_fits`
  triplet (visibilities / uv_wavelengths / noise_map, `(Nvis, 2)`), via the
  visibility branch acquire → split → extract → assemble → package.
- Headless modular CASA (`casatools`/`casatasks`) replaces the interactive
  `casa` shell; grounded in an active ALMA modeler's continuum recipe;
  validation anchor 2016.1.00282.S / G09v1.40.
- Open: scriptForPI restore automation, emission-line/cube extraction
  (reduction side of the shipped `alma-datacube` modelling work).

## Per-exposure frame products (`_flt`/`_flc` with cosmic rays)

**Shipped (HST)** as the opt-in `TargetSpec.frame_products` packaging mode —
per-exposure cutouts, ERR-based native-pixel noise maps, DQ + deepCR
cosmic-ray masks and the WCS manifest; design in `hst_acs_pipeline.md`
("Per-exposure frame products"). Open items:

- **Per-frame native-pixel PSFs** (TinyTim/ePSF, undrizzled) — the frames
  are not fully modeling-ready for PyAutoLens without them.
- **JWST analogue** — whether/what to emit pre-`calwebb_image3` is a
  research task (PyAutoMind
  `research/pyautoreduce/jwst_individual_frame_feasibility.md`).
- **Per-adapter `dq_bad_bits`** — refinement of the any-nonzero-bit masking
  policy, if a consumer ever needs to keep e.g. warm-pixel bits.

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
