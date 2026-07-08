# PyAutoReduce — Agent Instructions

Data reduction of HST (and future JWST/other) imaging into modeling-ready
datasets for PyAutoLens/PyAutoGalaxy. Part of the PyAutoLabs workspace — the
workspace-level `AGENTS.md` (routing, safety rules, workflow) applies here.

## What this repo is

- Package `autoreduce`: pipeline stages `acquire/`, `align/`, `drizzle/`,
  `noise/`, `psf/`, `package/`, with instrument specifics isolated in
  `instruments/` adapters.
- Output contract: the `al.Imaging.from_fits` product set — `data.fits`,
  `noise_map.fits`, `psf.fits`, `psf_full.fits` — plus `reduction.json`
  provenance. The reference quality bar is the SLACS ACS/F814W reductions.
- Design docs are authoritative while the project is young:
  `docs/design/hst_acs_pipeline.md` (HST/ACS stages, defaults vs lensing
  deviations, validation) and `docs/design/roadmap.md` (WFC3, JWST,
  per-exposure frame products).

## Boundaries

- **Never imports** `autolens` / `autogalaxy` / `autoarray` / `autofit` — it
  emits their input format only, and stays releasable independently.
- **Default pipelines first**: stages wrap the instrument's standard tooling
  (`astroquery.mast`, `drizzlepac`, `photutils`); any deviation from STScI
  defaults must be justified by a lensing requirement and documented in the
  design doc.
- Unit tests in `test_autoreduce/` are numpy/astropy-only — no network, no
  drizzlepac. Anything needing MAST or the heavy STScI stack lives in
  `prototypes/` or (later) integration scripts.
- FITS files are never committed (`.gitignore` enforces this); `prototypes/`
  writes to `prototypes/output/` and `prototypes/cache/`.
