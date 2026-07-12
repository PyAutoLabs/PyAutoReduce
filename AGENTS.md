# PyAutoReduce — Agent Instructions

Data reduction of HST, JWST and Keck-AO (and future other) imaging into
modeling-ready datasets for PyAutoLens/PyAutoGalaxy. Part of the PyAutoLabs
workspace — the workspace-level `AGENTS.md` (routing, safety rules, workflow)
applies here.

## What this repo is

- Package `autoreduce`: pipeline stages `acquire/`, `align/`, `calibrate/` +
  `sky/` (ground-based only), `drizzle/`, `noise/`, `psf/`, `package/`, with
  instrument specifics isolated in `instruments/` adapters.
- Output contract: the `al.Imaging.from_fits` product set — `data.fits`,
  `noise_map.fits`, `psf.fits`, `psf_full.fits` — plus `reduction.json`
  provenance. The reference quality bar is the SLACS ACS/F814W reductions.
- Design docs are authoritative while the project is young:
  `docs/design/hst_acs_pipeline.md` (HST/ACS stages, defaults vs lensing
  deviations, validation), the per-instrument delta docs (`wfc3.md`,
  `jwst.md`, `keck_ao.md`) and `docs/design/roadmap.md`.

## Boundaries

- **Never imports** `autolens` / `autogalaxy` / `autoarray` / `autofit` — it
  emits their input format only, and stays releasable independently.
- **Default pipelines first**: stages wrap the instrument's standard tooling
  (`astroquery.mast`, `drizzlepac`, `photutils`); any deviation from STScI
  defaults must be justified by a lensing requirement and documented in the
  design doc.
- Unit tests in `test_autoreduce/` are numpy/astropy-only — no network, no
  drizzlepac/jwst stack. The one sanctioned extra is the lightweight
  standalone `drizzle` resampler (behind `pytest.importorskip`) for the
  nirc2_native backend. Anything needing an archive or the heavy STScI
  stack lives in `prototypes/` or (later) integration scripts.
- FITS files are never committed (`.gitignore` enforces this); `prototypes/`
  writes to `prototypes/output/` and `prototypes/cache/`.

<!-- repos_sync:history:begin -->
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.
<!-- repos_sync:history:end -->
