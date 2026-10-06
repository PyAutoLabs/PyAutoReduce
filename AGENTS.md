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

## The workspace

The user-facing tutorial workspace (per-instrument `start_here.py` /
`step_by_step.py` / `psf.py` / `simulator.py` examples, guides, README,
llms.txt) lives in its own repository:
https://github.com/PyAutoLabs/autoreduce_workspace (typically cloned as a
sibling, `../autoreduce_workspace`). It was staged under `workspace/` in this
repo until 2026-08-07 and extracted with history via
`git filter-repo --subdirectory-filter workspace`. Workspace scripts consume
only the public `autoreduce` API — a public-API change here means the
workspace examples may need updating; flag it in your PR.

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->

<!-- repos_sync:deliverable:begin -->
## Sessions end at their deliverable

A session ends when it reports its deliverable — never arm anything that
outlives the turn to wait for CI, a review or a merge: no `send_later`, no
`subscribe_pr_activity`, no `CronCreate`, no `ScheduleWakeup`, no `/loop`, no
`RemoteTrigger` create/update/run. Judge once, report, stop; the human re-runs
`/prm` (or the batch review) when it is green. Measured: five batch members
armed hourly check-ins on 2026-08-31, and a mobile `/prm` re-armed a 60-minute
`send_later` hourly all night on 2026-09-03 with no task active, draining usage.
<!-- repos_sync:deliverable:end -->

<!-- repos_sync:filing:begin -->
## Where to file

Questions, help with code or an analysis, ideas, bug reports and results from a
user or collaborator — or an agent acting for one — go to
<https://github.com/orgs/PyAutoLabs/discussions> in the matching category
(Help & Questions, Ideas & Proposals, Bugs & Errors, Show and tell;
Announcements is maintainers-only), never to this repo's Issues. An agent never
runs `gh issue create` for such a report: it drafts the title, category and
body and hands them to the human (sessions cannot create Discussions). Only the
development flow — Mind prompt → `/start_dev` → `/create_issue` → one issue per
task → PR — opens issues here. Why: `PyAutoMind/policy/community_surface.md`.
<!-- repos_sync:filing:end -->

<!-- repos_sync:standards:begin -->
## Shared standards

Before changing a shared interface, consult the applicable
[organism standard](https://github.com/PyAutoLabs/PyAutoBrain/blob/main/docs/standards.md)
on demand, identify affected consumers, and validate their adoption. Change
generated guidance at its canonical source and regenerate.
<!-- repos_sync:standards:end -->
