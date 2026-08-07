# PyAutoReduce Workspace — Agent Instructions

This is the tutorial and example workspace for **PyAutoReduce**, a Python library that reduces raw
archival telescope data (HST, JWST, Keck-AO, ALMA, survey cutout services) into modeling-ready
datasets for **PyAutoLens** and **PyAutoGalaxy**. These are the canonical, agent-agnostic
instructions for this repo; `CLAUDE.md` imports this file.

## Repository Structure

- `scripts/` — Runnable Python tutorial scripts, organised by instrument:
  - `start_here.py` — the top-level overview: why reduction quality matters, the declarative
    `TargetSpec` philosophy, a complete HST/ACS reduction of a SLACS lens, routing to every folder.
  - `guides/` — cross-instrument guides: `output_contract.py` (the four-file + `reduction.json`
    contract), `noise_maps.py` (noise recipes + the Casertano correlated-noise factor),
    `target_spec.py` (every `TargetSpec` dial; runs offline).
  - `hst_acs/`, `hst_wfc3_uvis/`, `hst_wfc3_ir/` — HST reductions (AstroDrizzle path).
  - `jwst_nircam/` — JWST NIRCam reductions (calwebb_image3 path; MJy/sr units).
  - `keck_nirc2/` — Keck NIRC2 LGS-AO reductions (ground-based calibrate/sky stages, native combine).
  - `alma/` — ALMA visibility extraction (uv-plane datasets for `al.Interferometer`).
  - `surveys/` — survey cutout fetching (colour context only, never modeling data).
- `notebooks/` — Jupyter notebook versions, generated from `scripts/` at release time (do not
  edit directly).
- `config/` — build/CI configuration only; **PyAutoReduce** itself needs no config YAML.
- `dataset/` — per-target `TargetSpec` YAML files only; FITS is never committed.
- `output/`, `cache/` — reduction products and downloaded exposures (gitignored, created at runtime).

## Running Scripts

Scripts are run **from the workspace root**:

```bash
python scripts/hst_acs/start_here.py
```

Path logic inside scripts does not depend on the cwd — every script anchors paths to the
workspace root via `Path(__file__).resolve().parents[...]`, because **PyAutoReduce**'s drizzle
step changes the working directory internally and requires absolute paths.

**Almost every script needs network access and a heavy instrument stack** (MAST/KOA/ALMA archive
downloads, drizzlepac / jwst / casatools). First runs of a target download exposures into
`cache/` (re-used afterwards) and write products into `output/<target>/`. Both directories are
gitignored. The exceptions that run offline are `scripts/guides/target_spec.py` and most of
`scripts/guides/noise_maps.py`.

### Standard imports

```python
from autoreduce import TargetSpec, reduce_target
```

The public surface is exactly these two names plus the documented helpers
(`autoreduce.instruments`, `autoreduce.validation`, `autoreduce.noise.rms`). The canonical idiom
— build a frozen `TargetSpec`, call `reduce_target`, read the returned provenance dict, load the
FITS off disk — is shown in `scripts/start_here.py`; read that rather than relying on a recipe
here.

## Style Contract

Every script is a long-form narrated tutorial in the PyAuto workspace house style:

- Module docstring with the title underlined with `=`, framing paragraphs, and a `__Contents__`
  bullet list.
- Every prose block is a bare triple-quoted `"""__Section__"""` docstring at module level. **No
  `# ----` banner comments.** Inline `#` is reserved for short line-level notes and trailing
  per-kwarg explanations in constructor calls.
- Second person, present tense; **PyAutoReduce** / **PyAutoLens** bolded on every mention; papers
  cited inline with arXiv IDs; honest caveats are first-class sections.
- Scripts close with `__Wrap Up__` and, where the script needs network, a final
  `__Env__ (Developer Only)` section containing the bare line `ENV: network`.

## Testing

`smoke_tests.txt` is the curated smoke list. It is currently **all commented out**: every script
needs archive network access plus a heavy dependency stack, so none run in CI smoke tests yet.
Each commented entry cites its reason. `config/build/profile_smoke.yaml` carries only writable
cache-dir defaults; `config/build/no_run.yaml` is empty apart from its conventions.

## Bulk-edit safety

When editing the same region across many scripts in one pass (adding a section, renaming a
symbol, updating an import block), only rewrite the targeted region. **Never produce a whole-file
write unless you have read the entire current contents of that file** — a whole-file write based
on a header skim silently deletes every section below the header. Prefer targeted edits over
whole-file writes.

## Hard Boundaries

- Workspace scripts consume the **public** `autoreduce` API (`TargetSpec`, `reduce_target`, the
  documented helpers) and optionally `autolens` for loading the finished products. They **never
  import private `_`-prefixed autoreduce functions** — the pipeline stages are internal, and the
  step-by-step scripts teach them by reading the evidence out of `reduction.json` and the `work/`
  directory, not by calling them.
- `autolens` imports are always guarded with `try/except ImportError` — the reduction itself
  never depends on the modeling stack (**PyAutoReduce** never imports it either).
- No FITS is ever committed; `output/` and `cache/` stay out of git.

## Related Repos

The PyAutoReduce stack (all on the `PyAutoLabs` GitHub org):

- https://github.com/PyAutoLabs/PyAutoReduce — the library this workspace demonstrates.
- https://github.com/PyAutoLabs/PyAutoLens — strong-lens modeling; loads these products via
  `al.Imaging.from_fits` / `al.Interferometer.from_fits`.
- https://github.com/PyAutoLabs/PyAutoGalaxy — galaxy-morphology modeling; same input format.
- https://github.com/PyAutoLabs/autolens_workspace — the modeling workspace these datasets feed;
  its `data_preparation` examples state the standards every product here satisfies.
- https://github.com/PyAutoLabs/PyAutoHands — notebook generation + CI tooling.

## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
