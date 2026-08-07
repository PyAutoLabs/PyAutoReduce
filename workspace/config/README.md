# config/

**PyAutoReduce needs no configuration YAML.** Unlike the PyAutoLens / PyAutoGalaxy workspaces —
whose `config/` folders customise priors, visualization and search behaviour — a reduction has no
tunable global state: everything a run needs is declared on the `TargetSpec` itself (see
`scripts/guides/target_spec.py`), and the pipeline is a pure function of that spec plus the
archive. There is deliberately nothing here for a user to edit.

The only contents are `config/build/`, which is **CI-only** — files consumed by the automated
build and test system, not by **PyAutoReduce**:

- `config/build/profile_smoke.yaml` — per-script environment for automated runs. Since no script
  currently runs in CI smoke tests (they all need archive network access + heavy instrument
  stacks; see `smoke_tests.txt`), this holds only writable cache-dir defaults.
- `config/build/no_run.yaml` — scripts to skip during automated runs, with the repo's
  `# SLOW` / `# NEEDS_FIX` tagging conventions. Currently empty.
