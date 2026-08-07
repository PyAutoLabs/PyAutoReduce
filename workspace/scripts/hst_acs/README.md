# HST ACS/WFC

The flagship folder of the workspace: ACS/WFC is **PyAutoReduce**'s reference instrument, and
its SLACS parity reductions set the quality bar every other pipeline is described against. The
other instrument folders teach their *deltas* relative to what is established here.

Recommended reading order:

- `start_here.py` — the default pipeline end to end on the SLACS lens SDSS J0008-0004: one
  frozen `TargetSpec`, one `reduce_target` call, a modeling-ready dataset.
- `step_by_step.py` — the same reduction dissected stage by stage: MAST/CRDS acquisition, WCS,
  drizzle, noise construction, PSF and packaging, each grounded in the STScI handbooks.
- `dials.py` — the trade study for the four user-facing dials (`final_scale`, `final_pixfrac`,
  `final_kernel`, `cr_method`), with the Casertano R factor computed live.
- `psf.py` — the PSF story: star selection, the ePSF tiers, the star-pass dial, STARRED, and
  the diagnostics shipped in `reduction.json`.
- `individual.py` — per-exposure frame products (`frame_products=True`): native-frame cutouts
  with uncorrelated noise, per-frame deepCR masks and per-frame PSFs.
- `simulator.py` — injection testing: place a synthetic lensed arc into the real exposures, run
  the identical pipeline, and verify the flux comes back out.
