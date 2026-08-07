# guides/

Cross-instrument guides: the concepts every reduction in this workspace shares, independent of
which telescope the data came from. Read them after `scripts/start_here.py`, in this order:

- `output_contract.py` — the five products every reduction emits (`data.fits`, `noise_map.fits`,
  `psf.fits`, `psf_full.fits`, `reduction.json`) in depth: headers, units, the masked-by-noise
  convention, frame products, the ALMA visibility triplet, and loading into **PyAutoLens**. Runs
  offline on the output of `scripts/start_here.py` (clean-exits with a message if that hasn't
  run yet).
- `noise_maps.py` — the noise story: the per-domain recipes, the Casertano correlated-noise
  factor across pixfrac values, blank-sky closure diagnostics, and why chi-squared needs honest
  uncorrelated noise. Mostly runs offline; the real-data closure section is guarded.
- `target_spec.py` — every `TargetSpec` dial annotated, spec-YAML round-trips,
  `dataclasses.replace` variants, and the validation guard rails. Runs fully offline.
