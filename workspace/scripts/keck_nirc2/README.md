# Keck NIRC2 AO

Reducing Keck II NIRC2 laser-guide-star AO imaging — the first ground-based path, where
calibration, sky subtraction and registration are the pipeline's own job. The validation
anchor is B1938+666, the SHARP Einstein ring in which Vegetti et al. 2012 detected a
10^8 solar-mass dark satellite.

Run the scripts in this order (each later script reads the cache/products of the first):

- `start_here.py` — the full pipeline end to end on B1938+666: KOA discovery, pinned frame
  ids, the ground stages, provenance, noise closure, and loading into PyAutoLens with the
  plate-scale correction.
- `step_by_step.py` — every ground stage demonstrated standalone on the cached frames:
  calibration matching, master flats/darks, the scaled running sky, phase-correlation
  registration, the one-pixmap combine, and the detector noise budget.
- `psf.py` — the AO PSF problem: tier-A PSF-star epochs, the vetting gates, candidates,
  and the `psf_provisional` contract.
- `simulator.py` — synthetic-ring injection into the real prepared frames, with flux
  recovery and registration-invariance checks.
