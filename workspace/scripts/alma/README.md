# ALMA — uv-plane visibility reduction

The visibility domain: calibrated ALMA measurement sets in, the
`al.Interferometer.from_fits` triplet (`data.fits` / `uv_wavelengths.fits` /
`noise_map.fits`, each `(Nvis, 2)`) out. No image is ever made — lens modeling
fits the visibilities directly.

Read in this order:

1. `start_here.py` — the full branch end-to-end on the validation anchor
   G09v1.40 (project 2016.1.00282.S): why visibilities, the archive/restore
   reality, the split → extract → assemble → package chain, and the
   **PyAutoLens** dirty-image round trip.
2. `step_by_step.py` — every stage by hand via the public modules: MS anatomy,
   headless modular CASA, the bandwidth-smearing budget for channel averaging,
   and the WEIGHT-column audit against the visibility scatter.
3. `simulator.py` — CASA `simobserve` as the acquire-alternative: a synthetic
   Jy/pixel source through the identical chain, with a flux-recovery closure test.

All three need modular CASA (`pip install casatools casatasks`); the first two
need a calibrated MS directory on disk (they print acquisition guidance and exit
cleanly without one). Continuum only — cube/line extraction is deferred.
