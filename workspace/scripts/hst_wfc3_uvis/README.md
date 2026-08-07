# hst_wfc3_uvis

HST WFC3/UVIS reductions — the blue/optical CCD channel, taught as a *delta* against the
ACS/WFC reference (`scripts/hst_acs/`): same `_flc`/AstroDrizzle machinery, with `calwf3`,
`iref` references, the 0.0396"/pixel native scale, post-flash and a 63 ke- full well as the
channel-specific story. The validation anchor is the published Bayer et al.
(arXiv:1803.05952) F390W reduction of SDSS J0252+0039.

Read in order:

1. `start_here.py` — the anchor reduction end to end at the published dials, with the noise
   map validated against the published sigma_sky ~ 0.002 e-/s.
2. `step_by_step.py` — every stage's UVIS delta vs ACS, plus the live MAST filter-discovery
   idiom.
3. `psf.py` — the UVIS PSF on a star-rich field (Omega Cen F606W): photutils vs STARRED
   head to head.
4. `simulator.py` — inject a synthetic arc into the real J0252 frames and measure flux
   recovery.

All scripts need network access (MAST + CRDS) and the `[hst]` extra (drizzlepac).
