# hst_wfc3_ir

HST WFC3/IR reductions — the infrared HgCdTe channel, which rewrites the CCD rules: no
shutter, MULTIACCUM up-the-ramp readout with per-exposure cosmic-ray rejection, `_flt`
products already in e-/s, no CTE correction at all, and 0.128"/pixel undersampling that
makes the drizzle scale the central dial (the adapter recommends 0.065"/pixel). Shared HST
machinery is documented in `scripts/hst_acs/`; the CCD sibling deltas are in
`scripts/hst_wfc3_uvis/`.

The validation anchor is the WFC3/IR F160W snapshot of SDSS J0252+0039 — including the
recorded zero-weight-speckle finding: pixfrac 0.8 at 0.065"/pixel on few-dither data left
zero-weight pixels and the finite-noise guard refused to ship, establishing the IR rule
"pixfrac -> 1.0 or a coarser scale".

Read in order:

1. `start_here.py` — the IR anchor reduction end to end: detector physics, the 0.065"
   drizzle choice, the fine-grid Casertano R, and the speckle finding.
2. `step_by_step.py` — the `calwf3` IR calibration chain and every stage's delta, with the
   WHT-uniformity reading on few-dither data.

Both scripts need network access (MAST + CRDS) and the `[hst]` extra (drizzlepac).
