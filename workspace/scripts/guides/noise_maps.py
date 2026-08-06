"""
Guides: Noise Maps
==================

The noise map is the least glamorous product of a reduction and the one your science leans on
hardest: every chi-squared and every Bayesian evidence **PyAutoLens** computes trusts it
completely. This guide develops the noise story across the workspace — the per-domain recipes,
the Casertano correlated-noise factor and where it comes from, the closure diagnostics that
validate a finished noise map, and the honest reason **PyAutoReduce** noise maps run ~30% higher
than the legacy SLACS ones.

Almost everything here runs offline: the Casertano factor and the construction recipe are pure
functions we demonstrate on synthetic arrays. Only the final closure section reads real data (the
output of `scripts/start_here.py`), and it is guarded — the script runs to completion without it.

__Contents__

- **Why Chi-Squared Cares:** Independent Gaussian noise is the likelihood's founding assumption — and resampling breaks it.
- **Imports:** The public noise helpers and the instrument adapters.
- **The Recipes Per Domain:** Constructed (HST/Keck), read (JWST), weighted (ALMA) — one recipe per data domain.
- **The Casertano Factor:** `casertano_r` demonstrated across pixfrac values, including the fine-grid branch.
- **A Synthetic Construction:** `noise_map_from` on a toy mosaic — see the recipe do its work.
- **The Masked-By-Noise Convention:** Bad pixels carry noise 1e8 so the likelihood ignores them.
- **Blank-Sky Closure:** The diagnostic that validates a noise map against the sky it ships with — synthetic first, then the real SLACS reduction (guarded).
- **Thirty Percent Above Legacy, By Design:** Why honest noise maps are bigger than the ones you may be used to.
- **Wrap Up:** Summary and good places to checkout next.

__Why Chi-Squared Cares__

A pixel-level likelihood is built on one assumption: that the noise map gives the RMS of
*independent* Gaussian noise in each pixel. Under that assumption chi-squared is calibrated,
posterior widths mean what they claim, and Bayesian evidence comparisons between lens models are
fair. Pixelized source reconstructions — the workhorse of substructure detection — lean on it
hardest of all.

Image resampling breaks the assumption. Drizzle (Fruchter & Hook 2002, PASP 114, 144,
https://ui.adsabs.harvard.edu/abs/2002PASP..114..144F) shares each input pixel among several
output pixels, so neighbouring output pixels carry common noise: the map of per-pixel RMS values
is still correct pixel-by-pixel, but it *understates* the uncertainty of any structure larger
than a pixel, and a likelihood that ignores the covariance is over-confident. The canonical
treatment is Casertano et al. (2000, AJ 120, 2747), whose appendix derives a scalar ratio R
between the true noise of large-scale structure and the naive per-pixel RMS as a function of the
drizzle geometry; the DrizzlePac Handbook (https://hst-docs.stsci.edu/drizzpac, section 3.4)
carries the same analysis. Inflating the per-pixel map by R is the standard scalar compromise —
exact for large scales, conservative in between.

The lensing literature takes this seriously. The SL2S survey drizzled preserving native
orientation and pixel scale *explicitly* to avoid correlated noise (Gavazzi et al. 2012,
https://arxiv.org/abs/1202.3852); Bayer et al. treat drizzle-correlated noise explicitly in their
lens-modeling error budgets (https://arxiv.org/abs/1803.05952,
https://arxiv.org/abs/2302.00480). **PyAutoReduce** applies R to every resampled noise map and
records it in the provenance — and offers two escape hatches for science that wants no
correlation at all: per-exposure frame products (nothing resampled, no R — see the
`individual.py` scripts) and the ALMA visibility branch (fit in the uv-plane, where the noise is
independent by construction).

__Imports__

The noise helpers are public **PyAutoReduce** API: `casertano_r` and `noise_map_from` are the
construction recipe, `empirical_background_rms` the closure diagnostic, `MASKED_NOISE_VALUE` the
masking sentinel. The instrument adapters supply the scale ratios the factor depends on.
"""

import json
from pathlib import Path

import numpy as np

from autoreduce import instruments
from autoreduce.noise.rms import (
    MASKED_NOISE_VALUE,
    casertano_r,
    empirical_background_rms,
    noise_map_from,
)

"""
__The Recipes Per Domain__

One recipe per data domain, chosen by what the upstream pipeline reliably provides:

- **HST and Keck — construct.** The drizzle/native combine emits a science mosaic (e-/s) and an
  inverse-variance weight map, so the pipeline constructs the RMS itself:

      sigma_i = R * sqrt( max(sci_i, 0) / t_exp  +  1 / W_i )

  The first term is the source's Poisson noise (floored at zero so blank sky does not go
  imaginary), the second the background variance encoded in the IVM weight — the recipe of Bayer
  et al. (https://arxiv.org/abs/1803.05952, section 3.1), scaled by the Casertano R.

- **JWST — read, don't construct.** The calwebb_image3 resample step already propagates a full
  ERR array (Poisson + read noise + flat) through the drizzle, so the pipeline *reads* it and
  applies the same R — reconstructing what the observatory pipeline already did well would only
  add ways to be wrong. A consistency block (below) guards against trusting a broken ERR.

- **ALMA — weights.** Each visibility carries a weight nominally equal to 1/sigma^2, so the
  per-visibility noise is sigma = 1/sqrt(weight) — and **no** Casertano factor, because nothing
  is resampled in the uv-plane. (Whether archival weights are correctly scaled is its own story
  — the uv-plane literature routinely recalibrates them against the visibility scatter, e.g.
  Hezaveh et al. 2016, https://arxiv.org/abs/1601.01388, section 2 — told in
  `scripts/alma/step_by_step.py`.)

__The Casertano Factor__

`casertano_r(pixfrac, scale_ratio)` returns R for a drizzle with drop size `pixfrac` (p) onto an
output grid `scale_ratio` (s) times the native pixel. Two limits anchor the intuition: p = 1 at
s = 1 is plain shift-and-add, the most correlated case (R = 1.5); shrinking the drops toward
interlacing (p -> 0) removes the sharing and R -> 1.

The table below is the trade every HST reduction navigates (at native output scale, s = 1) — the
same numbers the `scripts/hst_acs/dials.py` trade study explores with real data:
"""
print("Casertano R at s = 1.0 (output grid = native pixels):")
for pixfrac in (1.0, 0.8, 0.6, 0.4):
    print(f"  pixfrac {pixfrac:.1f}  ->  R = {casertano_r(pixfrac, 1.0):.3f}")

"""
Reading it: the SLACS convention (pixfrac 0.8) accepts a 36% noise inflation; pushing to
pixfrac 0.6 buys R = 1.25 but demands a dither pattern rich enough to fill the coverage (the
weight-uniformity diagnostic in every provenance record polices exactly this — RMS/median of the
weight map above ~0.2 means the drops got too small for the dithers).

There is a second branch worth respecting: when the output grid is *finer* than the drop
(s < p), correlation grows quickly — this is the fine-grid regime WFC3/IR reductions live in,
because the undersampled 0.128" native pixels are typically drizzled to ~0.065". The adapters
carry the native scales, so we can compute the real case:
"""
ir = instruments.get("wfc3_ir")
s_ir = ir.scale_ratio(ir.recommended_final_scale)  # s = 0.065 / 0.128
print(
    f"WFC3/IR at its recommended {ir.recommended_final_scale}\"/pix "
    f"(native {ir.native_scale}\"/pix, s = {s_ir:.3f}):"
)
for pixfrac in (1.0, 0.8):
    print(f"  pixfrac {pixfrac:.1f}  ->  R = {casertano_r(pixfrac, s_ir):.3f}")

"""
R materially above the s = 1 numbers is the price of recovering resolution from undersampled
pixels — worth paying, but only with the noise map telling the truth about it. (The
`scripts/hst_wfc3_ir/` examples add the companion rule: few-dither IR data at sub-native scales
needs pixfrac 1.0, or the coverage develops zero-weight speckles and packaging refuses to ship.)

__A Synthetic Construction__

The recipe is a pure function, so we can watch it work on a toy mosaic: flat sky with a bright
source, a uniform weight map, 1000 seconds of exposure, drizzled at the SLACS dials.
"""
rng = np.random.default_rng(1)

shape = (61, 61)
sigma_sky = 0.005                                   # e-/s background RMS
exptime = 1000.0                                    # seconds
wht = np.full(shape, 1.0 / sigma_sky**2)            # IVM weight: inverse background variance
sci = rng.normal(0.0, sigma_sky, shape)             # blank sky ...
sci[28:33, 28:33] += 5.0                            # ... plus a bright source (e-/s)

r_factor = casertano_r(pixfrac=0.8, scale_ratio=1.0)

noise = noise_map_from(sci, wht, exptime=exptime, correlated_noise_factor=r_factor)

print(f"Blank-sky corner noise: {noise[:10, :10].mean():.4f} e-/s "
      f"(R * sigma_sky = {r_factor * sigma_sky:.4f})")
print(f"Source-peak noise:      {noise[30, 30]:.4f} e-/s (Poisson term now dominates)")

"""
Blank sky lands at R * sigma_sky (a hair above it, because positive sky fluctuations contribute
a whisper of Poisson term — the floor at zero only protects the negative ones); on the source
the Poisson term sqrt(sci/exptime) takes over. Note also what the function refuses to do: zero or negative weights propagate as NaN, and the
packaging stage fails loudly if any land inside the cutout — a hole in the coverage is a problem
to fix, never a pixel to patch silently.

__The Masked-By-Noise Convention__

The one sanctioned exception to that loudness: *isolated* dead or fully-rejected pixels (routine
in deep resampled stacks) are shipped with their noise set to `MASKED_NOISE_VALUE` and the data
zeroed, so any chi-squared ignores them without a separate mask file. The policy is strict —
scattered singletons only, bounded fraction, never near the target — and every masked pixel is
counted in the provenance (`scripts/guides/output_contract.py` inspects the block).
"""
print(f"MASKED_NOISE_VALUE = {MASKED_NOISE_VALUE:.0e}  (effectively infinite noise)")

"""
__Blank-Sky Closure__

How do you *validate* a noise map? Close the loop against the data it ships with: the noise
floor the map claims for blank sky must match the sky RMS the mosaic actually shows. The
empirical side is a sigma-clipped RMS of the mosaic (`empirical_background_rms`); the claimed
side is the low percentile of the noise map divided by R. On the synthetic mosaic the closure is
exact by construction:
"""
sky_measured = empirical_background_rms(sci)
floor_claimed = float(np.nanpercentile(noise, 5)) / r_factor

print(f"Empirical sky RMS:            {sky_measured:.4f} e-/s")
print(f"Noise-map floor (pre-R):      {floor_claimed:.4f} e-/s")
print(f"Closure ratio (want ~1):      {floor_claimed / sky_measured:.3f}")

"""
On real data the same two numbers are computed for you: every HST/Keck provenance records
`noise.empirical_background_rms`, and the JWST read-don't-construct path records an equivalent
`sky_over_err_floor` consistency ratio (large disagreement there means the upstream ERR model
and the sky disagree — to be investigated, never absorbed). If you have run
`scripts/start_here.py`, we can close the loop on the real SLACS reduction:
"""
WORKSPACE = Path(__file__).resolve().parents[2]
out_dir = WORKSPACE / "output" / "slacs0008-0004"

if (out_dir / "reduction.json").exists():
    from astropy.io import fits

    record = json.loads((out_dir / "reduction.json").read_text())
    real_noise = fits.getdata(out_dir / "noise_map.fits").astype(float)
    physical = real_noise[real_noise < MASKED_NOISE_VALUE]

    r_real = record["noise"]["correlated_noise_factor"]
    sky_real = record["noise"]["empirical_background_rms"]
    floor_real = float(np.nanpercentile(physical, 5)) / r_real

    print(f"SLACS J0008-0004: R = {r_real:.3f}, empirical sky RMS = {sky_real:.3e} e-/s")
    print(f"  Noise-map floor / (R * sky RMS) closure: {floor_real / sky_real:.3f} (want ~1)")
else:
    print(
        "Real-data closure skipped: no output at output/slacs0008-0004 — run "
        "scripts/start_here.py (network required) to produce it. Everything above ran offline."
    )

"""
__Thirty Percent Above Legacy, By Design__

If you compare a **PyAutoReduce** HST noise map against a legacy lens-modeling dataset of the
same target, expect the new one to run ~30% higher (the SLACS validation measured a registered
noise ratio of ~1.31). This is not a bug to normalise away: the legacy SLACS noise maps do not
carry the Casertano correction, and at the SLACS drizzle dials R is ~1.36. The historical maps
understate the uncertainty of extended structure; the new ones state it. Posterior widths from
models fitted to the new datasets will be honestly wider — that is the point.

__Wrap Up__

One recipe per domain, one scalar R where resampling correlates the noise, loud failure where
the coverage breaks, a strict masking convention for the survivable cases, and a closure
diagnostic that ties the shipped map back to the shipped sky. When a noise map obeys all of
that, the chi-squared downstream can be trusted — which is the only reason any of this exists.

The following locations of the workspace are good places to checkout next:

- `scripts/hst_acs/dials.py`: the R-vs-pixfrac trade on real data, with the weight-uniformity limit.
- `scripts/hst_acs/individual.py`: frame products — the uncorrelated-noise escape hatch.
- `scripts/alma/step_by_step.py`: visibility weights, and why they get recalibrated before modeling.
- `scripts/guides/output_contract.py`: where the noise map sits in the full product contract.

This guide needs no network: the demonstrations are synthetic, and the single real-data section
is skipped cleanly when `scripts/start_here.py` has not been run.
"""
