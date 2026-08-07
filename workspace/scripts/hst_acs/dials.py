"""
HST ACS: Dials
==============

Most of a **PyAutoReduce** reduction is deliberately not up for debate: the calibrated `_flc`
exposures come straight from MAST, the noise recipe is fixed, the PSF is drizzled exactly like the
science mosaic. But four dials on the `TargetSpec` are exposed precisely because the strong-lensing
literature does *not* agree on them: the output pixel scale (`final_scale`), the drizzle drop size
(`final_pixfrac`), the drizzle kernel (`final_kernel`) and the cosmic-ray strategy (`cr_method`).

This script is the trade study for those dials. It explains what each one does physically, what it
costs in correlated noise (with the Casertano factor computed live), what the SLACS, SL2S and
Bayer et al. reductions actually chose, and how to decide for your own target. Everything down to
the final section runs offline in seconds — the only network-touching part is an optional
re-reduction at a second pixfrac, which is switched off by default.

If you have not yet run `start_here.py` in this folder, read it first: this script assumes you
know the shape of the default SLACS reduction it dissects.

__Contents__

- **Imports:** Import **PyAutoReduce** and the supporting libraries.
- **The Four Dials:** Where they live on the `TargetSpec` and why only these four are dials.
- **Final Scale:** The output pixel size — resolution versus per-pixel depth and noise correlation.
- **Final Pixfrac:** The drizzle drop size — the PSF/noise-correlation trade at the heart of drizzling.
- **The Casertano Factor, Live:** Compute R across pixfrac and scale with `casertano_r`.
- **Weight Uniformity:** The diagnostic that tells you when pixfrac is too small for your dithers.
- **Final Kernel:** The drop shape — square default, and what the alternatives buy you.
- **What The Literature Chose:** SLACS V, SLACS IX, SL2S and Bayer et al. disagree — a field guide.
- **Cosmic-Ray Strategy:** `driz_cr` median flagging, its failure mode on lens cores, and the deepCR opt-in.
- **Decision Guidance:** A short flowchart for choosing dials on a new target.
- **A Second Reduction (Optional):** Re-reduce SLACS J0008-0004 at a different pixfrac and compare.
- **Wrap Up:** Where to go next.

__The Four Dials__

Every dial that the literature is agreed on is hard-wired: the mosaic is north-up
(`final_rot=0`), inverse-variance weighted (`final_wht_type='IVM'`), in electrons per second
(`final_units='cps'`), sky-matched (`skymethod='globalmin+match'`). Those defaults are documented
deviations-or-adoptions of the AstroDrizzle defaults (DrizzlePac Handbook,
https://hst-docs.stsci.edu/drizzpac), justified once in the **PyAutoReduce** design docs and not
re-litigated per target.

The four that remain are exposed on the `TargetSpec` because published lensing reductions genuinely
differ on them, so an informed user must be able to differ too:

- `final_scale` (default 0.05"/pixel — the SLACS convention, equal to the ACS/WFC native scale).
- `final_pixfrac` (default 0.8).
- `final_kernel` (default "square").
- `cr_method` (default "driz_cr").

Every choice is recorded in `reduction.json`, so two datasets reduced with different dials are
always distinguishable after the fact.
"""

"""
__Imports__

The dial mathematics below needs only the `casertano_r` helper and numpy — no network, no
drizzlepac. The optional re-reduction at the end imports `reduce_target` like every other script.
"""

from pathlib import Path
import os

import numpy as np

from autoreduce import TargetSpec
from autoreduce.noise.rms import casertano_r
from autoreduce.drizzle.diagnostics import WEIGHT_UNIFORMITY_LIMIT

"""
__Final Scale__

`final_scale` sets the output mosaic's pixel size in arcseconds. For ACS/WFC the native detector
scale is 0.05"/pixel, and the default keeps it: SLACS drizzled to 0.05"/pixel, every legacy SLACS
modeling dataset is on that grid, and matching it means masks, light-profile sizes and pixel-scale
conventions carry over unchanged into **PyAutoLens**.

Drizzling to a *finer* grid (0.03"/pixel is common in time-delay-lens work) partially recovers
resolution from well-dithered data, because the drizzle algorithm (Fruchter & Hook 2002,
PASP 114, 144, https://ui.adsabs.harvard.edu/abs/2002PASP..114..144F) interlaces the dithered
samples onto the sub-pixel grid. The costs are real, though:

- Each output pixel receives fewer electrons, so the per-pixel signal-to-noise drops.
- Fine grids need *more, well-placed* dithers to fill every output pixel — with too few, the
  weight map develops holes and the reduction fails its own uniformity check (see below).
- Noise correlation between neighbouring pixels grows (next section) unless pixfrac shrinks
  with the scale — and shrinking pixfrac needs yet more dithers.

For galaxy-scale lens modeling on typical 2-8 exposure ACS visits, the native 0.05"/pixel default
is the robust choice; reach for finer scales only with rich dither sets and a science case that
needs them.
"""

"""
__Final Pixfrac__

`final_pixfrac` is the fraction by which each input pixel is shrunk before being "dripped" onto
the output grid — the central free parameter of the drizzle algorithm. At `pixfrac=1.0` drizzle
behaves like shift-and-add: every input pixel overlaps several output pixels, photometry is
maximally stable, but neighbouring output pixels share input electrons and their noise is
correlated. As pixfrac shrinks toward 0, drizzle approaches pure interlacing: sharper effective
PSF, less noise correlation — but each drop lands on fewer output pixels, so sparse or poorly
placed dithers leave under-covered pixels with wildly varying weights.

There is no free lunch here, only a trade, which is exactly why **PyAutoReduce** exposes the dial
instead of hiding it. The default of 0.8 follows the SLACS IX-era convention for dithered ACS
data: most of the noise-decorrelation benefit of shrinking the drop, while staying robust on the
4-8 exposure visits typical of lens programs.
"""

"""
__The Casertano Factor, Live__

Whatever pixfrac you choose, the drizzled mosaic's pixel-to-pixel RMS *understates* the noise a
lens-model chi-squared actually experiences, because drizzling shares each input electron between
neighbouring output pixels. The standard correction is the analytic factor R of Casertano et al.
2000 (AJ 120, 2747, appendix) — a function of only pixfrac p and the scale ratio s
(output scale / native scale). **PyAutoReduce** multiplies every mosaic noise-map by R and records
it in `reduction.json` as `drizzle.correlated_noise_factor`.

`casertano_r` is a public helper, so we can map the whole trade space in a few lines. The s = 1
column is the ACS default grid; the s = 0.6 column shows how much harsher the correction gets on
a 0.03"/pixel fine grid.
"""

print("Casertano correlated-noise factor R:\n")
col_native = 'R (s=1.0, 0.05")'
col_fine = 'R (s=0.6, 0.03")'
print(f"{'pixfrac':>8} | {col_native:>17} | {col_fine:>17}")
print("-" * 50)
for pixfrac in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
    r_native = casertano_r(pixfrac, 1.0)  # output at the ACS native 0.05"/pixel
    r_fine = casertano_r(pixfrac, 0.6)  # output at 0.03"/pixel, s = 0.03/0.05
    print(f"{pixfrac:>8.1f} | {r_native:>17.3f} | {r_fine:>17.3f}")

"""
The native-scale column reproduces the numbers quoted throughout this workspace: R = 1.500 at
pixfrac 1.0, 1.364 at the 0.8 default, 1.250 at 0.6. Two readings of this table matter:

- Shrinking pixfrac from 1.0 to 0.6 buys you ~17% less noise inflation at native scale — real,
  but not transformative. The fine-grid column is where R gets punishing, which is another reason
  the default stays at the native scale.
- R is a *scalar approximation* to a correlation that is really a neighbourhood covariance. It
  makes the noise-map faithful for a chi-squared that treats pixels as independent — the honest
  correction, not a perfect one. `guides/noise_maps.py` develops this point in full.
"""

"""
__Weight Uniformity__

The guard-rail on shrinking pixfrac is coverage: if the shrunken drops no longer tile the output
grid evenly, the drizzle weight map becomes ragged and per-pixel depths vary wildly. The
DrizzlePac Handbook's rule of thumb is that the RMS of the weight map should stay below ~20-30%
of its median; **PyAutoReduce** computes exactly this statistic on every run, stores it at
`reduction.json["drizzle"]["weight_uniformity"]`, and flags it against a limit of
"""

print(f"\nWEIGHT_UNIFORMITY_LIMIT = {WEIGHT_UNIFORMITY_LIMIT}  (RMS/median of the drizzle weight map)")

"""
The default SLACS J0008-0004 reduction (7 exposures, pixfrac 0.8) measures ~0.066 — comfortably
uniform. If you shrink pixfrac or refine the scale and this diagnostic climbs toward the limit,
the dials have outrun your dither set: back off pixfrac toward 1.0, coarsen the scale, or accept
the reduction refusing to ship (the packaging stage crashes loudly on non-finite noise pixels
rather than silently interpolating holes — see `hst_wfc3_ir/start_here.py` for a real example of
that guard firing).
"""

"""
__Final Kernel__

`final_kernel` sets the shape of the drop each input pixel is drizzled through. The default
"square" (a shrunken image of the pixel itself) is the DrizzlePac default and the workhorse of
published reductions. The alternatives ("gaussian", "point", "turbo", "lanczos3") trade edge
sharpness against noise properties and speed; SLACS IX used a Gaussian kernel with MultiDrizzle,
which is part of why the legacy SLACS mosaics cannot be reproduced bit-for-bit today.

Unless you are chasing parity with a specific historical reduction, keep "square": it is the
best-characterised choice, and the Casertano R treatment above is derived for it.
"""

"""
__What The Literature Chose__

The reason these dials exist: three exemplary lensing reductions of HST imaging made three
different sets of choices, each defensible.

- **SLACS V (Bolton et al. 2008, https://arxiv.org/abs/0805.1931)** — for single-exposure
  snapshot F814W imaging, did *not* drizzle at all ("not well suited to single-exposure Snapshot
  data"): frames were rectified by bilinear interpolation (ACSPROC), cosmic rays masked with
  L.A.Cosmic (van Dokkum 2001, PASP 113, 1420), and the TinyTim model PSF rectified through the
  identical resampling — the precedent for **PyAutoReduce**'s drizzled-PSF invariant.
- **SLACS IX (Auger et al. 2009)** — drizzled the dithered subset to 0.05"/pixel with a Gaussian
  kernel and an unstated pixfrac. The convention this workspace's defaults descend from, and a
  cautionary tale about recording your dials (which `reduction.json` now does).
- **Bayer et al. (arXiv:1803.05952)** — the WFC3/UVIS F390W reduction of SDSS J0252+0039 used
  the native scale with pixfrac 1.0, and handled noise correlation not with a scalar R but with
  blank-sky *realizations* drizzled through the same footprint. Maximum robustness, at the price
  of the strongest per-pair correlation (R = 1.5) — coherent because their analysis measured the
  noise power spectrum directly.
- **SL2S (Gavazzi et al. 2012, arXiv:1202.3852)** — drizzled WFPC2 data preserving the native
  CCD orientation and pixel scale *explicitly to avoid producing correlated noise*, and
  CR-cleaned single ACS/WFC3 exposures with L.A.Cosmic. The cleanest published statement that
  uncorrelated noise-maps are a lens-modeling requirement, not a nicety.

None of these is wrong. The defaults in this workspace (0.05", pixfrac 0.8, square, driz_cr) are
the choice **PyAutoReduce** validated against SLACS parity; the dials let you reproduce any of
the others.
"""

"""
__Cosmic-Ray Strategy__

`cr_method` selects how cosmic rays are removed at the combine stage, and it earns its place as a
dial because the default has a measured failure mode on exactly the object you care about.

**"driz_cr" (default).** AstroDrizzle's standard approach: drizzle each exposure separately,
build a median image, blot it back to each frame's geometry, and flag pixels that deviate
sharply from the blotted median (DrizzlePac Handbook). It needs >= 2-3 overlapping exposures and
it works well on flat sky — but on the steep brightness gradient of a lens-galaxy core,
sub-pixel dither offsets make the blotted median a poor predictor of the pixel it is compared
against. On the SLACS acceptance targets this mis-flagging was measured to cost roughly a third
of the deflector's central flux before the comparison thresholds were tuned — the kind of bias
that propagates straight into a lens-light model.

**"deepcr" (opt-in).** Per-frame cosmic-ray masks from the deepCR convolutional network
(Zhang & Bloom 2020, ApJ 889, 24, https://ui.adsabs.harvard.edu/abs/2020ApJ...889...24Z),
trained largely on ACS F814W data, written into each frame's DQ array; the combine then becomes a
plain weighted-mean drizzle with the median/blot/driz_cr machinery switched off. Because each
frame is cleaned independently, the lens core's gradient never enters a frame-to-frame
comparison. Requires the `[frames]` extra (deepCR pulls torch) and is available for `acs_wfc` and
`wfc3_uvis` (WFC3/IR needs no CR model — its up-the-ramp readout rejects cosmic rays at
calibration; see `hst_wfc3_ir/start_here.py`).

One implementation detail is worth knowing because it is invisible when it works: AstroDrizzle's
`resetbits` default *clears* DQ bit 4096 — the very bit the per-frame masks are written into — so
the deepCR path must run with `resetbits=0`. **PyAutoReduce** pins this internally (and tests
it); if you ever hand-roll a drizzle over deepCR-flagged frames, carry the same setting or your
masks will be silently ignored.

The library default remains "driz_cr" with tuned thresholds: flipping the default to deepCR is
deliberately gated on a human-reviewed SLACS validation, not on this workspace. Use
`cr_method="deepcr"` today when your visit has few exposures, sub-pixel dithers, or you see the
tell-tale flux depression at the lens centre in the difference between a driz_cr and a no-CR
star-pass mosaic (`psf.py` shows how to make one).
"""

"""
__Decision Guidance__

A short field guide, dial by dial, for a new galaxy-scale lens target on ACS:

- **final_scale** — keep 0.05" (native). Go finer only with >= 8 well-placed dithers *and* a
  science case (e.g. point-image astrometry) that needs it; then expect to lower pixfrac and
  re-check weight uniformity.
- **final_pixfrac** — keep 0.8. Raise to 1.0 if the weight-uniformity diagnostic climbs (few
  dithers, IR-style speckle holes); lower toward 0.6 only with rich dither sets when you want the
  last ~8% off the R factor.
- **final_kernel** — keep "square" unless chasing parity with a Gaussian-kernel legacy reduction.
- **cr_method** — keep "driz_cr" for well-dithered visits; switch to "deepcr" for few-exposure
  visits or when the lens core shows CR-rejection bites.

And in every case: read `reduction.json` afterwards. The dials you chose, the R they implied and
the uniformity they achieved are all recorded — a reduction whose diagnostics you have not read
is not finished.
"""

"""
__A Second Reduction (Optional)__

Everything above ran offline. To *feel* the pixfrac trade on real data, re-reduce the
`start_here.py` target at pixfrac 1.0 and compare the two `reduction.json` records (and, if you
wish, the two noise maps — their ratio should track R(1.0)/R(0.8) = 1.100 in the blank-sky
limit).

This re-runs the full pipeline (the exposure cache makes it a re-drizzle, not a re-download), so
it is gated behind an environment variable rather than running on import:

    AUTOREDUCE_DIALS_RERUN=1 python scripts/hst_acs/dials.py
"""

if os.environ.get("AUTOREDUCE_DIALS_RERUN") == "1":
    from autoreduce import reduce_target

    WORKSPACE = Path(__file__).resolve().parents[2]
    CACHE_ROOT = WORKSPACE / "cache"  # shared with start_here.py — the exposures are already here
    OUTPUT_ROOT = WORKSPACE / "output"

    spec = TargetSpec(
        name="slacs0008-0004_pixfrac10",  # a distinct name so the default products are untouched
        ra=2.012333,  # SDSS J0008-0004 (SLACS)
        dec=-0.068944,
        proposal_ids=("10886",),  # the SLACS ACS program
        final_pixfrac=1.0,  # the dial under study; all other dials stay at their defaults
    )

    print("\nRe-reducing SLACS J0008-0004 at pixfrac 1.0 (re-uses the cached exposures)...")
    record = reduce_target(spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)

    print("\npixfrac 1.0 diagnostics:")
    print(f"  correlated_noise_factor : {record['drizzle']['correlated_noise_factor']:.3f}")
    print(f"  weight_uniformity       : {record['drizzle']['weight_uniformity']}")
    print("\nCompare against output/slacs0008-0004/reduction.json from start_here.py.")
else:
    print(
        "\nOptional re-reduction skipped (set AUTOREDUCE_DIALS_RERUN=1 to re-reduce the"
        " start_here.py target at pixfrac 1.0 and compare diagnostics)."
    )

"""
__Wrap Up__

You have seen the four dials **PyAutoReduce** exposes on an ACS reduction, the correlated-noise
mathematics that couples the scale and pixfrac choices, the uniformity diagnostic that polices
them, and the published reductions whose disagreements motivated making them dials at all.

Good places to checkout next:

- `hst_acs/psf.py` — the star-pass dial (`psf_star_pass`) interacts with `cr_method`: CR flags
  eat star cores, and the no-CR star pass exists precisely to win them back.
- `hst_acs/individual.py` — per-exposure frame products sidestep the drizzle dials entirely
  (nothing is resampled, so R = 1 by construction).
- `guides/noise_maps.py` — the full noise story these dials feed into.
- `hst_wfc3_ir/start_here.py` — the pixfrac rule colliding with real few-dither IR data.

__Env__ (Developer Only)

ENV: network
"""
