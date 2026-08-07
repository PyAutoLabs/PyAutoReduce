"""
Step By Step: HST WFC3/UVIS
===========================

This script walks the WFC3/UVIS reduction stage by stage, teaching what the instrument
pipeline does to the data at each step — but as a *delta* against the ACS/WFC reference.
UVIS is the ACS-like channel: the same `_flc` CTE-corrected products, the same AstroDrizzle
combination, the same noise recipe and PSF tiers. What changes is the calibration pipeline
name (`calwf3` instead of `calacs`), the reference-file environment (`iref` instead of
`jref`), the plate scale (0.0396"/pixel), the saturation level (~63 ke-) and the post-flash
term in the noise budget.

For everything the two channels *share* — drizzle theory and its dials, the alignment story,
the full noise derivation, the PSF tier system — read `hst_acs/step_by_step.py`, which is the
canonical stage-by-stage reference. This script covers each stage briefly, flags the UVIS
delta, and defers the depth by cross-reference rather than duplication.

Along the way it demonstrates one genuinely new idiom: *live MAST discovery* — asking the
archive what imaging exists at a position before committing to a filter, and filtering the
answer down to real, directly-reducible observations.

__Contents__

- **Imports:** Import the required Python libraries.
- **Paths:** Anchor every path to the workspace root, absolutely.
- **Stage 1, Acquisition:** MAST `_flc` downloads, `iref` references and live filter discovery.
- **Stage 2, Calibration:** What `calwf3` already did to the exposures before we downloaded them.
- **Stage 3, Alignment:** Shared with ACS — the archive's Gaia-tied WCS is trusted.
- **Stage 4, Drizzle:** Shared machinery, UVIS-native scale, and the 63 ke- saturation ceiling.
- **Stage 5, Noise:** The shared recipe, with post-flash in the background budget.
- **Stage 6, PSF and Packaging:** Shared with ACS; cross-references only.
- **The Reduction:** Run the pipeline and produce the evidence to read.
- **Reading the Evidence:** Audit each stage from `reduction.json` and the public helpers.
- **Wrap Up:** Summary of the script and next steps.

__Imports__

Alongside `TargetSpec` and `reduce_target`, this script imports two public helpers it
demonstrates standalone: `select_observations` (the MAST query-hygiene filter used in the
discovery idiom below) and `casertano_r` (the correlated-noise factor, computed by hand to
check the pipeline's arithmetic).
"""
import json
from pathlib import Path

import numpy as np

from autoreduce import TargetSpec, reduce_target
from autoreduce.acquire.mast import select_observations
from autoreduce.noise.rms import casertano_r, empirical_background_rms

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the folder
containing `scripts/`). **PyAutoReduce** requires absolute paths: its drizzle step changes the
working directory internally, so relative paths would break.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"      # downloaded exposures + CRDS references (re-used across runs)
OUTPUT_ROOT = WORKSPACE / "output"    # reduced datasets, one folder per target

RA, DEC = 43.188375, 0.666222  # SDSS J0252+0039 — the UVIS anchor (see start_here.py).

"""
__Stage 1, Acquisition__

The acquire stage queries MAST (via `astroquery.mast`;
https://astroquery.readthedocs.io/en/latest/mast/mast.html) for the target's exposures and
downloads the calibrated products. The UVIS deltas versus ACS are small but load-bearing:

- The calibrated product is still the CTE-corrected `_flc` (see Stage 2), but the reference
  files that produced it live under the WFC3 environment key **`iref`** (ACS uses `jref`),
  and **PyAutoReduce** syncs them from CRDS (https://hst-crds.stsci.edu) into its cache under
  a `wfc3` subpath. CRDS maps each exposure to its best reference files; syncing them locally
  every run keeps the reduction reproducible (`sync_references=False` opts out for offline
  re-runs on a warm cache).
- Everything else — the per-target exposure cache with its manifest, the footprint filter,
  the query hygiene below — is identical to ACS (`hst_acs/step_by_step.py`, Stage 1).

The query hygiene deserves demonstration, because MAST will happily mislead a naive query. A
plain coordinate search also returns *Hubble Advanced Products* (HAP;
https://outerspace.stsci.edu/spaces/HAdP/pages/54558799/Improvements+in+HST+Astrometry) —
skycell mosaics and visit-level associations whose members re-reference the same exposures —
and reducing those alongside the direct exposures would ingest everything twice.
`select_observations` keeps only *direct* program observations (numeric proposal IDs, obs_id
not an `hst_*` HAP product).

Here is the live-discovery idiom: ask the archive which WFC3/UVIS filters actually cover
J0252+0039, before hard-coding one. Note the final subtlety — HAP composite rows carry the
pseudo-filter `"detection"`, which is not a filter you can point a reduction at, so it is
dropped explicitly.
"""
print("Querying MAST for WFC3/UVIS observations of J0252+0039 (network)...")

from astropy.coordinates import SkyCoord
from astroquery.mast import Observations

obs = Observations.query_criteria(
    coordinates=SkyCoord(RA, DEC, unit="deg"),  # The target position.
    radius="0.5 arcmin",                        # Match the acquire stage's default search radius.
    obs_collection="HST",                       # HST only (MAST also serves JWST, TESS, ...).
    instrument_name="WFC3/UVIS",                # The channel this folder is about.
    dataproduct_type="image",                   # Imaging only — no spectra.
)

direct = select_observations(obs)  # Keep only direct program observations (drop HAP products).

filters = sorted({str(row["filters"]) for row in direct} - {"detection"})

print(f"Direct WFC3/UVIS observations found; filters available: {filters}")

"""
The list should include F390W — the filter of the published Bayer et al.
(https://arxiv.org/abs/1803.05952) reduction, and the one we reduce below. In your own work
this is the moment to choose: **PyAutoReduce** reduces one filter per `TargetSpec`, so a
multi-band target is simply several specs.

__Stage 2, Calibration__

**PyAutoReduce** never runs the detector calibration itself — MAST serves products already
processed by the instrument pipeline with the current best reference files. For UVIS that
pipeline is **`calwf3`** (Python tooling and docs: https://wfc3tools.readthedocs.io), the
WFC3 counterpart of ACS's `calacs`. Its CCD chain is the same story you know from
`hst_acs/step_by_step.py` Stage 2 — DQ initialization, bias, dark, flat-field, photometric
keywords — with two UVIS-specific wrinkles:

- **`wf3cte`**: the pixel-based CTE correction runs as its own standalone-capable step inside
  `calwf3`, producing the `_flc` product from the trap model of Anderson & Bedin (2010, PASP
  122, 1035; https://ui.adsabs.harvard.edu/abs/2010PASP..122.1035A) as implemented for UVIS
  by Anderson et al. (2021, WFC3 ISR 2021-09). The WFC3 Data Handbook Chapter 6
  (https://hst-docs.stsci.edu/wfc3dhb) collects the references. One honest caveat from that
  literature: the correction amplifies read noise in low-S/N pixels — another reason the
  noise map is measured from the data, not assumed.
- **Post-flash** (the FLSHCORR step): UVIS observations routinely add an LED pre-exposure
  that raises the background to ~12-20 e- so faint charge survives readout past the CTE
  traps. It is subtracted like a dark, but its Poisson noise stays in the pixels — see
  Stage 5.

The output `_flc` exposures are in electrons, with an ERR extension (read noise + Poisson)
and a DQ extension flagging known-bad pixels. That is the input state for everything below.

__Stage 3, Alignment__

Identical to ACS: **PyAutoReduce** trusts the archive's Gaia-tied a-priori WCS solutions
rather than re-running TweakReg, and records a cross-correlation diagnostic per run so drift
would be caught. The full reasoning — what the archive's WCS solutions are, when re-alignment
would be warranted — is in `hst_acs/step_by_step.py` Stage 3; nothing changes for UVIS.

__Stage 4, Drizzle__

The combination stage is byte-for-byte the ACS machinery: AstroDrizzle
(https://hst-docs.stsci.edu/drizzpac) with sky matching, median-stack cosmic-ray flagging
(`driz_cr`, DQ bit 4096) and IVM weighting, drizzling the `_flc` frames onto a north-up
output grid in e-/s. The UVIS deltas are numerical:

- The native scale is **0.0396"/pixel**, and the adapter recommends output at native scale —
  the Bayer et al. anchor dials (0.0396, pixfrac 1.0) rather than the ACS/SLACS convention
  (0.05, pixfrac 0.8). The trade study between these choices — resolution versus noise
  correlation versus coverage — is `hst_acs/dials.py`.
- The full well saturates near **63 ke-** (ACS: ~80 ke-). This threshold feeds the PSF star
  selection: candidate stars with peaks above a fraction of saturation are rejected, so the
  brightest (and most tempting) stars in the field do not corrupt the PSF with bleeding.

__Stage 5, Noise__

The recipe is shared with ACS (`guides/noise_maps.py` derives it in full):

    sigma_i = R * sqrt(N_i / t_exp + 1 / W_i)

where N_i is the source count rate, W_i the IVM drizzle weight (inverse background variance)
and R the Casertano et al. (2000, AJ 120, 2747) correlated-noise factor for the chosen
pixfrac and scale. The UVIS delta lives inside W_i: the background variance that the weight
encodes includes the *post-flash* Poisson contribution alongside sky, dark and read noise.
A UVIS frame with a 15 e- post-flash is intrinsically noisier than its exposure time
suggests — the price of surviving readout — and because the weights are built from the real
per-frame backgrounds, the noise map inherits this automatically. No hand-tuning, but worth
knowing when comparing depth between UVIS and ACS programs.

__Stage 6, PSF and Packaging__

Both stages are identical to ACS and are covered there: PSF tiers and star selection in
`hst_acs/psf.py` (with the UVIS-specific star-rich walkthrough in `hst_wfc3_uvis/psf.py`),
the packaging contract — strict-mode cutout, masked-by-noise policy, header preservation —
in `hst_acs/step_by_step.py` Stage 6 and `guides/output_contract.py`.

__The Reduction__

Now run the pipeline with the anchor dials. The spec is identical to `start_here.py` —
including the `name` — so if you ran that script first, the exposures and CRDS references are
already cached and this run skips the downloads entirely.
"""
spec = TargetSpec(
    name="j0252+0039_f390w",  # Same name as start_here.py -> same cache + output folder.
    ra=RA,                    # Target right ascension in degrees.
    dec=DEC,                  # Target declination in degrees.
    instrument="wfc3_uvis",   # UVIS adapter: _flc, iref, native 0.0396"/pix, sat 63 ke-.
    filter_name="F390W",      # Chosen from the live-discovered filter list above.
    final_scale=0.0396,       # Native-scale output (Bayer anchor dial).
    final_pixfrac=1.0,        # Full drop: simplest noise correlation (Bayer anchor dial).
)

print(
    "Running the J0252+0039 F390W reduction (cached exposures re-used if "
    "start_here.py ran first; otherwise the first run downloads from MAST "
    "and takes tens of minutes)..."
)

record = reduce_target(spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)

out_dir = OUTPUT_ROOT / spec.name

"""
__Reading the Evidence__

The stages above are internal to `reduce_target` — you cannot call them one at a time, by
design (a reduction is one declared, reproducible unit). What you *can* do is audit every
stage from the provenance it leaves behind. `reduction.json` carries one block per stage;
here we walk the UVIS-relevant evidence.

First, acquisition: how many exposures, from which proposal, through which instrument
adapter. Cross-check against the filter discovery above.
"""
provenance = json.loads((out_dir / "reduction.json").read_text())

print(f"instrument block : {json.dumps(provenance['instrument'], indent=2)}")
print(f"n_exposures      : {provenance['acquire']['n_exposures']}")

"""
Second, the drizzle block. `weight_uniformity` is the STScI rule-of-thumb statistic
(RMS/median of the weight map, acceptable below ~0.2); with pixfrac 1.0 on a dithered
program it should pass with room to spare. The block also records the CR method and the
dials actually used — the audit trail for every number in this script.
"""
print(f"drizzle block    : {json.dumps(provenance['drizzle'], indent=2)}")

"""
Third, the noise block — and a by-hand check of its arithmetic. The recorded
`correlated_noise_factor` should equal the Casertano R for our dials: pixfrac p = 1.0 at
scale ratio s = 1.0 (output scale / native scale = 0.0396 / 0.0396), which the public
`casertano_r` helper computes directly. At p = 1, s = 1 the variance-reduction factor is
r = 1 - p/(3s) = 2/3, so R = 1/r = 1.5.
"""
r_by_hand = casertano_r(
    pixfrac=1.0,      # The final_pixfrac dial.
    scale_ratio=1.0,  # final_scale / native scale = 0.0396 / 0.0396.
)

print(f"casertano_r(p=1.0, s=1.0) = {r_by_hand:.4f}")
print(f"recorded in provenance    = {provenance['noise']['correlated_noise_factor']:.4f}")

"""
The noise block also records `empirical_background_rms` — the sigma-clipped sky RMS measured
from the mosaic, the number validated against the published sigma_sky ~ 0.002 e-/s in
`start_here.py`. The helper behind it is public too, so you can reproduce it on the packaged
cutout (expect a value close to, though not identical to, the recorded one — the record
measures the full mosaic, this measures the cutout):
"""
from astropy.io import fits

data = fits.getdata(out_dir / "data.fits").astype(float)

sky_rms_cutout = empirical_background_rms(data)

print(f"empirical sky RMS (cutout)   : {sky_rms_cutout:.5f} e-/s")
print(f"empirical sky RMS (recorded) : {provenance['noise']['empirical_background_rms']:.5f} e-/s")

"""
Finally, the PSF block: which tier built the PSF, from how many stars, and — UVIS-relevant —
which drizzle pass the stars came from (`star_source_pass`). The `psf_star_pass="no_cr"`
second-pass option, which recovers stars that `driz_cr` clipped, is an ACS-shared dial
covered in `hst_acs/psf.py`.
"""
print(f"psf block        : {json.dumps(provenance['psf'], indent=2)}")

"""
And the package block, which seals the contract: the pixel scale a **PyAutoLens** load
consumes, the data units, and the product list. If you ever wonder whether a dataset on disk
was reduced the way you think it was, this file — not your memory of the run — is the
answer.
"""
print(f"package block    : {json.dumps(provenance['package'], indent=2)}")

"""
__Wrap Up__

You have seen every stage of the UVIS reduction and what distinguishes it from ACS: `calwf3`
and its standalone `wf3cte` CTE step, `iref` references, the post-flash term riding inside
the drizzle weights, the native 0.0396"/pixel scale and the 63 ke- saturation ceiling — plus
the live MAST discovery idiom and the query hygiene that keeps HAP products out of the
stack. Everything else is shared machinery, documented once on the ACS side.

The following locations of the workspace are good places to checkout next:

- `scripts/hst_acs/step_by_step.py`: the full stage-by-stage reference this script deltas
  against.
- `scripts/hst_acs/dials.py`: the drizzle dial trade study (scale, pixfrac, kernel, CR
  method).
- `scripts/hst_wfc3_uvis/psf.py`: the UVIS PSF walkthrough on a genuinely star-rich field.
- `scripts/hst_wfc3_ir/step_by_step.py`: the IR channel's stages — a different detector
  physics entirely.
- `scripts/guides/noise_maps.py`: the noise recipe and the Casertano R in full.

__Env__ (Developer Only)

ENV: network
"""
