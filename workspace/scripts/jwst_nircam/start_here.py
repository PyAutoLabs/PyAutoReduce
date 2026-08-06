"""
Start Here: JWST NIRCam
=======================

JWST/NIRCam is, right now, the most powerful imaging instrument in existence for strong
gravitational lens science: diffraction-limited resolution beyond 2 microns, extraordinary
depth in minutes of exposure time, and continuous wavelength coverage from 0.6 to 5 microns
across its short-wavelength (SW) and long-wavelength (LW) channels.

This script uses **PyAutoReduce** to reduce NIRCam imaging of the COSMOS-Web ring — a
spectacular Einstein ring discovered *during the data reduction* of the COSMOS-Web survey —
from raw archive products into a modeling-ready dataset: `data.fits`, `noise_map.fits`,
`psf.fits`, `psf_full.fits` and a full `reduction.json` provenance record. This is the
exact product set that `autolens_workspace/scripts/imaging/start_here.py` loads to model
this very lens.

Everything runs through two public names — `TargetSpec` and `reduce_target` — so the
script is short on code and long on explanation. The first run downloads the level-2
exposures from MAST and runs the official `jwst` calwebb_image3 pipeline, which takes
tens of minutes; reruns re-use the cache and are much faster.

__Contents__

- **The COSMOS-Web Ring:** The lens we reduce, and why it is the perfect JWST anchor.
- **Imports:** Import **PyAutoReduce** and the supporting libraries.
- **Paths:** Anchor the cache and output folders to the workspace root.
- **Instrument Adapter:** Route the F277W filter to the NIRCam LW adapter.
- **Target Spec:** Declare the reduction — every dial explained.
- **Archive Products and Units:** What a `_cal` file is and why its units are MJy/sr.
- **CRDS References:** The reference-file system and how to pin a context for reproducibility.
- **The Reduction:** One function call: MAST download, calwebb_image3, noise, PSF, package.
- **Provenance Walk:** Read the evidence out of the returned reduction record.
- **Noise — Read, Don't Construct:** The resampled ERR array, the Casertano R factor and the blank-sky closure.
- **Parity Stance:** Why "close + internally consistent" is the honest bar against the bespoke COSMOS-Web team pipeline.
- **PSF:** An empirical ePSF from mosaic field stars, and why surface-brightness units change star selection.
- **Plots:** Inspect the data, noise map and PSF.
- **Loading in PyAutoLens:** Load the products with `al.Imaging.from_fits`, with the MJy/sr caveat spelled out.
- **Wrap Up:** Where to go next.

__The COSMOS-Web Ring__

COSMOS-Web (Casey et al. 2023) is the largest JWST Cycle 1 GO program: 0.54 square degrees
of NIRCam imaging in four bands (F115W, F150W, F277W, F444W) over the COSMOS field. While
the team was reducing the very first epoch of data, a complete Einstein ring appeared in
the mosaics around a massive elliptical galaxy — the "COSMOS-Web ring"
(Mercier et al. 2024, A&A 687, A61, https://arxiv.org/abs/2309.15986). It is one of the
most striking strong lenses JWST has observed, alongside "JWST-ER1"
(van Dokkum et al. 2024, Nature Astronomy), whose source lies at redshift 5.1.

The ring is also the anchor of a much bigger science programme: COWLS, the COSMOS-Web Lens
Survey, has published over 100 high-confidence lens candidates with **PyAutoLens**
reconstructions in all four bands (COWLS I: Nightingale et al. 2025, MNRAS 543, 203,
https://arxiv.org/abs/2503.08777; COWLS II: https://arxiv.org/abs/2503.08782; COWLS III:
https://arxiv.org/abs/2503.08785). Every one of those models consumed a reduced dataset
shaped exactly like the one this script produces — which is why the ring is
**PyAutoReduce**'s JWST validation anchor.

__Imports__

**PyAutoReduce** exposes exactly two names — `TargetSpec` (a frozen dataclass declaring
*what* to reduce) and `reduce_target` (the function that does it). NIRCam adds one routing
helper, `nircam_adapter_for_filter`, which maps a filter name to the right channel adapter.
Everything else here is the standard scientific Python stack.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

from autoreduce import TargetSpec, reduce_target
from autoreduce.instruments import nircam_adapter_for_filter

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the
folder containing `scripts/`). **PyAutoReduce** requires absolute paths: its combine step
changes the working directory internally, so relative paths would break.

The cache folder holds the downloaded exposures and the CRDS reference files — both are
re-used across runs, so a second reduction of the same target (or a different band of the
same target) skips the download entirely.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"  # downloaded exposures + CRDS references (re-used across runs)
OUTPUT_ROOT = WORKSPACE / "output"  # reduced datasets, one folder per target

"""
__Instrument Adapter__

NIRCam observes through two channels simultaneously via a dichroic: the short-wavelength
(SW) channel (0.6–2.3 microns, eight 2048x2048 H2RG detectors at 0.031"/pixel) and the
long-wavelength (LW) channel (2.4–5.0 microns, two detectors at 0.063"/pixel). See the
JDox NIRCam pages (https://jwst-docs.stsci.edu) for the instrument overview.

**PyAutoReduce** encodes each channel as an *instrument adapter* — a frozen bundle of
archive, calibration and combination facts (which MAST products to fetch, which CRDS
server to sync from, which combine backend to run, the native and recommended output
pixel scales). You never pick the channel yourself: `nircam_adapter_for_filter` routes
any NIRCam filter name to the right adapter, and raises loudly on a filter it does not
know.

F277W is an LW filter, so we get the `nircam_lw` adapter: native 0.063"/pixel, with a
recommended output scale of 0.06"/pixel — the COSMOS-Web mosaic convention.
"""
band = "F277W"

adapter = nircam_adapter_for_filter(band)

print(f"Adapter key: {adapter.key}")  # nircam_lw
print(f"Native pixel scale: {adapter.native_scale}\"/pixel")  # 0.063
print(f"Recommended output scale: {adapter.recommended_final_scale}\"/pixel")  # 0.06
print(f"Combine backend: {adapter.combine_backend}")  # jwst_image3

"""
__Target Spec__

A **PyAutoReduce** reduction is *declared*, not scripted: you build a single frozen
`TargetSpec` and every downstream stage is a pure function of it plus the archive. The
same spec always produces the same dataset — that is what makes the reduction
reproducible and its provenance meaningful.

Each non-default dial is explained by its trailing comment. Two deserve emphasis:

- `proposal_ids=("1727",)` restricts the archive query to the COSMOS-Web program. Other
  JWST programs have since observed these coordinates; mixing their exposures in would
  change the depth and the noise properties of the mosaic in ways the COSMOS-Web
  literature comparisons could not track.

- `final_pixfrac=1.0` uses the full drizzle "drop" — the COSMOS-Web mosaic convention.
  Shrinking the drop reduces noise correlation but demands more dither coverage; with
  the ring's dither pattern the full drop is the safe, convention-matching choice.
"""
RA, DEC = 150.10048, 1.89301  # the COSMOS-Web ring (Mercier et al. 2024)

spec = TargetSpec(
    name=f"cosmos_web_ring_{band.lower()}",  # names the output folder: output/cosmos_web_ring_f277w/
    ra=RA,  # target right ascension in degrees (J2000)
    dec=DEC,  # target declination in degrees (J2000)
    instrument=adapter.key,  # "nircam_lw" — the adapter routed from the filter above
    filter_name=band,  # the NIRCam filter to reduce
    proposal_ids=("1727",),  # COSMOS-Web only: other programs at these coords would change the depth
    final_scale=adapter.recommended_final_scale,  # 0.06"/pixel output grid — the COSMOS-Web LW convention
    final_pixfrac=1.0,  # full drizzle drop, matching the COSMOS-Web mosaics
    cutout_shape=(209, 209),  # 209 px x 0.06" ~ 12.5" — generous margin around the ~2" ring
)

"""
__Archive Products and Units__

**PyAutoReduce** enters the JWST calibration chain at level 2: it downloads the `_cal`
products of the `calwebb_image2` pipeline from MAST. These are individual exposures that
have already been through detector-level corrections (stage 1) and per-exposure
calibration (stage 2) at STScI — flat-fielded, WCS-assigned and flux-calibrated. The
stage-by-stage story of what those pipelines do lives in `step_by_step.py`; the short
version is that stages 1 and 2 are pure STScI defaults, and **PyAutoReduce** only takes
over at the combination stage where the lensing-specific choices live.

One consequence matters immediately: the `photom` step of calwebb_image2 converts every
`_cal` file to **megajanskys per steradian (MJy/sr)** — a *surface brightness*, not a
flux. HST reductions ship in electrons per second (a count rate per pixel); JWST products
instead tell you the brightness per unit solid angle, independent of pixel size.
**PyAutoReduce** follows its defaults-first principle and keeps the native MJy/sr units
all the way into `data.fits` (with `BUNIT` recorded in the header).

To recover a flux from these units you multiply by the pixel solid angle, which the JWST
headers carry as `PIXAR_SR` (steradians per pixel):

    flux [Jy] = surface_brightness [MJy/sr] x 1e6 x PIXAR_SR

and AB magnitudes follow as m_AB = -2.5 log10(flux / 3631 Jy). We return to this when
loading the dataset into **PyAutoLens** below, because it changes how you interpret
every fitted intensity.

__CRDS References__

JWST calibration reference files come from the CRDS system (https://jwst-crds.stsci.edu).
Unlike the HST path — where **PyAutoReduce** syncs best references explicitly before
running — the `jwst` pipeline fetches what it needs *lazily* through the `CRDS_PATH`
environment variable, which **PyAutoReduce** points into the cache folder. The references
persist there across runs and are never evicted. For byte-for-byte reproducibility you
can additionally pin `CRDS_CONTEXT` to a fixed context; the software versions actually
used are recorded in `reduction.json` either way.

__The Reduction__

Everything happens in the single call below. Internally it runs, in order:

1. **acquire** — query MAST for COSMOS-Web `_cal` exposures overlapping the target
   footprint in F277W, download them into the cache.
2. **combine** — build a level-3 association and run the official `jwst`
   calwebb_image3 pipeline (tweakreg / skymatch / outlier_detection / resample), with
   the lensing dials mapped onto the resample step.
3. **noise** — read the resampled ERR array and apply the correlated-noise correction.
4. **psf** — find stars on the mosaic and build an empirical ePSF.
5. **package** — cut out the target region, apply the bad-pixel policy, write the four
   FITS products plus `reduction.json`.
"""
print(
    f"""
    Starting the {band} reduction of the COSMOS-Web ring.

    First run: this downloads the program-1727 _cal exposures from MAST (a few GB) and
    runs the jwst calwebb_image3 pipeline — expect tens of minutes, and a large CRDS
    reference download on a cold cache. Reruns re-use the exposure cache and are much
    faster.
    """
)

record = reduce_target(spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)

out_dir = OUTPUT_ROOT / spec.name

print(f"Reduction complete. Products written to: {out_dir}")

"""
__Provenance Walk__

`reduce_target` returns the full provenance record (the same content written to
`reduction.json`). Nothing about the reduction is hidden: every stage writes a block
recording what it did and the diagnostics it measured. Walking this record after every
reduction is a habit worth building — it is how you catch a marginal reduction before
you spend GPU-hours modeling it.

The `drizzle` block records the combine backend and the exact resample keywords that
were mapped from your spec, and two diagnostics worth reading every time:

- `weight_uniformity` — the RMS/median of the drizzle weight map. A uniform weight map
  means every output pixel saw comparable total exposure; values creeping toward the
  0.2 policy limit flag ragged dither coverage.
- `weight_uniformity_cutout` — the same statistic restricted to the shipped cutout,
  which is the number that actually matters for the lens.
"""
print("Backend:", record["drizzle"]["backend"])  # "jwst_image3"
print("Resample kwargs:", json.dumps(record["drizzle"]["resample_kwargs"], indent=2))
print("Exposures combined:", record["acquire"]["n_exposures"])
print("Weight uniformity (mosaic):", record["drizzle"]["weight_uniformity"])
print("Weight uniformity (cutout):", record["drizzle"]["weight_uniformity_cutout"])

"""
The `psf` block records how the PSF was built (method, number of stars used, the FWHM in
output pixels and which mosaic pass fed the star finder), and the `package` block records
the pixel scale and data units your modeling code needs.
"""
print("PSF:", json.dumps(record["psf"], indent=2))
print("Pixel scale:", record["package"]["pixel_scale"])
print("Data units:", record["package"]["data_units"])  # MJy/sr

"""
__Noise — Read, Don't Construct__

On HST, **PyAutoReduce** *constructs* the noise map from first principles: Poisson counts
over exposure time plus the inverse of the drizzle weight. On JWST it deliberately does
the opposite — it **reads** the noise, because the `jwst` pipeline already propagates a
full per-pixel error budget. Each `_cal` exposure carries `VAR_POISSON`, `VAR_RNOISE`
and `VAR_FLAT` variance planes; the resample step drizzles each plane separately onto
the output grid and recombines them into the mosaic `ERR` array. Re-deriving that budget
from scratch would discard information the pipeline already has (per-pixel read noise,
flat-field uncertainty, ramp-fit weighting).

One correction is still required. Resampling correlates neighbouring output pixels —
each input pixel's flux is shared among several output pixels — so the *per-pixel* ERR
values underestimate the uncertainty of any quantity summed over more than one pixel
(Fruchter & Hook 2002, PASP 114, 144). **PyAutoReduce** applies the standard scalar
correction factor R of Casertano et al. (2000, AJ 120, 2747), computed from the same
pixfrac and scale ratio the resample step used:

    noise_map = R x ERR

This is exactly the factor the HST path applies, because resample correlates pixels
exactly as drizzle does. R is recorded as `correlated_noise_factor`.

The `noise` block then closes the loop with an internal consistency check: it compares
the sigma-clipped RMS of blank sky in the mosaic against the 5th percentile of the ERR
array (the ERR "floor", pre-R). If the pipeline's error model and the actual sky
fluctuations agree, `sky_over_err_floor` sits near 1. A large disagreement means the
upstream error model and the data disagree — something to investigate, never to absorb
silently.
"""
print("Noise recipe:", record["noise"]["recipe"])
print("Correlated-noise factor R:", record["noise"]["correlated_noise_factor"])
print("Empirical sky RMS:", record["noise"]["empirical_sky_rms"])
print("ERR 5th percentile (pre-R):", record["noise"]["err_5th_percentile_pre_R"])
print("sky / ERR floor:", record["noise"]["sky_over_err_floor"])

"""
__Parity Stance__

An honest caveat, stated up front because it frames how to interpret this dataset.

The published COSMOS-Web mosaics were *not* made by running calwebb with defaults. The
team's reduction (Franco et al., https://arxiv.org/abs/2506.03256) layers survey-grade
corrections on top of the official pipeline: custom 1/f-noise destriping, wisp
subtraction, snowball handling, a custom background model and astrometry tied to a
Gaia-registered COSMOS reference frame. Every major deep survey does something similar
(CEERS: Bagley et al. 2023, https://arxiv.org/abs/2211.02495; JADES: Rieke et al. 2023,
https://arxiv.org/abs/2306.02466), because wide-area depth uniformity demands it.

**PyAutoReduce** deliberately does not chase those corrections. Its bar for JWST is
**"close + internally consistent," not reproduction**: strong lens modeling needs a
reduction whose noise map matches its data, whose weights are uniform over the cutout,
and whose masked-pixel policy is explicit — all properties this pipeline checks itself,
on your cutout, every run. Order-unity data/noise ratios against the team mosaics are
expected and acceptable; what must hold are the internal closures you just read
(`sky_over_err_floor`, weight uniformity) and cross-band consistency of any global
offset (which `multi_band.py` demonstrates). At lens-cutout scale (~12"), the survey-wide
corrections mostly matter through slightly elevated large-scale noise — which the
noise-map closure would reveal if it mattered.

__PSF__

The PSF ships in two forms: `psf.fits` (21x21 pixels) sized for convolution during
model fitting, and `psf_full.fits` (61x61) capturing the extended wings for
flux-sensitive work. Both are odd-shaped, unit-normalised, and — crucially — built
*from the mosaic itself*, so the delivered PSF has been through the identical
resampling (same pixel scale, pixfrac, kernel and rotation) as the data it will be
convolved with. A PSF that has not shared the data's resampling history is subtly
wrong everywhere.

The builder is the empirical "effective PSF" (ePSF) estimator of Anderson & King
(2000, PASP 112, 1360) as implemented in photutils, fed by stars found on the mosaic.
One JWST-specific detail: on HST, star selection rejects stars whose peaks approach
the detector full well, because their cores are non-linear. In MJy/sr units a peak
threshold in counts is meaningless — and saturated cores arrive from level 2 already
blanked to NaN — so the JWST star finder is NaN-masked and applies **no peak cut**.
The full JWST PSF story (undersampling, STARRED, model PSFs) lives in `psf.py`.

__Plots__

Now inspect what was produced. We use an arcsinh stretch for the data — linear near
zero so the sky noise is visible, logarithmic on the bright lens galaxy — and a linear
stretch for the noise map, where structure (depth variations, masked pixels at 1e8)
is what you are looking for.
"""
plot_dir = out_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

data = fits.getdata(out_dir / "data.fits").astype(float)
noise = fits.getdata(out_dir / "noise_map.fits").astype(float)
psf = fits.getdata(out_dir / "psf.fits").astype(float)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

scale = np.nanpercentile(data, 99.5)
axes[0].imshow(np.arcsinh(data / (0.05 * scale)), origin="lower", cmap="magma")
axes[0].set_title(f"data.fits ({band}, MJy/sr, arcsinh)")

axes[1].imshow(np.where(noise > 1.0e7, np.nan, noise), origin="lower", cmap="viridis")
axes[1].set_title("noise_map.fits (masked px blanked)")

axes[2].imshow(np.arcsinh(psf / psf.max() * 1000.0), origin="lower", cmap="magma")
axes[2].set_title("psf.fits (21x21, arcsinh)")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plot_path = plot_dir / "start_here_products.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)

print(f"Product plot saved to: {plot_path.resolve()}")

"""
In the data panel you should see the lens galaxy at the centre with the Einstein ring
wrapped around it — the same morphology as Figure 1 of Mercier et al. 2024. In the noise
panel, depth structure follows the dither coverage; any isolated blanked pixels are the
masked-by-noise policy at work (bad pixels carry noise 1e8 and zeroed data, so a chi^2
simply ignores them — no separate mask file needed downstream).

__Loading in PyAutoLens__

The products are designed to drop straight into `al.Imaging.from_fits`. The pixel scale
comes from the provenance record, never from memory.

**The MJy/sr caveat, spelled out.** **PyAutoLens** is unit-agnostic: it fits whatever
units the data arrives in, and the model's `intensity` parameters inherit those units.
With this dataset:

- Fitted intensities are in **MJy/sr** — surface brightnesses, not counts. This is
  actually natural for galaxy profiles (a Sersic is a surface-brightness profile), but
  it means you cannot compare intensity values against an HST fit in e-/s without
  converting.
- To quote a **flux**, integrate the model image and multiply by the pixel solid angle:
  `flux_jy = model_image.sum() * (pixel_scale / 206265.0)**2 * 1e6`, then
  m_AB = -2.5 log10(flux_jy / 3631).
- The noise map is in the same units, so signal-to-noise and chi^2 are unit-free — the
  *fit* is unaffected. Only the physical interpretation of intensities needs care.
"""
try:
    import autolens as al
except ImportError:
    al = None
    print(
        "PyAutoLens is not installed (pip install autolens), so the final loading step "
        "is skipped — the reduction itself is complete and the products above are "
        "ready for any modeling tool."
    )

if al is not None:
    dataset = al.Imaging.from_fits(
        data_path=out_dir / "data.fits",
        noise_map_path=out_dir / "noise_map.fits",
        psf_path=out_dir / "psf.fits",
        pixel_scales=record["package"]["pixel_scale"],
    )
    print(
        f"Loaded al.Imaging: shape {dataset.data.shape_native}, "
        f"pixel scale {dataset.pixel_scales} — ready for lens modeling."
    )

"""
__Wrap Up__

You have reduced JWST/NIRCam imaging of the COSMOS-Web ring from MAST archive products
to a modeling-ready dataset, and audited the reduction through its provenance record:
the calwebb_image3 backend and its resample dials, weight uniformity, the read-don't-
construct noise map with its Casertano R factor and blank-sky closure, and the mosaic
ePSF.

The following locations of the workspace are good places to checkout next:

- `scripts/jwst_nircam/step_by_step.py`: what Detector1, Image2 and Image3 actually do to the data, stage by stage, with the pipeline documentation.
- `scripts/jwst_nircam/multi_band.py`: reduce all four COSMOS-Web bands into a multi-wavelength dataset.
- `scripts/jwst_nircam/psf.py`: the JWST PSF story in depth — undersampling, ePSF lineage, STARRED vs photutils.
- `scripts/jwst_nircam/individual.py`: per-exposure frame products instead of a mosaic.
- `scripts/jwst_nircam/simulator.py`: inject a synthetic source into the real exposures and test flux recovery.
- `scripts/guides/output_contract.py`: the four-file + reduction.json contract in full detail.

__Env__ (Developer Only)

ENV: network
"""
