"""
JWST NIRCam: PSF
================

No single ingredient of a lens dataset repays care like the PSF. Strong lens modeling
convolves every trial image with it, so PSF errors masquerade as source structure, bias
the inferred mass model, and — for the quasar and substructure work JWST excels at — can
dominate the entire error budget. This script is the JWST PSF story in **PyAutoReduce**:
how the PSF is measured, when each backend wins, and which diagnostics tell you whether
to trust what you got.

The anchor here is not a lens but a globular cluster: M92 (NGC 6341), observed by
program 1334 — the JWST NIRCam astrometric and flux calibration field. A genuinely
stellar, deep field is the right place to *validate* PSF estimation, because it offers
hundreds of isolated point sources where a lens field offers a handful. We reduce it in
F150W (SW — the undersampled regime) and F277W (LW — well-sampled), with both PSF
backends, and read the verdicts out of the diagnostics.

__Contents__

- **Why the PSF Matters for Lensing:** Convolution errors become science errors.
- **Imports:** Import **PyAutoReduce** and the supporting libraries.
- **Paths:** Anchor the cache and output folders to the workspace root.
- **Undersampling:** The SW channel's defining PSF problem, and why 2 microns is the dividing line.
- **The ePSF Lineage:** Anderson & King's effective PSF, its arrival on NIRCam, and library ePSFs.
- **The M92 Reductions:** Reduce the calibration field in F150W and F277W with the photutils Tier-1 ePSF.
- **Diagnostics Walk:** method, star counts, FWHM, star source pass — the record every PSF ships with.
- **psf vs psf_full and the Drizzled-PSF Invariant:** Two kernels, one resampling history.
- **STARRED — Tier 1b:** The super-sampled alternative, and the M92 regime rule with its undersampled flag.
- **Model PSFs — STPSF as Tier 2b:** Why pure model PSFs are disfavoured, and where **PyAutoReduce** still uses them.
- **Spatially-Varying PSFs — the Roadmap:** PSFEx/ShOpt-style Tier 2 is planned, not implemented.
- **Plots:** The measured kernels, compact and full, in both bands.
- **Wrap Up:** Where to go next.

__Why the PSF Matters for Lensing__

A lens model predicts the sky, but you observe the sky *convolved with the PSF*. During
fitting the model image is convolved with `psf.fits` before comparison with the data, so:

- An underestimated PSF width leaves compact residuals at the arcs that a flexible
  source model will happily absorb as spurious structure.
- PSF wing errors leak deflector light into the arc region, biasing the source
  reconstruction and — through the source, the mass model.
- For lensed quasars and substructure searches, percent-level PSF errors are the
  systematics floor.

This is why **PyAutoReduce** treats the PSF as a first-class product with its own
diagnostics block, not a footnote.

__Imports__
"""
from dataclasses import replace
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
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"  # downloaded exposures + CRDS references (re-used across runs)
OUTPUT_ROOT = WORKSPACE / "output"  # reduced datasets, one folder per target

"""
__Undersampling__

The JWST optical PSF shrinks with wavelength (diffraction), but the NIRCam pixels do
not. The result, documented on the JDox NIRCam imaging pages
(https://jwst-docs.stsci.edu): beyond ~2 microns the telescope delivers a
diffraction-limited, well-sampled PSF (Strehl ~0.8) on the LW channel's 0.063" pixels —
but *below* ~2 microns, on the SW channel's 0.031" pixels, the PSF FWHM spans less than
two native pixels. The image on the detector no longer Nyquist-samples the PSF.

Undersampling is the defining problem of SW PSF work. A star's measured shape depends
on where its centre lands within a pixel; naive stacking of stars blurs them by their
subpixel phases; and a mosaic only partially recovers the lost information through the
subpixel dither pattern (this is exactly why STScI designed those patterns, and why
`resample` onto a finer 0.03" grid helps but cannot fully cure it). Every method below
is ultimately a strategy for coping with this fact.

__The ePSF Lineage__

The standard answer is the *effective PSF* (ePSF) of Anderson & King (2000, PASP 112,
1360): model the PSF as seen *by the pixel grid* — the optical PSF convolved with the
pixel response — on an oversampled grid, built iteratively from many stars at different
subpixel phases. The subpixel dithers that sample the star field become the very thing
that beats undersampling.

On JWST this lineage arrived immediately: Nardiello et al. (2022, MNRAS 517, 484) built
the first NIRCam ePSFs — 5x5 grids of 4x-oversampled library ePSFs per filter and
detector — and the approach extended to NIRISS (Libralato et al. 2023) and MIRI
(Libralato et al. 2024), alongside Anderson's STDPSF library ePSFs for NIRCam. The
existence of *library* ePSFs matters for star-poor fields; **PyAutoReduce**'s Tier 1
instead builds the ePSF from the mosaic's own stars via the photutils implementation of
the same algorithm, because a PSF measured on your mosaic has automatically been through
your mosaic's resampling — the invariant discussed below.

One JWST-specific wrinkle in star selection: HST star finding rejects stars near the
full well, but a peak cut in counts is meaningless in MJy/sr surface-brightness units —
and saturated cores arrive from level 2 already blanked. The JWST star finder is
therefore NaN-masked and applies **no peak cut**.

__The M92 Reductions__

Now reduce the calibration field in both channels with the default photutils Tier-1
backend. The 501x501 cutout (~15" in both channels) spans M92's density gradient,
giving the star finder a rich sample.
"""
RA, DEC = 259.28079, 43.13594  # M92 (NGC 6341) cluster centre

PSF_BANDS = ("F150W", "F277W")  # SW undersampled / LW well-sampled — the two regimes


def m92_spec(band: str, psf_backend: str = "epsf") -> TargetSpec:
    adapter = nircam_adapter_for_filter(band)
    return TargetSpec(
        name=f"m92_{band.lower()}",  # one output folder per band
        ra=RA,  # M92 cluster centre, right ascension in degrees
        dec=DEC,  # M92 cluster centre, declination in degrees
        instrument=adapter.key,  # "nircam_sw" (F150W) or "nircam_lw" (F277W)
        filter_name=band,  # the NIRCam filter to reduce
        proposal_ids=("1334",),  # the JWST NIRCam astrometric/flux calibration program
        final_scale=adapter.recommended_final_scale,  # SW 0.03" / LW 0.06"
        final_pixfrac=1.0,  # full drizzle drop
        cutout_shape=(501, 501),  # ~15" — a rich stellar sample across the density gradient
        psf_backend=psf_backend,  # "epsf" (photutils Tier 1) | "starred" (Tier 1b)
    )


records = {}

for band in PSF_BANDS:
    spec = m92_spec(band)
    print(
        f"""
    [{band}] reducing M92 (program 1334, {spec.instrument}). First run downloads the
    _cal exposures from MAST and runs calwebb_image3 — expect tens of minutes per band;
    reruns re-use the cache.
    """
    )
    records[band] = reduce_target(spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)
    print(f"[{band}] done -> {OUTPUT_ROOT / spec.name}")

"""
__Diagnostics Walk__

Every PSF ships with a diagnostics block in `reduction.json` — the `psf` block — so no
PSF is ever a bare array you must take on faith. The keys to read every time:

- `method` — which tier produced the kernel ("epsf-tier1" here).
- `n_stars_used` — how many stars survived selection and fed the fit. The Tier-1
  builder requires at least 8 and fails loudly below that; on M92 expect far more.
- `fwhm_pix` — the fitted kernel's FWHM in *output* pixels. Compare against the
  expectation for the band: F150W's PSF is ~1.7 native pixels, so an undersampled ~1.7
  on the 0.03" grid; F277W is comfortably above 2.
- `star_source_pass` — which mosaic pass fed the star finder. On JWST this is always
  the science mosaic (the CR story lives in the ramps, so no CR-avoiding second
  drizzle is ever needed — the reason is recorded rather than assumed).
"""
for band in PSF_BANDS:
    print(f"[{band}] psf block:")
    print(json.dumps(records[band]["psf"], indent=2))

"""
__psf vs psf_full and the Drizzled-PSF Invariant__

Each reduction ships two kernels:

- `psf.fits` (21x21 by default) — compact, sized for the convolution inside model
  fitting, where kernel size costs runtime on every likelihood evaluation.
- `psf_full.fits` (61x61) — the extended wings, for flux-sensitive work (aperture
  corrections, quasar deblending, checking how much deflector light the compact kernel
  ignores).

Both are odd-shaped and unit-normalised, and both obey the **drizzled-PSF invariant**:
the delivered PSF is measured *from the mosaic*, so it has passed through the identical
resampling — same pixel scale, pixfrac, kernel, rotation — as the data. A PSF from any
other source (a model PSF evaluated on a detector grid, a library ePSF at native
sampling) must be pushed through the same resampling before it is comparable; skipping
that step is one of the classic silent PSF errors in lens modeling.

__STARRED — Tier 1b__

**PyAutoReduce** offers a second empirical backend: STARRED (Millon, Michalewicz et al.
2024, AJ, https://arxiv.org/abs/2402.08725), a JAX-based, wavelet-regularised joint PSF
reconstruction developed in the COSMOGRAIL lensed-quasar tradition and demonstrated on
JWST imaging. It fits all stars simultaneously for a super-sampled PSF with a
starlet-regularised residual channel — a higher-fidelity alternative to the photutils
ePSF for demanding quasar/AGN-grade work. It is optional (GPL-licensed, JAX-dependent):
install the `starred` extra to enable it, and if it is missing the backend raises
loudly rather than silently falling back.

The M92 validation is exactly why this field is in the workspace, and it produced a
clean **regime rule**:

- **F277W (LW, well-sampled): STARRED wins.** With the PSF resolved by the pixel grid,
  the super-sampled joint fit extracts more information than the classic ePSF.
- **F150W (SW, undersampled): STARRED loses.** The reconstruction *broadens* — the
  super-sampled model cannot be constrained by undersampled data and the
  regularisation fills the gap with width.

The pipeline encodes the rule as a diagnostic: any STARRED PSF whose fitted FWHM falls
below 1.6 output pixels is flagged `undersampled` in its diagnostics block, warning you
that you are in the regime where the reconstruction broadens. Practical guidance:
STARRED for well-sampled or crowded fields (LW; also WFC3/UVIS — see the HST leg of
this validation in `hst_wfc3_uvis/psf.py`), photutils Tier 1 for the undersampled SW.

The code below re-runs both bands with `psf_backend="starred"` — same specs, same
caches, separate output tree — and prints each verdict. It is wrapped in a try/except
so the script survives a missing optional extra.
"""
starred_records = {}

for band in PSF_BANDS:
    spec = replace(m92_spec(band), psf_backend="starred")  # frozen dataclass -> replace, not mutate
    print(f"[{band}] re-running with the STARRED Tier-1b backend (same cache)...")
    try:
        starred_records[band] = reduce_target(
            spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT / "starred"
        )
    except Exception as error:
        print(
            f"[{band}] STARRED backend unavailable or failed ({error}). Install the "
            "optional extra (pip install 'autoreduce[starred]') to run Tier 1b; the "
            "Tier-1 reductions above are complete regardless."
        )
        break

for band, record in starred_records.items():
    psf_block = record["psf"]
    verdict = "UNDERSAMPLED — expect broadening" if psf_block.get("undersampled") else "well-sampled"
    print(
        f"[{band}] starred: fwhm {psf_block.get('fwhm_pix')} px, "
        f"n_stars {psf_block.get('n_stars_used')} -> {verdict}"
    )

"""
__Model PSFs — STPSF as Tier 2b__

JWST has an outstanding optical model: STPSF (formerly WebbPSF;
https://stpsf.readthedocs.io/; Perrin et al. 2014, Proc. SPIE 9143) simulates the PSF
from the measured, time-dependent wavefront. So why not just use it?

Because the literature keeps finding that **empirical beats model** for photometric and
morphological work. Zhuang et al. (2024,
https://iopscience.iop.org/article/10.3847/1538-4357/ad1517) find pure STPSF models
consistently disfavoured against empirical PSFs for AGN host decomposition — the model
lacks charge diffusion, interpixel capacitance, the source's spectral energy
distribution and residual wavefront error, and a mosaic PSF is additionally shaped by
the drizzle kernel and dither pattern. A model PSF can only compete after being pushed
through the same resampling as the data (the `spike` tool, published in JOSS, exists
for exactly this), and even then it is the fallback, not the first choice.

**PyAutoReduce** therefore uses STPSF only as **Tier 2b, for frame products**: when a
single native frame's star field cannot support a per-frame ePSF, `stpsf` evaluates the
model at that frame's detector and target position and ships the detector-sampled,
distortion-included kernel — keeping the frame modelable rather than shipping nothing.
Every such kernel carries the literature caveat verbatim in its diagnostics ("model-PSF
fallback — the JWST literature consistently prefers empirical PSFs for decomposition"),
so a model PSF is flagged, never silent. The mosaic path never uses it. See
`individual.py` for this fallback in action.

__Spatially-Varying PSFs — the Roadmap__

A single ePSF is a *position-independent* model, and the NIRCam PSF does vary across
the field. Zhuang & Shen (2024, https://arxiv.org/abs/2304.13776) quantified it across
8 filters: the spatial FWHM variation shrinks strongly with wavelength, from ~20% max
(~5% RMS) at F070W to ~3% max (~0.6% RMS) at F444W — and among SWarp, photutils and
PSFEx they rank PSFEx's polynomial spatially-varying model best. COSMOS-Web's own
weak-lensing pipeline characterises the PSF with ShOpt (Berman & McCleary,
https://arxiv.org/abs/2401.11625), benchmarked against PSFEx and PIFF.

This shapes **PyAutoReduce**'s JWST PSF tiering:

- **Tier 1 (implemented, default):** single mosaic-star ePSF — adequate for
  lens-galaxy work in the **LW** bands, where spatial variation is ~1% RMS.
- **Tier 1b (implemented, optional):** STARRED, per the regime rule above.
- **Tier 2 (roadmap — NOT implemented):** a spatially-varying empirical model
  (PSFEx/ShOpt-style polynomial) evaluated at the lens position — the quality upgrade
  path for the **SW** bands (~5% RMS variation) and any weak-lensing-grade use. Do not
  look for it in the current release; this paragraph exists so you know the limitation
  a single SW ePSF carries.
- **Tier 2b (implemented, frame products only):** STPSF, flagged, as above.
- **Not a tier:** target-based PSF reconstruction (fitting the PSF jointly with the
  lensed quasar images) — that is a modeling-stage technique, out of reduction scope.

__Plots__

The measured kernels: compact and full, both bands, log-stretched to show the wings and
the first Airy structure (crisp in F277W, softened by undersampling + resampling in
F150W).
"""
plot_dir = OUTPUT_ROOT / "m92_psf_plots"
plot_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for col, band in enumerate(PSF_BANDS):
    out_dir = OUTPUT_ROOT / f"m92_{band.lower()}"
    for row, product in enumerate(("psf.fits", "psf_full.fits")):
        kernel = fits.getdata(out_dir / product).astype(float)
        axes[row, col].imshow(
            np.log10(np.clip(kernel / kernel.max(), 1e-6, None)),
            origin="lower",
            cmap="magma",
            vmin=-5,
            vmax=0,
        )
        axes[row, col].set_title(f"{band} {product} ({kernel.shape[0]}x{kernel.shape[1]})")
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

fig.tight_layout()
plot_path = plot_dir / "m92_psf_kernels.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)

print(f"PSF kernel plot saved to: {plot_path.resolve()}")

"""
__Wrap Up__

The JWST PSF story in one paragraph: build it empirically from the mosaic's own stars
(Tier 1 ePSF, no peak cut in surface-brightness units), so the drizzled-PSF invariant
holds by construction; upgrade to STARRED where the PSF is well-sampled (LW) and avoid
it where it is not (SW — the 1.6-pixel undersampled flag); treat model PSFs as a
flagged frame-level fallback, never the mosaic default; and know that a single ePSF on
SW carries a ~5%-RMS spatial-variation limitation until the spatially-varying Tier 2
lands.

The following locations of the workspace are good places to checkout next:

- `scripts/jwst_nircam/individual.py`: per-frame ePSFs and the STPSF Tier-2b fallback in action.
- `scripts/hst_acs/psf.py`: the HST PSF story — star passes, tiers and the same invariant on AstroDrizzle.
- `scripts/jwst_nircam/multi_band.py`: how PSF quality differs across the four COSMOS-Web bands.
- `scripts/guides/output_contract.py`: where psf.fits and psf_full.fits sit in the product contract.

__Env__ (Developer Only)

ENV: network
"""
