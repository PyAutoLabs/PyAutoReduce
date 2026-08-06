"""
HST ACS: PSF
============

No product **PyAutoReduce** ships is more consequential — or easier to get subtly wrong — than
the PSF. A lens model convolves every trial image with it before comparing to the data, so a PSF
error does not average away like noise: it imprints *structured* residuals exactly where the
science lives, at the lensed arcs and the deflector core. Those residuals bias mass-model
parameters, and in substructure work they are actively dangerous — a PSF mismatch can mimic the
perturbation of a dark subhalo, or mask a real one. Empirical PSF craft is why the weak-lensing
community abandoned pure optical models for HST years ago, and lens modeling inherits that
lesson wholesale.

This script is the full ACS PSF story: where the effective-PSF idea comes from, how stars are
selected, which drizzle pass they are measured on (and the dial that controls it), the STARRED
alternative backend and the regimes where it wins and loses, what happens on star-poor fields
(an honest hard stop, currently), the model-PSF literature **PyAutoReduce** does *not* yet wrap,
and the diagnostics that let you judge every shipped kernel.

It runs the SLACS anchor up to three times (default pass, `no_cr` star pass, and — if the
optional extra is installed — the STARRED backend), sharing one exposure cache throughout. Warm
cache: a few minutes per run, plus one extra AstroDrizzle pass for the `no_cr` leg.

__Contents__

- **Imports:** Import **PyAutoReduce** and the other libraries we need.
- **Paths:** Anchor the cache and output locations to the workspace root.
- **Why PSF Errors Bias Lens Models:** Structured residuals at arcs; fake or masked substructure.
- **The ePSF Lineage:** Anderson & King 2000 to `photutils.EPSFBuilder`.
- **Star Selection:** The cuts, their defaults, and why they are not TargetSpec dials.
- **The Default Run:** Tier-1 ePSF on the SLACS anchor.
- **The Star Pass Dial:** `psf_star_pass` — "auto", "science", "no_cr", and the 344-to-599 story.
- **STARRED (Tier 1b):** The super-sampled backend, its regimes, and its validation numbers.
- **The Star-Poor Hard Stop:** Tier 2 is a stub — honesty about what cannot ship yet.
- **Model PSFs In The Literature:** TinyTim and focus-diverse ePSF libraries, as context.
- **Quality Diagnostics:** fwhm_pix, n_stars_used, moment FWHM — and the FWHM-definition wart.
- **The Drizzled-PSF Invariant:** Why the delivered PSF matches the mosaic geometry, always.
- **psf.fits vs psf_full.fits:** The compact kernel and the wings.
- **Plots:** The PSF at log stretch, and its radial profile.
- **Wrap Up:** Where to go next in the workspace.

__Imports__
"""
import dataclasses
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from autoreduce import TargetSpec, reduce_target
from autoreduce.psf.moments import moment_fwhm
from autoreduce.psf.stars import StarSelection

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the folder
containing `scripts/`). **PyAutoReduce** requires absolute paths: its drizzle step changes the
working directory internally, so relative paths would break.

Each PSF experiment gets its own output root; the exposure cache is shared, so the archive is
hit at most once.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"                # Shared with every other script in this folder.
OUTPUT_ROOT = WORKSPACE / "output" / "psf"     # One subfolder per PSF experiment below.

SPEC = TargetSpec(
    name="slacs0008-0004",  # The SLACS parity anchor; same cache entry as start_here.py.
    ra=2.012333,  # Right ascension in degrees.
    dec=-0.068944,  # Declination in degrees.
    proposal_ids=("10886",),  # The SLACS ACS program.
)

"""
__Why PSF Errors Bias Lens Models__

A pixel-based lens likelihood compares `PSF (x) model_image` against the data. If the kernel is
too narrow, every fit leaves a positive-ringed residual at the deflector core and sharpened arc
edges; too broad, the reverse. Unlike noise these residuals are deterministic and spatially
coherent — they sit at the arcs, which is precisely where the mass model gets its constraining
power. Consequences, in increasing severity:

- biased deflector light subtraction (the core is pure PSF at small radii);
- biased mass-model parameters (the arc surface-brightness gradients drive the fit);
- spurious or hidden substructure: a percent-level PSF error produces exactly the kind of
  localized arc residual a dark subhalo would.

This is why pure optical-model PSFs (TinyTim, below) fell out of favour at lensing fidelity —
they struggle with HST's time-variable focus ("breathing"), source-SED effects, CTE trailing and
charge diffusion, and their mismatch shows up as structured arc residuals that mimic or mask
substructure. The modern default, here and across HST practice, is *empirical*: measure the PSF
from stars in your own data.

__The ePSF Lineage__

The construction **PyAutoReduce** uses is the *effective PSF* (ePSF) of Anderson & King 2000
(PASP 112, 1360, https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A): the instrumental PSF
convolved with the pixel response, derived empirically by iterating between star-centroid
estimates and an oversampled PSF model built from many dithered star images. It is the
foundation of essentially all modern HST PSF work.

The implementation is `photutils.EPSFBuilder`
(https://photutils.readthedocs.io/en/stable/user_guide/epsf.html) — the open-source Anderson &
King iteration — run on star cutouts from the drizzled mosaic, at 2x oversampling, then
evaluated back onto the mosaic pixel grid and unit-normalised. At least 8 usable stars are
required; fewer is a loud `InsufficientStarsError`, not a degraded kernel.

__Star Selection__

Which stars feed the builder is governed by fixed selection cuts — deliberately *not*
`TargetSpec` dials. PSF fidelity should not vary target-to-target with user whim; the cuts are
tuned once for ACS-like mosaics and recorded here so you know exactly what they are:
"""
selection = StarSelection()

print("--- StarSelection defaults (not TargetSpec-tunable) ---")
print(f"detection_sigma     : {selection.detection_sigma}   (DAOStarFinder threshold, x sky RMS)")
print(f"fwhm_pix            : {selection.fwhm_pix}    (detection kernel FWHM)")
print(f"sharp_range         : {selection.sharp_range}  (rejects CRs/extended sources)")
print(f"round_limit         : {selection.round_limit}    (rejects elongated detections)")
print(f"saturation_fraction : {selection.saturation_fraction}    (of full well — ACS ~80 ke-)")
print(f"min_separation_pix  : {selection.min_separation_pix}   (uncrowded: no neighbour within)")
print(f"edge_margin_pix     : {selection.edge_margin_pix}     (full extraction window on-image)")
print(f"exclusion_radius_pix: {selection.exclusion_radius_pix}   (around the lens itself)")

"""
The physics behind the cuts: saturated stars have flat-topped, bleeding profiles (rejected by
the peak cap at 70% of the ~80,000 e- ACS full well); crowded stars carry neighbour flux into
the model; edge stars have truncated extraction windows; and the target itself — the lens galaxy
— must never contaminate its own PSF.

__The Default Run__

Run the reference reduction and read its PSF provenance.
"""
print(
    "\n"
    "Run 1/2: default PSF configuration.\n"
    "Warm cache: a few minutes. Cold cache: adds the ~0.5 GB MAST download.\n"
)

record_auto = reduce_target(SPEC, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT / "auto")

print("\n--- psf block, default run ---")
print(json.dumps(record_auto["psf"], indent=2))

"""
__The Star Pass Dial__

Note `star_source_pass` in that block. It answers a question you might not have thought to ask:
*which image were the stars measured on?* — and it exists because the obvious answer is wrong.

The science mosaic has been through `driz_cr` cosmic-ray rejection, and driz_cr's blotted-median
reference reads systematically low on steep gradients — which includes *star cores*. Sub-pixel
dither shifts smear the median's peak, the core pixels deviate, and genuine stellar flux gets
flagged as cosmic rays. Star finding on the science mosaic then sees cored, hole-punched stars:
fewer survive the sharpness cuts, and the survivors are damaged. Measured across the pipeline's
validation fields, rebuilding star selection from a CR-flag-ignoring pass raised the usable star
count from 344 to 599 (+74%) and rescued four lens/filter pairs that would otherwise have had no
viable ePSF at all.

`TargetSpec.psf_star_pass` controls this, with the star pass decoupled from the shipped science
mosaic:

- `"auto"` (default) — never adds a drizzle. The science mosaic is used, and the *reason* is
  recorded: on the single-exposure branch and the `cr_method="deepcr"` route it is genuinely the
  least-CR-rejected pass available, and otherwise it is the only pass available without doubling
  combine time.
- `"no_cr"` — the explicit opt-in: drizzle a second, CR-flag-ignoring star pass (`final_bits`
  gains the CR bit; `resetbits=0` so the science DQ flags survive for frame products) onto the
  same grid — same kernel, pixfrac, scale and rotation, so the drizzled-PSF invariant holds.
  Costs one extra AstroDrizzle run.
- `"science"` — pin star finding to the shipped mosaic, explicitly.

Whatever happens, `reduction.json` records `star_source_pass` (and the reason, when no second
pass ran) — the coupling can never silently regress. Let's buy the second pass and compare:
"""
print(
    "\n"
    "Run 2/2: psf_star_pass='no_cr' — adds a second, CR-flag-ignoring AstroDrizzle pass\n"
    "for star finding. Warm cache: a few extra minutes.\n"
)

spec_no_cr = dataclasses.replace(SPEC, psf_star_pass="no_cr")  # Frozen spec -> variants via replace().

record_no_cr = reduce_target(
    spec_no_cr, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT / "no_cr"
)

print("\n--- star pass comparison ---")
print(f"auto : pass={record_auto['psf']['star_source_pass']}, "
      f"n_stars={record_auto['psf']['n_stars_used']}")
print(f"       reason: {record_auto['psf'].get('star_source_reason')}")
print(f"no_cr: pass={record_no_cr['psf']['star_source_pass']}, "
      f"n_stars={record_no_cr['psf']['n_stars_used']}")

"""
On sparse fields like this one the star-count gain is modest; on richer fields it is the
difference between a viable ePSF and none. If your reduction reports few stars and a
`star_source_pass` of "science", the `no_cr` pass is the first thing to try.

__STARRED (Tier 1b)__

`TargetSpec.psf_backend="starred"` swaps the photutils builder for STARRED (Michalewicz et al.
2023, JOSS 8(85), 5340; Millon et al. 2024, https://arxiv.org/abs/2402.08725) — the
COSMOGRAIL/lensing community's JAX-based PSF reconstruction: an analytic Moffat core plus a
starlet-regularized, super-sampled residual grid, fit jointly to the same field stars Tier 1
selects, weighted by the per-pixel noise-map.

It is a *conditional* upgrade, not a universal one. The pipeline's validation mapped the regimes
on real data:

- **Wins — well-sampled and crowded/few-star fields.** On an Omega Cen WFC3/UVIS F606W field,
  STARRED's ePSF concentration (0.54) matched the empirical star-stack (0.58) far better than
  the photutils Tier-1 build (0.39 — under-concentrated and neighbour-contaminated in the
  crowd). On JWST NIRCam it matched the empirical PSF on the well-sampled long-wavelength
  channel (F277W) where the photutils build collapsed on few blended stars.

- **Loses — undersampled data.** On undersampled NIRCam short-wavelength imaging (F150W at
  0.03"/pix, PSF ~1.7 px) STARRED *broadens* — excess starlet-channel wings — and photutils
  wins. The backend flags this regime itself: kernels with FWHM below 1.6 px carry an
  `undersampled` warning in the diagnostics. ACS/WFC at 0.05"/pix sits near this boundary, so
  check the flag.

Two constraints keep STARRED optional: it is GPL-licensed (the PyAuto* stack is permissive) and
it depends on JAX — so it ships only as the extra `pip install "autoreduce[starred]"`, imported
lazily. If requested and unavailable the pipeline raises loudly (`StarredUnavailableError`); it
**never** silently falls back to Tier 1. We run it only if importable:
"""
if importlib.util.find_spec("starred") is not None:
    print("\nOptional run: psf_backend='starred' (STARRED tier 1b).\n")
    spec_starred = dataclasses.replace(SPEC, psf_backend="starred")
    record_starred = reduce_target(
        spec_starred, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT / "starred"
    )
    print("\n--- psf block, STARRED run ---")
    print(json.dumps(record_starred["psf"], indent=2))
else:
    print(
        "\n[starred] the STARRED extra is not installed — skipping the tier-1b run.\n"
        '[starred] install it with:  pip install "autoreduce[starred]"  (GPL + JAX).'
    )

"""
__The Star-Poor Hard Stop__

Honesty section. SLACS-like fields — an isolated elliptical in a snapshot pointing — are often
*star-poor*: fewer than the 8 usable stars Tier 1 needs. The designed answer is a Tier-2 *model*
PSF (a focus-diverse ePSF grid or TinyTim raytrace, drizzled through the same footprint as the
science frames). **That tier is currently a stub**: requesting it raises
`ModelPSFUnavailableError` unconditionally. A star-poor ACS field therefore cannot ship a PSF
today — the reduction stops loudly rather than delivering a kernel it cannot stand behind, and
the frame products (`individual.py`) record `method: "none"` per star-poor frame for the same
reason.

If you hit this: try `psf_star_pass="no_cr"` first (it exists exactly to rescue marginal
fields), then STARRED (robust at low star counts on well-sampled data). If both fail, the field
genuinely lacks the stars, and you should treat any externally-sourced PSF with the suspicion
this script's opening section motivates.

__Model PSFs In The Literature__

Context for what the Tier-2 stub will eventually wrap — literature, not implemented capability:

- **TinyTim** (Krist 1995; Krist, Hook & Stoehr 2011, Proc. SPIE 8127): the classical HST
  optical raytrace model. SLACS itself used it — Bolton et al. 2008 (SLACS V,
  https://arxiv.org/abs/0805.1931) rectified a TinyTim PSF with *identical sampling* to their
  rectified snapshot images, a clean precedent for the drizzled-PSF invariant below. Its known
  weaknesses at lensing fidelity: time-variable focus/breathing, SED dependence, CTE trailing,
  charge diffusion.
- **Focus-diverse ePSF libraries**: STScI's modern empirical alternative — ePSF grids indexed by
  focus, with the focus estimated from a handful of stars in the science frame itself (the
  COSMOS weak-lensing method recovered focus to <1 micron rms from few stars; Rhodes et al.
  2007, https://arxiv.org/abs/astro-ph/0701480). For ACS/WFC: Bellini et al. 2018 (ACS ISR
  2018-08) built the library, and ACS ISR 2023-06 documents the retrieval tooling
  (`acstools.focus_diverse_psfs`); STScI's maintained notebooks demonstrate it
  (https://spacetelescope.github.io/hst_notebooks/).

- **Not any tier — target-based PSF reconstruction.** Reconstructing the PSF from the science
  point sources themselves (lensed quasar images) is deliberately out of *reduction* scope: it
  entangles the PSF with the lensed arc and belongs to the modeling stage. **PyAutoReduce**
  ships the inputs and stops.

__Quality Diagnostics__

Every shipped PSF carries diagnostics in `reduction.json`; judge the kernel before you model
with it:

- `n_stars_used` — more stars, better-constrained ePSF; near the minimum of 8, be cautious.
- `fwhm_pix` — a *crude* radial half-max estimate, recorded as a build diagnostic.
- `star_source_pass` — which drizzle pass the stars came from (above).

One wart to know about, stated plainly: **different PSF tiers use different FWHM definitions.**
The tier-1 `fwhm_pix` is a radial half-max-extent estimate; the moment-based estimator below is
a Gaussian-sigma proxy (2.3548 x second moment); and the Keck tier-A vetting statistic is an
equivalent-area measure. They agree only for a Gaussian kernel — never cross-compare FWHM
numbers between tiers or instruments. Within one tier, they track relative quality fine.

The moment estimator is public, so we can compute it ourselves and see the wart directly:
"""
out_auto = OUTPUT_ROOT / "auto" / SPEC.name
psf = fits.getdata(out_auto / "psf.fits").astype(float)
psf_full = fits.getdata(out_auto / "psf_full.fits").astype(float)

print("\n--- FWHM, two definitions, same kernel ---")
print(f"recorded fwhm_pix (radial half-max): {record_auto['psf']['fwhm_pix']:.2f}")
print(f"moment_fwhm (2.3548 x sigma proxy) : {moment_fwhm(psf):.2f}")
print("These differ by construction — compare within one definition only.")

"""
__The Drizzled-PSF Invariant__

The non-negotiable rule, whatever tier produced the kernel: **the delivered PSF is the drizzled
PSF** — measured (or evaluated) at the same kernel, pixfrac, scale and orientation as the
science mosaic. The mosaic's blur is the *drizzled* blur; a native-frame PSF paired with a
drizzled image is simply the wrong kernel, and every path in **PyAutoReduce** honours this by
construction: mosaic stars are measured on a same-geometry drizzle pass, the `no_cr` star pass
reuses the exact science grid, and the frame-combined route (`individual.py`) applies the
drizzle drop and WCS geometry to the per-frame ePSFs explicitly.

__psf.fits vs psf_full.fits__

Two kernels ship per dataset, both odd-shaped, centred and unit-normalised:

- `psf.fits` (21x21, ~1" across) — the fit-convolution kernel. Big enough to capture the core
  and the first Airy structure — the great majority of the blurring — while keeping model-image
  convolution fast.
- `psf_full.fits` (61x61, ~3") — the extended kernel with the wings and diffraction-spike
  structure, for work where far-flung PSF flux matters (bright point sources, lensed quasars).
"""
print("\n--- kernel properties ---")
print(f"psf.fits      : {psf.shape}, sum={psf.sum():.6f}")
print(f"psf_full.fits : {psf_full.shape}, sum={psf_full.sum():.6f}")

"""
__Plots__

A PSF should always be inspected at log stretch — the core saturates any linear display, and the
structure you need to check (asymmetry, neighbour contamination, wings) lives orders of
magnitude down. We also plot the radial profile of both kernels.
"""
plot_dir = out_auto / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

floor = psf_full[psf_full > 0].min()
axes[0].imshow(np.log10(np.clip(psf, floor, None)), origin="lower", cmap="magma")
axes[0].set_title("psf.fits (log10)")
axes[1].imshow(np.log10(np.clip(psf_full, floor, None)), origin="lower", cmap="magma")
axes[1].set_title("psf_full.fits (log10)")

def radial_profile(kernel):
    ny, nx = kernel.shape
    y, x = np.mgrid[0:ny, 0:nx]
    r = np.hypot(y - ny // 2, x - nx // 2).astype(int)
    profile = np.bincount(r.ravel(), weights=kernel.ravel()) / np.bincount(r.ravel())
    return profile

axes[2].semilogy(radial_profile(psf), label="psf (21x21)")
axes[2].semilogy(radial_profile(psf_full), label="psf_full (61x61)", linestyle="--")
axes[2].set_xlabel("radius (pix, 0.05\"/pix)")
axes[2].set_ylabel("mean pixel value")
axes[2].set_title("radial profiles")
axes[2].legend()

fig.tight_layout()
plot_path = plot_dir / "psf_diagnostics.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)

print(f"\nPlot saved to: {plot_path}")

"""
The two profiles should lie on top of each other over the inner ~10 pixels — same PSF, different
window — with `psf_full` continuing smoothly into the wings.

__Wrap Up__

The PSF story in one paragraph: empirical ePSFs from carefully-selected field stars, measured on
the least-CR-damaged drizzle pass available (a recorded, dialable choice), on the exact geometry
of the science mosaic; a super-sampled STARRED alternative for well-sampled or crowded fields; a
loud hard stop — not a degraded kernel — where stars are insufficient; and diagnostics with
every dataset so none of this is taken on faith.

The following locations of the workspace are good places to checkout next:

- `scripts/hst_acs/individual.py`: per-frame native ePSFs and the frame-combined mosaic PSF.
- `scripts/hst_acs/simulator.py`: injection testing — the end-to-end check that the shipped PSF
  and noise-map describe the data.
- `scripts/hst_acs/dials.py`: how pixfrac and kernel choices propagate into the PSF.
- `scripts/guides/output_contract.py`: the kernel standards (odd, centred, normalised) and how
  **PyAutoLens** consumes them.

__Env__ (Developer Only)

ENV: network
"""
