"""
HST ACS: Simulator (Injection)
==============================

How do you know a reduction pipeline is *right*? Parity against a legacy dataset (the SLACS
comparison behind this folder) checks consistency with history — but history's own reductions
carry their own choices. The sharper test is a controlled experiment: put a source of **known
brightness** into the data and check that the pipeline hands it back. If a synthetic lensed arc
of exactly 30 e-/s goes into the raw exposures, and 30 e-/s (within the noise-map's prediction!)
comes out of the packaged cutout, then the flux calibration, the drizzle, the PSF treatment and
the noise model have all been validated *together*, end to end.

**PyAutoReduce** builds this in as the inject stage: give the `TargetSpec` a plain FITS image
and the pipeline deposits it into the real calibrated exposures — through each frame's own
distortion, blurred by each frame's own PSF, with its own Poisson noise — before the unmodified
pipeline runs. This script does exactly that on the SLACS anchor: it builds a synthetic
lensed-arc image in pure numpy (the math is shown — no lensing library needed), injects it,
reduces the field twice (clean and injected) off one shared cache, and measures the recovery.

Warm cache: two pipeline runs, a few minutes each. Cold cache: add the ~0.5 GB MAST download.

__Contents__

- **Imports:** Import **PyAutoReduce** and the other libraries we need.
- **Paths:** Anchor the cache and output locations to the workspace root.
- **Why Inject Into Real Frames:** What real data gives you that simulation from scratch cannot.
- **The Input Image Contract:** Units, sampling, non-negativity — and no PSF convolution.
- **Building A Synthetic Arc:** An analytic ring/arc profile, formula by formula.
- **The Clean Reduction:** The baseline run.
- **The Injected Reduction:** `dataclasses.replace` with the `inject_*` dials.
- **What The Inject Stage Did:** Per-frame deposit, PSF, Poisson draw, and the untouched cache.
- **The Difference Image:** Injected minus clean — two identically-processed datasets.
- **Flux Recovery:** Aperture photometry of the difference vs the injected truth.
- **The Injected Provenance Block:** Semi-synthetic data must say so, permanently.
- **Wrap Up:** Where to go next in the workspace.

__Imports__
"""
import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from autoreduce import TargetSpec, reduce_target

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the folder
containing `scripts/`). **PyAutoReduce** requires absolute paths: its drizzle step changes the
working directory internally, so relative paths would break.

The clean and injected runs write to separate output roots but share one exposure cache — the
injection never touches the cached files (more on that below), so a single download serves both.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"                    # Shared; never mutated by injection.
OUTPUT_ROOT = WORKSPACE / "output" / "simulator"   # clean/ and injected/ subfolders below.

SPEC = TargetSpec(
    name="slacs0008-0004",  # The SLACS parity anchor; same cache entry as start_here.py.
    ra=2.012333,  # Right ascension in degrees.
    dec=-0.068944,  # Declination in degrees.
    proposal_ids=("10886",),  # The SLACS ACS program.
)

"""
__Why Inject Into Real Frames__

You could instead simulate an observation from scratch: render a lensed source, convolve with a
PSF model, add Gaussian-plus-Poisson noise on a clean grid. Every lens-modeling workspace does
this for method development, and it is the right tool there. But as a *pipeline validation* — or
as training data for machine learning that must survive contact with real images — synthetic-
from-scratch is missing everything that makes real data hard:

- cosmic rays, with realistic morphology, at the real rate;
- the real sky pedestal and its frame-to-frame variation;
- the real correlated-noise structure the drizzle geometry induces;
- real bad pixels, saturation bleeds, and DQ structure;
- the real dither geometry and its coverage non-uniformity;
- the real PSF, wings and all.

Injecting synthetic sources into *real* survey images is the established answer in wide-field
astronomy — Balrog (Suchyta et al. 2016, MNRAS 457, 786) is the archetype, built on the GalSim
rendering machinery (Rowe et al. 2015, A&C 10, 121) — and lens-finding has adopted it for
exactly the realism argument: HOLISMOKES XV (Cañameras et al., https://arxiv.org/abs/2411.18694)
paints lensed features onto real cutouts because injection preserves the observational
complexity synthetic-only training data lacks. **PyAutoReduce**'s inject stage is the same idea
one level deeper: into the calibrated *exposures*, before the pipeline, so every downstream
stage is exercised for free.

__The Input Image Contract__

The inject stage takes a plain 2-D FITS image — deliberately free of any lensing-library
dependency — with a strict contract:

- **Units: e-/s per pixel** (the ACS adapter's `inject_units`) — surface brightness on your
  input grid, converted per frame to electrons using each exposure's own time.
- **Finite and non-negative** everywhere.
- **North-up**, centred at `inject_position` (default: the target itself), sampled at
  `inject_pixel_scale` — which may be *finer* than the detector; the deposit is
  flux-conserving.
- **NOT PSF-convolved.** This is the one people trip over: the pipeline convolves your image
  with each frame's own tier-1 ePSF (or your `inject_psf`) as it deposits. Hand it the *true*
  sky-plane surface brightness; if you pre-blur it, it gets blurred twice.

__Building A Synthetic Arc__

We build a lensed-arc-like ring in pure numpy. No ray tracing — just an analytic profile that
*looks* like the thin, curved arcs strong lenses produce, which is all a flux-recovery test
needs. On a polar grid (r, phi) centred on the lens:

    radial   : I_r(r)    = exp( -1.678 * |r - r_E| / w )          # Sersic n=1 cross-section,
                                                                  # peaking on the ring r = r_E
    azimuthal: I_phi(phi) = exp( -(phi - phi_0)^2 / (2 sigma_phi^2) )   # a bright arc segment
               + f_c * exp( -(phi - phi_0 - pi)^2 / (2 sigma_c^2) )     # + faint counter-arc

    I(r, phi) = I_r(r) * I_phi(phi),  then normalised so  sum(I) * 1 = F_total  [e-/s]

with r_E = 1.2" (an Einstein-radius-like ring size), w = 0.15" (arc thickness), a ~100-degree
bright arc and a fainter counter-image on the opposite side — the classic morphology of a
galaxy-scale lens. The 1.678 factor makes w the radial half-light scale of the n=1 profile.
"""
INJECT_PIXEL_SCALE = 0.025  # arcsec/pix — finer than the 0.05" detector; deposit is flux-conserving.
INJECT_FLUX_CPS = 30.0      # Total injected flux, e-/s — bright enough for a clean recovery test.

shape = (241, 241)  # 241 x 0.025" = ~6" across; comfortably contains the r_E = 1.2" ring.
cy, cx = shape[0] // 2, shape[1] // 2

yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
r = np.hypot(yy - cy, xx - cx) * INJECT_PIXEL_SCALE     # radius from the lens centre, arcsec
phi = np.arctan2(yy - cy, xx - cx)                       # position angle, radians

r_einstein = 1.2      # ring radius, arcsec
width = 0.15          # radial arc thickness (n=1 scale), arcsec
phi_0 = np.pi / 3.0   # bright arc centre angle
sigma_phi = 0.9       # bright arc angular extent (~100 deg FWHM-ish), radians
counter_frac = 0.25   # counter-image brightness relative to the main arc
sigma_counter = 0.35  # counter-image angular extent, radians

radial = np.exp(-1.678 * np.abs(r - r_einstein) / width)


def wrapped(delta):
    """Angular difference wrapped to [-pi, pi], so arcs don't tear at the branch cut."""
    return np.angle(np.exp(1j * delta))


azimuthal = np.exp(-wrapped(phi - phi_0) ** 2 / (2.0 * sigma_phi**2))
azimuthal += counter_frac * np.exp(
    -wrapped(phi - phi_0 - np.pi) ** 2 / (2.0 * sigma_counter**2)
)

arc = radial * azimuthal
arc = INJECT_FLUX_CPS * arc / arc.sum()  # Normalise: total flux = INJECT_FLUX_CPS e-/s exactly.

assert np.isfinite(arc).all() and (arc >= 0.0).all()  # The input contract, checked.

input_dir = OUTPUT_ROOT
input_dir.mkdir(parents=True, exist_ok=True)
input_path = input_dir / "input_arc.fits"
fits.PrimaryHDU(arc.astype(np.float32)).writeto(input_path, overwrite=True)

print(f"Synthetic arc written to: {input_path}")
print(f"  total flux : {arc.sum():.3f} e-/s")
print(f"  peak       : {arc.max():.4f} e-/s per {INJECT_PIXEL_SCALE}\" pixel")

"""
__The Clean Reduction__

First the baseline: the unmodified field, reduced exactly as in `start_here.py`. The recovery
measurement below is a *difference* of two identically-processed datasets, so the real lens
galaxy, the real arc, the sky and every static artifact subtract out — only the injected flux
(and noise) remains.
"""
print(
    "\n"
    "Run 1/2: clean reduction (the baseline).\n"
    "Warm cache: a few minutes. Cold cache: adds the ~0.5 GB MAST download.\n"
)

record_clean = reduce_target(
    SPEC, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT / "clean"
)

"""
__The Injected Reduction__

`TargetSpec` is frozen, so the injected variant is a `dataclasses.replace` — the four `inject_*`
dials on top of the identical spec. Everything else about the pipeline is unchanged: same
exposures, same drizzle, same noise recipe, same PSF machinery.
"""
spec_injected = dataclasses.replace(
    SPEC,
    inject_image=str(input_path),  # The plain-FITS arc built above (absolute path, like all paths).
    inject_pixel_scale=INJECT_PIXEL_SCALE,  # Required with inject_image: the input's arcsec/pix.
    inject_seed=1,  # Seeds the Poisson draws; same seed = bit-identical injection, re-run to re-run.
)
# inject_position is omitted -> defaults to the target coordinates: the ring lands on the lens.

print(
    "\n"
    "Run 2/2: injected reduction — same spec + inject_* dials, same shared cache.\n"
)

record_injected = reduce_target(
    spec_injected, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT / "injected"
)

print("Both reductions complete.")

"""
__What The Inject Stage Did__

Between acquisition and combination, for every cached exposure and every CCD chip:

1. **Render**: the input image is deposited onto the chip's native pixels through the frame's
   *full* distortion WCS — a flux-conserving drizzle-style deposit, so your 0.025" input grid
   lands correctly on the 0.05" distorted detector pixels.
2. **Convolve**: the deposited image is blurred with the frame's own tier-1 ePSF (built from
   that frame's stars), or `inject_psf` if you supplied one.
3. **Poisson**: expected electrons = cps x exposure time; an actual Poisson realisation is
   drawn, seeded deterministically per file from `(inject_seed, crc32(filename))` — so the run
   is exactly reproducible, but no two frames share a realisation.
4. **Update**: SCI gains the counts, ERR is updated in quadrature (the injected source carries
   its own shot noise), and the headers gain INJECTED/INJIMG/INJSEED keywords.

Critically, **the cache is never mutated**: injection operates on copies under the run's
`work/injected/` directory. The clean run and every future run see pristine exposures — which
is what made sharing one cache across both runs safe.

__The Difference Image__

Load both cutouts and difference them. What remains should be *only* the injected arc, blurred
by the real PSF and carrying the real noise — riding on nothing, because everything real
subtracted out.
"""
clean_dir = OUTPUT_ROOT / "clean" / SPEC.name
inj_dir = OUTPUT_ROOT / "injected" / SPEC.name

data_clean = fits.getdata(clean_dir / "data.fits").astype(float)
data_inj = fits.getdata(inj_dir / "data.fits").astype(float)
noise_inj = fits.getdata(inj_dir / "noise_map.fits").astype(float)
header_inj = fits.getheader(inj_dir / "data.fits")

diff = data_inj - data_clean

sky_rms = record_injected["noise"]["empirical_background_rms"]

plot_dir = OUTPUT_ROOT / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(np.arcsinh(data_clean / (3.0 * sky_rms)), origin="lower", cmap="magma")
axes[0].set_title("clean data")
axes[1].imshow(np.arcsinh(data_inj / (3.0 * sky_rms)), origin="lower", cmap="magma")
axes[1].set_title("injected data")
axes[2].imshow(np.arcsinh(diff / (3.0 * sky_rms)), origin="lower", cmap="magma")
axes[2].set_title("difference (injected - clean)")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plot_path = plot_dir / "injection_difference.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)

print(f"\nPlot saved to: {plot_path}")

"""
The difference panel should show the ring-plus-counter-arc morphology you built above — now
PSF-blurred and drizzled — centred on the (subtracted-away) lens.

__Flux Recovery__

The quantitative test: sum the difference image inside a 3" aperture at the injection position
and compare against the injected truth. The noise-map supplies the uncertainty prediction — the
quadrature sum of the per-pixel RMS over the aperture — so this simultaneously tests the flux
calibration *and* the noise model's absolute scale.
"""
xy = WCS(header_inj).world_to_pixel_values(SPEC.ra, SPEC.dec)
pixel_scale = record_injected["package"]["pixel_scale"]

yy_c, xx_c = np.mgrid[0 : diff.shape[0], 0 : diff.shape[1]]
aperture = np.hypot(yy_c - xy[1], xx_c - xy[0]) * pixel_scale <= 3.0

# Exclude masked-by-noise pixels (1e8) from the aperture prediction.
usable = aperture & (noise_inj < 1.0e7)

recovered = float(diff[usable].sum())
noise_pred = float(np.sqrt((noise_inj[usable] ** 2).sum()))

report = {
    "injected_flux_cps": INJECT_FLUX_CPS,
    "recovered_flux_cps_3arcsec": recovered,
    "recovery_ratio": recovered / INJECT_FLUX_CPS,
    "aperture_noise_cps": noise_pred,
    "total_injected_e": record_injected["inject"]["total_injected_e"],
    "n_frames_injected": len(record_injected["inject"]["frames"]),
}
(OUTPUT_ROOT / "recovery_report.json").write_text(json.dumps(report, indent=2))

print("\n--- flux recovery ---")
print(json.dumps(report, indent=2))

ok = abs(report["recovery_ratio"] - 1.0) < max(0.05, 3.0 * noise_pred / INJECT_FLUX_CPS)
print(f"\nRECOVERY {'OK' if ok else 'DISCREPANT'} "
      f"(criterion: within 5% or 3x the aperture noise prediction)")

"""
A recovery ratio of ~1 says the whole chain conserves flux: the per-frame deposit, the PSF
convolution (the 3" aperture is wide enough to recapture the PSF-scattered flux), the drizzle,
the cps unit handling and the cutout. A ratio consistently off by more than the noise allows
would localise a real bug — which is exactly how this test is used in the pipeline's own
validation.

__The Injected Provenance Block__

The governing principle: **semi-synthetic data must never masquerade as real.** The injected
run's `reduction.json` carries an `inject` block recording the input image, its units and pixel
scale, the total input flux, the position, the PSF source used for the convolution, the seed,
the total electrons deposited, and a per-frame record. Any consumer of this dataset — including
you, in two years — can see at a glance that it is an injection experiment, and reproduce it
bit-for-bit.
"""
print("\n--- inject block (per-frame records truncated) ---")
inject_block = dict(record_injected["inject"])
inject_block["frames"] = f"[{len(inject_block['frames'])} per-frame records]"
print(json.dumps(inject_block, indent=2))

print("\nThe clean run's record has no inject block:",
      "inject" in record_clean, "(False = clean, as it should be)")

"""
__Wrap Up__

You built a synthetic lensed arc from two exponentials and a normalisation, pushed it through
the real SLACS exposures — real cosmic rays, real sky, real PSF, real drizzle — and measured it
back to within the noise. That closed loop is the strongest single statement this workspace can
make about the reduction being right; it is also the template for building injection-based
training sets with fully-realistic systematics.

The following locations of the workspace are good places to checkout next:

- `scripts/hst_acs/individual.py`: the per-frame products — injection happens at frame level,
  and the frames are where you can see it before the drizzle.
- `scripts/hst_acs/dials.py`: rerun this experiment at different pixfrac values to see the
  noise-correlation story in the recovery uncertainties.
- `scripts/hst_acs/psf.py`: the frame ePSFs that blurred your injected arc.
- `scripts/guides/target_spec.py`: all the inject_* dials, including inject_position and
  inject_psf, in one place.

__Env__ (Developer Only)

ENV: network
"""
