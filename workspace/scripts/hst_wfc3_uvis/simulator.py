"""
Simulator: HST WFC3/UVIS
========================

This script validates the WFC3/UVIS pipeline end-to-end by *injection*: it paints a synthetic
lensed arc into the real J0252+0039 F390W `_flc` exposures, runs the identical reduction on
the injected frames, and measures how faithfully the known input flux comes back out of the
packaged dataset.

Injection is the honest way to simulate reduced data. Rendering a synthetic image from
scratch means inventing the sky, the cosmic rays, the bad pixels, the dither geometry and the
noise correlations; injecting into real frames gets all of them for free, exactly as they are
in the data your lens models will fit. If the recovered flux, the noise map and the PSF are
consistent for a source you put in yourself, you can trust them for the sources nature put in.

This script is deliberately compact — the full injection methodology (unit handling, the
per-frame rendering path, Poisson realization, seed policy, aperture statistics) is
documented in `hst_acs/simulator.py`, which you should read for depth. Here we cover the
UVIS-specific points and run the loop.

__Contents__

- **Imports:** Import the required Python libraries.
- **Paths:** Anchor every path to the workspace root, absolutely.
- **The Input Image:** Build a pure-numpy Sersic arc in e-/s — no lensing code required.
- **The Injection Dials:** Declare the injection on top of the anchor spec.
- **Clean and Injected Reductions:** Run the pipeline twice with a shared cache.
- **Recovery:** Difference the two datasets and check aperture flux recovery.
- **Wrap Up:** Summary of the script and next steps.

__Imports__

Everything here is numpy, astropy and **PyAutoReduce** — building the input image needs no
lensing code, by design: the injection contract takes a plain 2-D FITS image.
"""
import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoreduce import TargetSpec, reduce_target

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the folder
containing `scripts/`). **PyAutoReduce** requires absolute paths: its drizzle step changes the
working directory internally, so relative paths would break.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"      # downloaded exposures + CRDS references (re-used across runs)
OUTPUT_ROOT = WORKSPACE / "output"    # reduced datasets, one folder per target

RA, DEC = 43.188375, 0.666222  # SDSS J0252+0039 — the UVIS anchor field (see start_here.py).

"""
__The Input Image__

The input to injection is a plain 2-D FITS image with three hard requirements: finite,
non-negative pixels; *not* PSF-convolved (the pipeline convolves it with each frame's own
PSF, so blurring it yourself would double-convolve); and surface brightness in the adapter's
injection units — for HST, electrons per second per pixel.

We build a simple analytic lensed-arc stand-in: a partial ring whose radial cross-section is
a Sersic n=1 (exponential) profile,

    I(r, phi) = exp( -1.678 * |r - r_ring| / r_eff ) * taper(phi)

with a smooth Gaussian taper in azimuth so the ring becomes an arc. The 1.678 factor makes
r_eff the profile's half-light radius. This is deliberately *not* a ray-traced lensed source
— the point of this script is the pipeline's bookkeeping, and a fully analytic input keeps
the truth exact. The image is normalized so its pixels sum to exactly the total flux we want
to recover, then written to FITS at a chosen pixel scale (which need not match the detector's
— the renderer resamples through each frame's WCS).
"""
ARC_FLUX_CPS = 25.0        # Total injected flux in e-/s: bright enough for a clean measurement.
ARC_PIXEL_SCALE = 0.02     # Input-image pixel scale in arcsec/pix (finer than UVIS native).
ARC_RING_RADIUS = 1.2      # Arc ring radius in arcsec — a typical galaxy-scale Einstein radius.
ARC_R_EFF = 0.15           # Sersic half-light width of the arc cross-section in arcsec.

shape = (241, 241)
yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
cy, cx = shape[0] // 2, shape[1] // 2

r = np.hypot(yy - cy, xx - cx) * ARC_PIXEL_SCALE     # Radius from image centre in arcsec.
phi = np.arctan2(yy - cy, xx - cx)                   # Azimuth in radians.

arc = np.exp(-1.678 * np.abs(r - ARC_RING_RADIUS) / ARC_R_EFF)  # Sersic n=1 cross-section.
arc *= np.exp(-0.5 * ((phi - np.pi / 4.0) / 0.8) ** 2)          # Azimuthal taper -> an arc, not a ring.
arc = ARC_FLUX_CPS * arc / arc.sum()                            # Normalize to the truth flux.

from astropy.io import fits

input_dir = OUTPUT_ROOT / "j0252+0039_f390w_injected"
input_dir.mkdir(parents=True, exist_ok=True)

input_path = input_dir / "input_arc.fits"
fits.PrimaryHDU(arc.astype(np.float32)).writeto(input_path, overwrite=True)

print(f"Input arc written to: {input_path} (total flux {arc.sum():.3f} e-/s)")

"""
__The Injection Dials__

The injection is declared on the `TargetSpec` like every other dial. Starting from the anchor
spec of `start_here.py`, we add the `inject_*` dials and a new `name` so the injected dataset
lives beside the clean one. The injection position is offset ~2" from the lens so the arc
lands on clean sky and the difference measurement is unambiguous.

The pipeline renders the arc into every `_flc` exposure through that frame's own
full-distortion WCS, convolves with the frame's PSF, draws Poisson noise for the source's own
counts (seeded per frame — reproducible), and updates the ERR extension. The cache is never
mutated: injection works on copies, so your downloaded exposures stay pristine.

One UVIS note: this adapter also supports `cr_method="deepcr"` — the CNN cosmic-ray flagger
of Zhang & Bloom (2020, ApJ 889, 24; https://ui.adsabs.harvard.edu/abs/2020ApJ...889...24Z)
as an opt-in alternative to `driz_cr` (see e.g. the label-free UVIS retraining of Chen et
al. 2024). Injection runs are a natural place to try it, since you can measure directly
whether either CR method eats injected flux — the trade study lives in `hst_acs/dials.py`.
"""
INJECT_RA = RA + 2.0 / 3600.0  # Inject ~2" east of the lens, on clean sky.

spec_clean = TargetSpec(
    name="j0252+0039_f390w",  # Same name as start_here.py -> its output is re-used if present.
    ra=RA,                    # Target right ascension in degrees.
    dec=DEC,                  # Target declination in degrees.
    instrument="wfc3_uvis",   # UVIS adapter: _flc, iref, native 0.0396"/pix.
    filter_name="F390W",      # The anchor filter.
    final_scale=0.0396,       # Bayer anchor dial (see start_here.py).
    final_pixfrac=1.0,        # Bayer anchor dial.
)

spec_injected = dataclasses.replace(
    spec_clean,
    name="j0252+0039_f390w_injected",   # Separate output folder for the injected dataset.
    inject_image=str(input_path),       # The plain-FITS arc built above (absolute path).
    inject_pixel_scale=ARC_PIXEL_SCALE, # arcsec/pix of the input image (required with inject_image).
    inject_position=(INJECT_RA, DEC),   # (ra, dec) degrees to centre the arc on; default is the target.
    inject_seed=0,                      # Seeds the per-frame Poisson draws -> bit-reproducible.
    # cr_method="deepcr",               # [Optional] CNN CR flagging instead of driz_cr (UVIS-supported).
)

"""
__Clean and Injected Reductions__

Two reductions with a shared cache: once clean, once injected, otherwise identical — so the
recovery measurement is a difference of two identically-processed datasets and everything
except the arc cancels. If you ran `start_here.py`, the clean reduction and all downloads are
already on disk and only the injected run costs time.
"""
print(
    "Running the clean reduction (skipped in effect if start_here.py already "
    "ran — the cache and output are shared)..."
)

record_clean = reduce_target(spec_clean, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)

print("Running the injected reduction (cache warm — no new downloads)...")

record_injected = reduce_target(
    spec_injected, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT
)

out_clean = OUTPUT_ROOT / spec_clean.name
out_injected = OUTPUT_ROOT / spec_injected.name

"""
The provenance of the injected run carries a dedicated `inject` block — the input image, its
units and flux, the position, the per-frame injected counts and the seed — so a
semi-synthetic dataset can never masquerade as real data.
"""
print(f"inject block: {json.dumps(record_injected['inject'], indent=2)}")

"""
__Recovery__

Now the measurement. Difference the injected and clean cutouts (identical processing means
the real sky, lens galaxy and neighbors cancel), locate the injection position through the
cutout's WCS, and sum the difference inside a 3" aperture. Three numbers to compare:

- the truth: the input arc's total flux in e-/s;
- the recovery: the aperture sum of the difference image;
- the noise prediction: the quadrature sum of the noise map inside the aperture, which sets
  the uncertainty on the recovery.

A recovery ratio consistent with 1.0 within the noise says the pipeline conserves flux
through rendering, PSF convolution, drizzling, CR flagging and packaging — and that the
noise map is a fair uncertainty for aperture photometry on this dataset.
"""
from astropy.wcs import WCS

data_clean = fits.getdata(out_clean / "data.fits").astype(float)
data_injected = fits.getdata(out_injected / "data.fits").astype(float)
noise_injected = fits.getdata(out_injected / "noise_map.fits").astype(float)
header = fits.getheader(out_injected / "data.fits")

diff = data_injected - data_clean

xy = WCS(header).world_to_pixel_values(INJECT_RA, DEC)
yy, xx = np.mgrid[0 : diff.shape[0], 0 : diff.shape[1]]

pixel_scale = record_injected["package"]["pixel_scale"]
aperture = np.hypot(yy - xy[1], xx - xy[0]) * pixel_scale <= 3.0

recovered = float(diff[aperture].sum())
noise_pred = float(np.sqrt((noise_injected[aperture] ** 2).sum()))

print(f"injected flux    : {ARC_FLUX_CPS:.3f} e-/s")
print(f"recovered flux   : {recovered:.3f} e-/s (3\" aperture)")
print(f"recovery ratio   : {recovered / ARC_FLUX_CPS:.3f}")
print(f"aperture noise   : {noise_pred:.3f} e-/s")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, (title, image) in zip(
    axes,
    [("clean", data_clean), ("injected", data_injected), ("difference", diff)],
):
    im = ax.imshow(np.arcsinh(image / 0.01), origin="lower", cmap="magma")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle("j0252+0039_f390w: arc injection and recovery")
fig.tight_layout()

plot_dir = out_injected / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

plot_path = plot_dir / "injection_recovery.png"
fig.savefig(plot_path, dpi=120)
plt.close(fig)

print(f"Plot saved to: {plot_path}")

"""
__Wrap Up__

You have injected a known synthetic arc into real UVIS exposures, reduced them exactly as the
real data, and recovered the input flux against the noise map's own uncertainty prediction —
the strongest end-to-end validation a reduction pipeline offers, and one you can rerun on any
field with dials of your choosing.

The following locations of the workspace are good places to checkout next:

- `scripts/hst_acs/simulator.py`: the full injection methodology — rendering, units, seeds,
  Poisson realization and the aperture statistics in depth.
- `scripts/hst_wfc3_uvis/start_here.py`: the clean anchor reduction differenced against here.
- `scripts/hst_acs/dials.py`: the CR-method trade study (`driz_cr` vs `deepcr`) injection
  helps you probe.
- `scripts/jwst_nircam/simulator.py`: the same idea through the JWST path, in Jy units.

__Env__ (Developer Only)

ENV: network
"""
