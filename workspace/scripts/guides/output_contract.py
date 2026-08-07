"""
Guides: The Output Contract
===========================

Every **PyAutoReduce** reduction, on every instrument, honours one output contract: four FITS
products plus a provenance record, landing in `output/<target>/`. This guide walks that contract
in depth — what each file contains, what its header promises, the conventions (masked-by-noise,
the drizzled-PSF invariant) that make the products modeling-ready with no further preparation,
and how `reduction.json` documents everything that happened.

This is a runnable script that operates on the output of `scripts/start_here.py` — the SLACS
J0008-0004 HST/ACS reduction. It needs no network itself: if you have not run `start_here.py`
yet, it prints a pointer and exits cleanly, and the prose still reads as documentation.

__Contents__

- **Imports:** Import the standard libraries and the **PyAutoReduce** constants we inspect.
- **Paths:** Anchor to the workspace root and guard on the `start_here.py` output existing.
- **The Five Products:** The contract at a glance.
- **data.fits:** The science cutout — WCS, units (electrons/second vs MJy/sr), header keywords, and the strict-coverage cutout rule.
- **noise_map.fits:** The matching RMS map and the masked-by-noise convention (bad pixels carry noise 1e8, data zeroed).
- **psf.fits vs psf_full.fits:** Compact kernel for convolution vs extended wings, and the drizzled-PSF invariant.
- **reduction.json:** The provenance record, block by block.
- **Frame Products:** The opt-in per-exposure tree — `frames/manifest.json` and per-frame product sets.
- **The Visibility Triplet:** What the contract becomes for ALMA — three `(N_vis, 2)` arrays.
- **Loading With PyAutoLens:** `al.Imaging.from_fits` / `al.Interferometer.from_fits`, guarded.
- **Wrap Up:** Summary and good places to checkout next.

__Imports__

Alongside the standard FITS/plotting libraries we import `MASKED_NOISE_VALUE` from
**PyAutoReduce**'s public noise module — the sentinel the masked-by-noise convention is built on.
"""

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from autoreduce.noise.rms import MASKED_NOISE_VALUE

"""
__Paths__

We anchor to the workspace root and point at the dataset `scripts/start_here.py` produces. The
guard below keeps this guide honest: it never fabricates products, so without the `start_here.py`
output on disk there is nothing to inspect and we exit cleanly.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = WORKSPACE / "output"

TARGET = "slacs0008-0004"
out_dir = OUTPUT_ROOT / TARGET

if not (out_dir / "reduction.json").exists():
    print(
        f"No reduction found at {out_dir}.\n"
        "Run `python scripts/start_here.py` first (network + autoreduce[hst] required) — "
        "it produces the SLACS J0008-0004 dataset this guide inspects.\n"
        "Exiting cleanly; the prose in this file still reads as documentation."
    )
    sys.exit(0)

record = json.loads((out_dir / "reduction.json").read_text())

"""
__The Five Products__

The contract, identical for every imaging instrument in the workspace:

- `data.fits` — float32 science cutout at the modeling pixel scale, WCS intact.
- `noise_map.fits` — the matching per-pixel RMS, correlated-noise corrected, masked pixels at 1e8.
- `psf.fits` — compact (default 21x21) unit-normalised PSF for fit convolution.
- `psf_full.fits` — extended (default 61x61) PSF carrying the wings.
- `reduction.json` — the provenance record (whose content `reduce_target` also returns).

The `package` block of the provenance lists exactly what shipped — on a Keck tier-A reduction,
for example, you would also see `psf_candidate_<i>.fits` epoch candidates here, and with
`frame_products=True` a `frames/manifest.json` entry.
"""
print(f"Products shipped for {TARGET}: {record['package']['products']}")

"""
__data.fits__

The science cutout. Three promises the header makes, and why each matters for modeling:

**Units.** `BUNIT` records the pixel units. For HST and Keck reductions this is electrons/second
(the drizzle combines in `cps` units), which is what **PyAutoLens** assumes when computing
physical quantities like magnitudes. JWST is the deliberate exception: NIRCam mosaics stay in
their native surface-brightness units of MJy/sr — flux work then goes through the pixel area,
and the `jwst_nircam` scripts spell out the conversion. Never assume; read `BUNIT` (or
`package.data_units` in the provenance).

**WCS.** The cutout keeps its world coordinate system, so the products remain on-sky addressable
— you can overlay catalogues, cross-match with survey cutouts, or check the target's position.
This is a deliberate deviation from legacy lens-modeling cutouts, whose stripped headers made
that impossible.

**Exposure metadata.** `EXPTIME`/`TEXPTIME`, `FILTER`, `INSTRUME` and `TELESCOP` ride along, so a
dataset found on disk two years from now identifies itself.

One packaging rule worth knowing: the cutout is taken in strict-coverage mode — if the requested
`cutout_shape` extends off the drizzled mosaic, packaging *raises* rather than padding with
garbage. Size the cutout to the coverage, not the other way round.
"""
data_hdu = fits.open(out_dir / "data.fits")[0]
data = data_hdu.data.astype(float)
header = data_hdu.header

print(f"data.fits shape: {data.shape}, dtype float32 on disk")
for key in ("BUNIT", "EXPTIME", "FILTER", "INSTRUME", "TELESCOP"):
    if key in header:
        print(f"  {key:8s} = {header[key]}")

wcs = WCS(header)
centre_world = wcs.pixel_to_world(data.shape[1] // 2, data.shape[0] // 2)
print(f"  Cutout centre on sky: {centre_world.to_string('decimal')} (deg)")

"""
__noise_map.fits__

The per-pixel RMS matching `data.fits` — the same shape, the same units, and already carrying the
Casertano correlated-noise correction (the full derivation and closure checks live in
`scripts/guides/noise_maps.py`).

The convention that saves you a mask file downstream is **masked-by-noise**: any pixel the
pipeline rejected (an isolated dead or fully-rejected pixel in the resampled stack) is shipped
with its noise set to `MASKED_NOISE_VALUE` = 1e8 and its data zeroed. A chi-squared then ignores
those pixels automatically — no separate mask FITS, no mask surgery in **PyAutoLens**.

The policy behind it is deliberately strict, and recorded in the `bad_pixel_policy` block of the
provenance: only *isolated* bad pixels may be masked, never more than 0.5% of the cutout, and
never within 1.5" of the target — a structured defect, a bad fraction, or a bad pixel on the lens
itself fails the reduction loudly instead. Masking is a convenience for scattered singletons, not
a licence to paper over a broken reduction.
"""
noise_map = fits.getdata(out_dir / "noise_map.fits").astype(float)

masked = noise_map >= MASKED_NOISE_VALUE
print(f"noise_map.fits shape: {noise_map.shape}")
print(f"  Masked-by-noise pixels (noise = {MASKED_NOISE_VALUE:.0e}): {int(masked.sum())}")
print(f"  Data zeroed at every masked pixel: {bool(np.all(data[masked] == 0.0))}")
print(f"  Physical noise range: {noise_map[~masked].min():.3e} - {noise_map[~masked].max():.3e}")
print(f"  bad_pixel_policy block: {json.dumps(record['bad_pixel_policy'], indent=2)}")

"""
__psf.fits vs psf_full.fits__

Two PSFs ship with every imaging reduction, and the split is about convolution cost versus wing
science:

- `psf.fits` (default 21x21) is the compact kernel for fit convolution. Convolution cost scales
  with kernel area, and 21x21 captures the great majority of the blurring for these instruments
  — the size **PyAutoLens**'s data standards recommend.
- `psf_full.fits` (default 61x61) carries the extended wings, for science where scattered light
  at larger radii matters (bright-deflector contamination, photometry checks). Fit with the
  compact one; consult the full one.

Both are odd-shaped (so the PSF has a centre pixel — an even kernel shifts every model image by
half a pixel) and unit-normalised (so convolution conserves flux).

The invariant behind them is the one to remember: **the delivered PSF is the drizzled PSF**. The
star images it was built from went through the same kernel, pixfrac, scale and rotation as the
science mosaic, so the kernel describes the blurring actually present in `data.fits`. A PSF
processed differently from its data — however carefully made — describes a different image. (The
precedent runs deep: SLACS processed its Tiny Tim PSFs through the identical rectification as its
science frames; see Bolton et al. 2008, https://arxiv.org/abs/0805.1931.)
"""
psf = fits.getdata(out_dir / "psf.fits").astype(float)
psf_full = fits.getdata(out_dir / "psf_full.fits").astype(float)

for name, kernel in (("psf.fits", psf), ("psf_full.fits", psf_full)):
    ny, nx = kernel.shape
    peak = np.unravel_index(np.argmax(kernel), kernel.shape)
    print(
        f"{name}: shape {kernel.shape} (odd: {ny % 2 == 1 and nx % 2 == 1}), "
        f"sum = {kernel.sum():.6f}, peak at {peak} (centre: {(ny // 2, nx // 2)})"
    )

"""
__reduction.json__

The provenance record — the reduction's lab notebook, and the same dictionary `reduce_target`
returned. It answers, block by block, every "what exactly produced this dataset?" question:

- **Envelope** — `written_at` and the software versions (autoreduce and the instrument stack)
  that produced the dataset. Reproducibility starts here: the pipeline is a pure function of
  spec + archive *given* these versions.
- **`target`** — the full `TargetSpec` as declared, every dial included. The spec is the
  reduction; this block is the reduction's definition.
- **`instrument`** — the adapter key (here `acs_wfc`).
- **`acquire`** — which exposures, from where: proposal/exposure identities and `n_exposures`.
  An audit of the stack starts by checking nothing unexpected joined it.
- **`align`** — the astrometric story. The HST pipeline trusts the archive's Gaia-tied a-priori
  WCS and *records* the cross-correlation evidence for that trust rather than silently
  re-registering.
- **`drizzle`** — the combine: the resolved dial set, the weight-uniformity diagnostic (with its
  0.2 rule-of-thumb limit and verdict), the `correlated_noise_factor` R, and the CR method.
- **`noise`** — the recipe string (construction for HST/Keck, propagated-ERR for JWST), R again,
  the exposure time, and the empirical blank-sky RMS closure number.
- **`psf`** — method, stars used, and which drizzle pass fed star finding (`star_source_pass`).
- **`bad_pixel_policy`** — how many pixels the masked-by-noise convention touched, and where.
- **`package`** — the shipped products, `cutout_shape`, `pixel_scale` and `data_units`: the
  three numbers a modeling script needs, read from provenance rather than remembered.

(On other paths, extra blocks appear in the same spirit: `calibrate`/`sky` for Keck's ground
stages, `inject` when synthetic-source injection ran, `frames` for frame products.)
"""
print(f"reduction.json blocks: {sorted(record.keys())}")
print(json.dumps(record["drizzle"], indent=2, default=str)[:1200])

"""
__Frame Products__

Setting `frame_products=True` on a `TargetSpec` (HST, JWST and Keck) adds a second, parallel
output tree: every calibrated exposure chip packaged as its own modeling-ready dataset at
*native* pixel scale, under `output/<target>/frames/`:

- `frames/manifest.json` — the index (schema version 2): the frame cutout shape, native scale,
  data units, CR method, DQ semantics, and — per frame — the target's pixel position with the
  measured registration residuals and a reliability flag.
- `frames/<rootname>_chip<EXTVER>/` — per frame: `data.fits`, `noise_map.fits`, `dq.fits`,
  `cr_mask.fits`, and (with per-frame PSFs) `psf.fits`/`psf_full.fits`.

Why bother? Nothing in a native frame was resampled — so its noise is *uncorrelated* and needs no
Casertano factor, at the price of modeling several frames jointly instead of one mosaic. The
`individual.py` scripts (`scripts/hst_acs/individual.py`, `scripts/jwst_nircam/individual.py`)
walk the trade in full; the Keck variant differs honestly (outlier masks instead of DQ/CR files,
offset-based rather than WCS registration).

The default reduction above did not request frames, so we just report their absence.
"""
frames_manifest = out_dir / "frames" / "manifest.json"
if frames_manifest.exists():
    manifest = json.loads(frames_manifest.read_text())
    print(f"Frame products present: manifest version {manifest.get('manifest_version')}")
else:
    print("No frame products (frame_products=False for this reduction) — see individual.py scripts.")

"""
__The Visibility Triplet__

For ALMA the contract changes shape, because the modeling-ready dataset is not an image at all:
fitting interferometer data in the uv-plane keeps the noise independent and the likelihood
well-defined, so the products are the calibrated visibilities themselves —

- `data.fits` — the complex visibilities as an `(N_vis, 2)` array of (real, imaginary).
- `uv_wavelengths.fits` — the `(N_vis, 2)` baseline coordinates (u, v) in wavelengths.
- `noise_map.fits` — the `(N_vis, 2)` per-visibility sigma, derived as 1/sqrt(weight) — and with
  **no** Casertano factor, because nothing was resampled.

Diagnostic sidecars (antennas, scans, times, frequencies per spectral window) ride along, and the
triplet loads with `al.Interferometer.from_fits`. The `scripts/alma/` folder owns this branch.

__Loading With PyAutoLens__

The contract's whole purpose: the products load directly, with the pixel scale read from the
provenance. The import is guarded — **PyAutoReduce** never depends on the modeling stack.
"""
try:
    import autolens as al
except ImportError:
    al = None
    print(
        "PyAutoLens is not installed (pip install autolens) — skipping the loading demo. "
        "The products above are complete and ready for any machine that has it."
    )

if al is not None:
    dataset = al.Imaging.from_fits(
        data_path=out_dir / "data.fits",
        noise_map_path=out_dir / "noise_map.fits",
        psf_path=out_dir / "psf.fits",
        pixel_scales=record["package"]["pixel_scale"],
    )
    print(
        f"Loaded {TARGET} into PyAutoLens: shape {dataset.data.shape_native} at "
        f"{dataset.pixel_scales} arcsec/pix — no further data preparation needed."
    )

"""
__Wrap Up__

The contract in one breath: four FITS files that load straight into **PyAutoLens**, conventions
(masked-by-noise, odd unit-normalised drizzle-consistent PSFs, strict-coverage cutouts) that
remove every downstream preparation step, and a provenance record that makes the whole reduction
auditable. Every instrument folder in this workspace ships this same contract; only the physics
that fills it changes.

The following locations of the workspace are good places to checkout next:

- `scripts/guides/noise_maps.py`: how `noise_map.fits` is constructed, and the closure checks that validate it.
- `scripts/guides/target_spec.py`: the declaration that produced everything this guide inspected.
- `scripts/hst_acs/individual.py`: the frame-products tree in practice.
- `scripts/alma/start_here.py`: the visibility triplet in practice.
- `autolens_workspace/scripts/imaging/data_preparation/start_here.py`: the modeling-side statement of these same standards.

This guide itself needs no network — it reads whatever `scripts/start_here.py` already produced,
and exits cleanly when that output is absent.
"""
