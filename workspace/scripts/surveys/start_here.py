"""
Start Here: Survey Cutouts
==========================

Not every product **PyAutoReduce** delivers is modeling data. This script demonstrates
the third and simplest of its domains: **survey cutouts** — postage stamps fetched from
the public cutout services of the big ground-based imaging surveys, packaged as colour
context for your lens fields.

Why does a reduction package ship a fetcher? Because knowing what a lens field looks like
in the optical is a routine need — identifying the deflector, spotting neighbouring
galaxies and stars, checking astrometry — and for some lenses it is the *only* optical
view you will ever have. Lensed dusty star-forming galaxies found at submillimetre
wavelengths (the ALMA targets of `scripts/alma/`) often have no optical counterpart at
all: the source is dust-obscured and the deflector may be faint, so a survey cutout of
the field is the entire optical story. In about 10 minutes this script fetches the
SLACS J0008-0004 field from three surveys, walks the honesty machinery that stops these
stamps masquerading as modeling data, and plots the multi-survey contact sheet.

__Contents__

- **The Cutout Domain:** Fetch + package, never reduce — why this is its own branch.
- **Context Not Modeling Data:** What cutouts are for, and emphatically not for.
- **Imports:** Import **PyAutoReduce** and the plotting libraries.
- **Paths:** Anchor every path to the workspace root.
- **The Field SLACS J0008-0004:** The lens field we fetch.
- **The Three Services:** Legacy Surveys DR10, SDSS and Pan-STARRS, and their trade-offs.
- **Fetch The Cutouts:** One `reduce_target` call per service, failures kept per-service.
- **Provenance:** The `products_optional` block — what was NOT produced, and why.
- **Noise From Inverse Variance:** The one service that ships variance, and what we make of it.
- **Postage Stamps:** The multi-survey, multi-band contact sheet.
- **Deferred Extensions:** HSC, unWISE/GALEX, variance elsewhere, approximate PSFs.
- **Wrap Up:** Summary and good places to check out next.

__The Cutout Domain__

DES, SDSS, Pan-STARRS and their peers deliver *pre-reduced* coadds through public cutout
services — mosaics the survey collaborations built with their own calibration, their own
astrometry, their own co-addition, at a quality no external re-reduction of the raw
frames would match. So **PyAutoReduce** treats them as a third adapter domain (`cutout`)
beside imaging and visibility, with a two-stage branch: **fetch + package, never
reduce**. There is no drizzle here, no PSF stage, no noise recipe — a cutout arrives as
the survey made it, gets its provenance stamped, and is written to disk.

The shared machinery is exactly what you know from the other domains: the same
`TargetSpec` (the `survey_bands` dial plus `cutout_shape` for the stamp size), the same
one-call `reduce_target`, the same `reduction.json` provenance.

__Context Not Modeling Data__

The design's loudest rule. A survey cutout is **for**:

- colour/context imaging of a lens field — which object is the deflector, what
  neighbours and stars sit nearby, does the field match your pointing;
- the optical view of targets whose *modeling* data lives elsewhere — especially ALMA
  lens fields with no optical counterpart in the modeling data at all;
- figures: the "here is the field" panel of a paper or proposal.

It is emphatically **not** a modeling input, for reasons the products themselves encode:

- **No PSF.** None of these services ships one, and lens modeling without a PSF model is
  not lens modeling. No `psf.fits` is ever written by this branch.
- **Mostly no noise map.** Only one of the three services exposes per-pixel variance;
  the others deliver data only. Without an RMS map there is no likelihood.
- **Uncharacterised correlations.** Survey coadds resample and stack with kernels these
  services do not document per-cutout, so even where variance exists the pixel-to-pixel
  correlations are uncharacterised — fine for context, unquantified for chi-squared.

The provenance block you will inspect below (`products_optional`) states per product
what was not produced and why, so no downstream tool can mistake a stamp for an
`al.Imaging` dataset by accident.

__Imports__
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from autoreduce import TargetSpec, reduce_target

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the
folder containing `scripts/`). **PyAutoReduce** requires absolute paths throughout the
workspace, and although this branch never changes directory, we keep the convention.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"
OUTPUT_ROOT = WORKSPACE / "output"

"""
__The Field SLACS J0008-0004__

We fetch the field of SLACS J0008-0004, one of the strong lenses of the Sloan Lens ACS
Survey (Bolton et al. 2008, https://arxiv.org/abs/0805.1931) — the same target that
anchors the HST/ACS reduction in `scripts/hst_acs/start_here.py`. That makes it the
perfect demonstration of the intended division of labour: the HST reduction produces the
modeling dataset; the survey cutouts below produce the wide-field colour context around
it. (For an ALMA field like G09v1.40 the cutout might be the *only* optical view — the
submillimetre selection of Negrello et al. 2010, Science 330, 800, cares nothing for
optical brightness.)
"""
NAME = "slacs0008-0004"
RA, DEC = 2.012333, -0.068944

"""
__The Three Services__

Three adapters, three public endpoints, one field:

- `legacy_surveys` — the DESI Legacy Imaging Surveys DR10 cutout service
  (legacysurvey.org), DECam g/r/z at 0.262"/pixel. **The one service that ships
  variance**: the same request can append an inverse-variance HDU, so Legacy cutouts get
  a real `noise_map.fits`. One more thing makes this the workhorse: **DR10 includes the
  full DECam/DES footprint** (and far more southern sky), so Legacy is the DES door —
  there is no separate DES adapter because none is needed.
- `sdss` — SDSS frames via astroquery, g/r/i at 0.396"/pixel. The shallowest and
  coarsest of the three, but the survey that found the SLACS lenses in the first place,
  and coverage where the others may lack it. Data only.
- `panstarrs` — Pan-STARRS PS1 stack cutouts via the STScI fitscut service
  (ps1images.stsci.edu), g/r/i at 0.25"/pixel. Northern-sky coverage to dec > -30 deg.
  Data only.

(HSC — deeper than all three — is deliberately absent: its cutout service sits behind a
credential-gated account, deferred until a real need justifies the auth plumbing. See the
deferred list at the end.)

__Fetch The Cutouts__

One `reduce_target` per service. Note the error handling: each service gets its own
try/except, because these are three independent external endpoints — a service outage or
a coverage gap in one must yield a per-service verdict, not one crash that loses the
other two. Each service's products land under its own output folder so the three
`reduction.json` records stay separate.
"""
SERVICES = ("legacy_surveys", "sdss", "panstarrs")

print(f"Fetching {NAME} cutouts from {len(SERVICES)} services (network, ~seconds each)...")

records, stamps = {}, {}
for key in SERVICES:
    print(f"\n== {key} ==")
    spec = TargetSpec(
        name=NAME,                  # products land at output/surveys/<service>/<name>/
        ra=RA,                      # field centre (deg)
        dec=DEC,
        instrument=key,             # selects the cutout-domain adapter for this service
        cutout_shape=(101, 101),    # stamp size in native survey pixels (101 px: ~26" Legacy, ~40" SDSS, ~25" PS1)
    )
    try:
        record = reduce_target(
            spec,
            cache_root=CACHE_ROOT,                      # unused by this branch (cutouts are fetched per run)
            output_root=OUTPUT_ROOT / "surveys" / key,  # one output tree per service
        )
    except Exception as error:
        print(f"    FAILED: {error}")
        records[key] = None
        continue
    records[key] = record
    print(f"    bands delivered: {record['acquire']['bands_delivered']}")
    print(f"    products       : {record['package']['products']}")
    print(f"    pixel scale    : {record['package']['pixel_scale']} arcsec/pixel")

    out_dir = OUTPUT_ROOT / "surveys" / key / NAME
    for product in record["package"]["products"]:
        if Path(product).name != "data.fits":
            continue
        band = Path(product).parent.name
        stamps[(key, band)] = fits.getdata(out_dir / product)

if not any(records.values()):
    print("\nNo survey service could be reached — check your network and re-run.")
    sys.exit(0)

"""
__Provenance__

Open any of the three `reduction.json` records and the honesty machinery is right there:
the `package.products_optional` block states, product by product, what was **not**
produced and why. For the data-only services it reads "not produced — service ships no
variance product"; for every service the PSF line records that colour-context products
carry no PSF, by design. This block is the contract that keeps a cutout from ever
pretending to be a modeling-ready dataset — a downstream loader that wants `al.Imaging`
inputs should find `psf.fits` absent *and* the record saying so on purpose.
"""
for key, record in records.items():
    if record is None:
        continue
    print(f"\n{key} products_optional:")
    print(json.dumps(record["package"]["products_optional"], indent=2))

"""
__Noise From Inverse Variance__

The Legacy service's `&invvar` option returns an inverse-variance map alongside the data
in the same request, and the pipeline converts it to the RMS convention the rest of the
ecosystem speaks: sigma = 1 / sqrt(invvar), with non-positive or non-finite inverse
variances left as NaN rather than invented. So Legacy stamps come with a real
`noise_map.fits` — useful for judging depth and spotting masked regions in the coadd.
The correlation caveat from the top of the script still applies: a coadd's
pixel-to-pixel noise correlations are uncharacterised here, which is (one of the reasons)
why even a Legacy cutout is context, not a modeling input.
"""
legacy_record = records.get("legacy_surveys")
if legacy_record is not None:
    for product in legacy_record["package"]["products"]:
        if Path(product).name != "noise_map.fits":
            continue
        band = Path(product).parent.name
        noise = fits.getdata(OUTPUT_ROOT / "surveys" / "legacy_surveys" / NAME / product)
        finite = noise[np.isfinite(noise)]
        print(
            f"legacy {band}-band noise map: median sigma = {np.median(finite):.4f} "
            f"(nanomaggies), {100.0 * (1.0 - finite.size / noise.size):.1f}% NaN "
            f"(no-coverage / masked coadd pixels)"
        )

"""
__Postage Stamps__

The contact sheet: one row per service, one column per band, arcsinh-scaled so both the
lens galaxy's core and the faint outskirts are visible. This is the figure this domain
exists to make — the field at a glance, across surveys and bands, with the survey depth
differences plain to see.
"""
plot_dir = OUTPUT_ROOT / "surveys" / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

available = [key for key in SERVICES if records.get(key) is not None]
n_cols = max(
    len([band for (k, band) in stamps if k == key]) for key in available
)
fig, axes = plt.subplots(
    len(available), n_cols, figsize=(3 * n_cols, 3 * len(available)), squeeze=False
)
for row, key in enumerate(available):
    bands = sorted(band for (k, band) in stamps if k == key)
    for col in range(n_cols):
        ax = axes[row][col]
        ax.set_xticks([])
        ax.set_yticks([])
        if col >= len(bands):
            ax.set_visible(False)
            continue
        band = bands[col]
        data = stamps[(key, band)]
        finite = data[np.isfinite(data)]
        scale = np.std(finite) if finite.size else 1.0
        ax.imshow(
            np.arcsinh(np.nan_to_num(data) / (scale if scale > 0 else 1.0)),
            origin="lower",
            cmap="gray_r",
        )
        ax.set_title(f"{key} {band}", fontsize=10)
fig.suptitle(f"{NAME}: survey colour context (101 px stamps, native scales)")
stamps_png = plot_dir / f"{NAME}_postage_stamps.png"
fig.savefig(stamps_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved postage-stamp contact sheet to {stamps_png.resolve()}")

"""
__Deferred Extensions__

Recorded, assessed, and deliberately not built yet:

- **HSC** — deeper than everything above, but its cutout service requires a credentialed
  account; deferred until a real need justifies the auth plumbing.
- **unWISE and GALEX** — served by the *same* Legacy viewer endpoint via a layer
  parameter, making IR and UV context the cheapest future extension on the books.
- **SDSS / Pan-STARRS variance** — both surveys expose the ingredients (frame metadata,
  weight files); wired up only if someone actually needs ground-based noise maps.
- **Approximate survey PSFs** — Legacy catalogs record per-brick seeing FWHM values; a
  Gaussian-FWHM kernel, clearly flagged approximate in provenance, is the recorded
  follow-up if colour context ever needs even a rough PSF. Until then: no PSF, ever,
  from this branch.

__Wrap Up__

You fetched one lens field from three survey cutout services through the same
`TargetSpec` + `reduce_target` idiom as every real reduction, saw the `products_optional`
provenance that keeps context data honest, got a real noise map from the one service
that ships variance, and made the multi-survey contact sheet.

Good places to checkout next:

- `scripts/hst_acs/start_here.py` — the *modeling* dataset for this same field, reduced
  from HST/ACS exposures.
- `scripts/alma/start_here.py` — the ALMA lens fields for which a survey cutout is often
  the only optical view.
- `scripts/guides/output_contract.py` — what a real modeling-ready product set contains,
  i.e. everything a cutout deliberately is not.
"""

"""
__Env__ (Developer Only)

ENV: network
"""
