"""
Start Here: ALMA Visibilities
=============================

ALMA is the premier instrument for observing strongly lensed dusty star-forming galaxies
(DSFGs): submillimetre-bright sources whose rest-frame far-infrared emission ALMA resolves
into Einstein rings and arcs at resolutions down to tens of milliarcseconds. The canonical
example is SDP.81, whose ~30 mas Einstein ring from the ALMA Long Baseline Campaign
(ALMA Partnership 2015, https://arxiv.org/abs/1503.02652) became the community's benchmark
lens dataset, and samples of tens of lensed DSFGs have been modeled from Herschel and SPT
selections (Bussmann et al. 2015, https://arxiv.org/abs/1504.05256; Spilker et al. 2016,
https://arxiv.org/abs/1604.05723).

This script runs the **PyAutoReduce** visibility branch end-to-end on the validation anchor
G09v1.40 (ALMA project 2016.1.00282.S), producing the `al.Interferometer.from_fits` product
triplet that **PyAutoLens** models directly in the uv plane. In about 20 minutes of reading
(and a few minutes of compute, given a calibrated measurement set on disk) you will have a
modeling-ready interferometric dataset and understand every choice that produced it.

Unlike the imaging instruments (HST, JWST, Keck), the ALMA reduction's product is not an
image: it is the visibilities themselves. The first two sections explain why that is the
right product for lens modeling, and why obtaining the input data involves one manual step
that **PyAutoReduce** is deliberately honest about.

__Contents__

- **Why Visibilities:** Why lens modeling fits visibilities rather than CLEAN images.
- **The Archive Reality:** The ALMA archive does not serve calibrated visibilities — what you actually do.
- **Imports:** Import **PyAutoReduce** and the other libraries the script needs.
- **Paths:** Anchor every path to the workspace root.
- **The Target G09v1.40:** The Herschel-selected lensed DSFG this reduction is validated on.
- **Calibrated Measurement Set:** Locate the calibrated MS directory (or print acquisition guidance and exit).
- **Target Spec:** Build the `TargetSpec` with every ALMA dial explained.
- **The Visibility Branch:** The four stages: split, extract, assemble, package.
- **Run The Reduction:** Call `reduce_target` and let the branch run.
- **Provenance:** Walk the returned record: acquisition, splits, assembly counts, products.
- **Stokes I And The Noise Map:** The polarization combine formula and sigma = 1/sqrt(WEIGHT).
- **The Product Triplet:** The three (Nvis, 2) FITS files and their diagnostic sidecars.
- **UV Coverage:** Plot the uv-plane sampling of the combined dataset.
- **Load In PyAutoLens:** `al.Interferometer.from_fits` plus a dirty-image sanity check.
- **Wrap Up:** Summary and good places to check out next.

__Why Visibilities__

An interferometer never measures an image. Each antenna pair measures a visibility — the
complex cross-correlation of the incoming wavefront — which, by the van Cittert-Zernike
theorem, samples one Fourier component of the sky brightness at the spatial frequency
(u, v) set by the projected baseline in wavelengths. Earth rotation sweeps each baseline
through an elliptical uv track, but the sampling is always incomplete: the "image" you see
in an ALMA press release is the sky convolved with the dirty beam, deconvolved by the
non-linear CLEAN algorithm (the textbook treatment is Thompson, Moran & Swenson 2017).

For lens modeling that matters in two ways:

- **CLEAN image noise is correlated between pixels.** The gridding and deconvolution that
  produce a CLEAN map couple neighbouring pixels, so a pixel-by-pixel chi-squared against a
  CLEAN image uses a wrong (and practically intractable) likelihood.

- **CLEAN is non-linear.** Its artefacts depend on the source structure itself, so there is
  no clean way to forward-model them.

The visibilities have neither problem: each is an independent measurement with (near)
Gaussian noise, so the likelihood of a lens model — Fourier transform the model image,
compare to the data at the measured (u, v) points — is exact and well defined. This is why
the field moved to fitting visibilities directly: Rybak et al. 2015
(https://arxiv.org/abs/1503.02025) reconstructed SDP.81's source at sub-50 pc resolution in
visibility space, Dye et al. 2018 (https://arxiv.org/abs/1705.05413) compared image-plane
and visibility-plane fits of the same data explicitly, and Powell et al. 2021
(https://arxiv.org/abs/2005.03609) showed the full ~10^8-10^9 visibility problem is
tractable with NUFFT methods — the same transform **PyAutoLens** uses. The flagship science
this enables includes the 6.9-sigma detection of a ~10^9 solar-mass subhalo in SDP.81 from
its visibilities alone (Hezaveh et al. 2016, https://arxiv.org/abs/1601.01388).

**PyAutoReduce** therefore never makes an image from ALMA data. Its product is the
visibility triplet **PyAutoLens** fits directly; CLEAN/tclean maps remain useful as
*diagnostics* (you will make a dirty image at the end of this script to sanity-check the
data), never as the science product.

__The Archive Reality__

Here is the honest part. The ALMA Science Archive does **not** serve calibrated
visibilities as a plain download. What you can download anonymously are the raw data
(ASDM format) and the calibration/QA2 product tarballs. Turning those into a calibrated
measurement set (MS) requires running the observatory's `scriptForPI.py` restore script
*inside the CASA version that originally processed the data* — the tarball names record
which version — because the calibration tables are not portable across CASA releases
(see https://almascience.eso.org/processing/science-pipeline and the ALMA Knowledgebase
article "How do I obtain a file of calibrated visibilities"). Version-matching the restore
is the classic ALMA footgun, and automating it is deliberately out of scope for
**PyAutoReduce** today.

So the canonical pipeline input is a **calibrated MS directory**, however obtained:

- an ARC delivery (the EU ARC "CalMS" service, EA/NA helpdesk requests, or NRAO's SRDP
  service for Cycle 5+ pipeline-calibrated data), or
- your own local `scriptForPI.py` restore of the downloaded tarballs, at the matching
  CASA version.

**PyAutoReduce** automates the archive *download* (`query_project`,
`download_product_tarballs`) and, if a downloaded tarball happens to contain a calibrated
MS (some ARC deliveries do), extracts it. When the tarballs contain only restore inputs —
the standard case — the pipeline raises loudly with step-by-step restore guidance rather
than pretending it produced calibrated data. You will see that guidance printed below if
no MS is on disk.

__Imports__

We import **PyAutoReduce**'s two-name public API (`TargetSpec`, `reduce_target`) plus the
ALMA acquisition helpers used for the guidance path. The heavy CASA dependencies
(`casatools`, `casatasks` — the pip-installable modular CASA) are imported by the pipeline
inside its stage functions, but this script checks for them up front so it can fail with a
friendly message instead of a mid-reduction traceback.
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoreduce import TargetSpec, reduce_target
from autoreduce.acquire import alma as alma_acquire

try:
    import casatools  # noqa: F401  (modular CASA: the `table` tool the extract stage uses)
    import casatasks  # noqa: F401  (modular CASA: the `split` task the split stage uses)
except ImportError:
    print(
        "Modular CASA is not installed. The ALMA visibility branch needs the "
        "pip-installable CASA packages:\n\n"
        "    pip install casatools casatasks\n\n"
        "No monolithic CASA installation is required — see scripts/alma/step_by_step.py "
        "for why the modular route is the right one. Exiting cleanly."
    )
    sys.exit(0)

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the
folder containing `scripts/`). **PyAutoReduce** requires absolute paths: CASA tasks write
scratch directories and logs relative to the working directory, and the pipeline changes
directory internally, so relative paths would break.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"      # archive tarball downloads (re-used across runs)
OUTPUT_ROOT = WORKSPACE / "output"    # reduced datasets, one folder per target

"""
__The Target G09v1.40__

G09v1.40 (H-ATLAS J085358.9+015537, z ~ 2.09) is a strongly lensed dusty star-forming
galaxy from the Herschel-ATLAS survey. It was found by the submillimetre flux-selection
technique of Negrello et al. 2010 (Science 330, 800): at 500 micron, essentially every
extragalactic source brighter than ~100 mJy is either a blazar or a gravitational lens,
because the steep DSFG number counts make unlensed sources that bright vanishingly rare.
H-ATLAS turned this into candidate lists of ~80 lensed DSFGs (Negrello et al. 2017,
MNRAS 465, 3558), with uv-plane lens models following (see e.g. Enia et al. 2018,
MNRAS 475, 3467, from SMA visibilities).

G09v1.40 itself was observed in ALMA Band 6 under project 2016.1.00282.S and modeled with
those data by Butler et al. 2021 (https://arxiv.org/abs/2104.10077), including OH+ and
CO(9-8) lines alongside the continuum this script reduces.

One caution when reading the literature on this source: Yang et al. 2019 (A&A 624, A138)
present sub-kpc CO and H2O lens modeling of **G09v1.97** — a *different* H-ATLAS lens with
an easily-confused name. Do not carry numbers between the two.
"""
NAME = "alma_g09v140"
PROJECT = "2016.1.00282.S"                              # the ALMA project code
FIELD = "G09v1.40"                                      # the science field name inside the MS
UIDS = ("A002_Xb9b1b9_X3046", "A002_Xb99cbd_X2456")     # the two execution blocks (one MS each)
SPWS = ("1", "2")                                       # the line-free continuum spectral windows
WIDTH = 240                                             # channels averaged per output channel
RA, DEC = 133.49542, 1.59367                            # H-ATLAS J085358.9+015537

"""
__Calibrated Measurement Set__

The reduction consumes a directory of calibrated measurement sets, one per execution
block, in the standard delivered layout `uid___<uid>.ms.split.cal`. Point the path below
at yours (or set the `AUTOREDUCE_ALMA_MS_DIR` environment variable).

If no MS directory is present, we print the acquisition guidance — including the exact
archive-download idiom and the restore instructions — and exit cleanly rather than crash.
The download idiom, for reference (network + ~tens of GB; run it deliberately):

    from autoreduce.acquire.alma import query_project, download_product_tarballs

    query_project("2016.1.00282.S")           # one archive row per member OUS product
    download_product_tarballs(               # the scriptForPI restore inputs
        "2016.1.00282.S", CACHE_ROOT / "alma_g09v140" / "tarballs"
    )

`reduce_target` itself runs the same download automatically when you give it
`alma_project_code` instead of `alma_ms_dir` — and raises with the restore guidance if the
tarballs contain no calibrated MS, which for standard product tarballs they will not.
"""
ALMA_MS_DIR = Path(
    os.environ.get(
        "AUTOREDUCE_ALMA_MS_DIR", WORKSPACE / "dataset" / "alma" / "g09v140_calibrated"
    )
)

if not ALMA_MS_DIR.is_dir():
    print(
        f"No calibrated measurement-set directory found at:\n\n    {ALMA_MS_DIR}\n\n"
        "The ALMA archive does not serve calibrated visibilities directly (see the\n"
        "'Archive Reality' section above). To run this script for real, obtain the\n"
        "calibrated MS for project 2016.1.00282.S via an ARC delivery (EU CalMS /\n"
        "NRAO SRDP) or a local scriptForPI.py restore at the matching CASA version,\n"
        "then place (or symlink) the uid___<uid>.ms.split.cal directories at the path\n"
        "above, or set AUTOREDUCE_ALMA_MS_DIR to their location.\n"
    )
    print("The pipeline's own guidance message for this project reads:\n")
    print(alma_acquire.restore_guidance(PROJECT, CACHE_ROOT / NAME / "tarballs"))
    print("\nExiting cleanly — nothing was downloaded or written.")
    sys.exit(0)

try:
    ms_paths = alma_acquire.resolve_calibrated_ms(ALMA_MS_DIR, UIDS)
except FileNotFoundError as error:
    print(f"{error}\n\nExiting cleanly — fix the MS directory layout and re-run.")
    sys.exit(0)

print(f"Calibrated measurement sets resolved in {ALMA_MS_DIR}:")
for path in ms_paths:
    print(f"    {path.name}")

"""
__Target Spec__

A **PyAutoReduce** reduction is declared, not scripted: you build one frozen `TargetSpec`
and hand it to one function. For the visibility domain the imaging dials (cutout shape,
drizzle parameters, PSF shapes) are ignored; the ALMA dials below are the whole interface.

The `alma_width=240` choice deserves a word: each of these spectral windows has 240
channels, so averaging 240 channels per output channel collapses each spw to a single
continuum channel. That is the continuum default (`alma_width=0` would auto-collapse by
reading each spw's channel count), and it is what cuts the visibility count to something a
lens-modeling likelihood evaluates quickly — the classic averaging step of published
practice (Hezaveh et al. 2016, https://arxiv.org/abs/1601.01388, section 2). Averaging is
not free: it smears the response to off-centre emission. `step_by_step.py` computes the
bandwidth-smearing limit for this dataset explicitly.
"""
spec = TargetSpec(
    name=NAME,                          # products land at output/<name>/
    ra=RA,                              # target RA (deg) — provenance for the visibility branch
    dec=DEC,                            # target Dec (deg)
    instrument="alma",                  # selects the visibility-domain adapter and branch
    alma_uids=UIDS,                     # execution blocks to reduce (one calibrated MS each)
    alma_field=FIELD,                   # science field to isolate from each MS (they also hold calibrators)
    alma_spws=SPWS,                     # continuum spectral windows to extract
    alma_width=WIDTH,                   # channel-averaging width (240 = collapse these 240-channel spws)
    alma_ms_dir=str(ALMA_MS_DIR),       # the calibrated-MS directory (the canonical input)
    alma_project_code=PROJECT,          # recorded in provenance; drives archive download if ms_dir unset
)

"""
__The Visibility Branch__

`reduce_target` dispatches on the instrument adapter's domain. For `instrument="alma"`
that is the visibility branch — four stages, none shared with the imaging pipeline:

- **split** — `casatasks.split`, run twice per execution block: first to isolate the
  science field (each MS also contains the bandpass/phase/flux calibrators), then per
  spectral window to average channels by `width`. Flagged rows are dropped at this point
  (`keepflags=False`), so no zero-weight placeholder rows survive into the products.

- **extract** — `casatools.table` reads of the split MS columns: DATA (the complex
  visibilities), UVW (baselines in metres), WEIGHT (the noise bookkeeping), CHAN_FREQ,
  and the ANTENNA1/ANTENNA2/TIME/SCAN_NUMBER diagnostics.

- **assemble** — pure numpy: convert UVW from metres to wavelengths (u * frequency / c),
  combine the two polarizations into Stokes I, form the noise map from the weights, and
  concatenate all (execution block, spw) pieces into the final arrays.

- **package** — write the `(Nvis, 2)` product triplet, the per-block diagnostic sidecars
  and `reduction.json`.

Both split steps are idempotent — an existing output MS is reused — so re-running this
script after the first reduction costs seconds, not minutes.

__Run The Reduction__

The first run splits each execution block's MS twice with CASA (a few minutes per MS,
depending on its size and your disk), then extraction and assembly take seconds.
Intermediate measurement sets land under `output/alma_g09v140/work/` and are reused on
re-runs.
"""
print(
    f"\nRunning the visibility branch on {len(UIDS)} execution blocks x "
    f"{len(SPWS)} spws (first run: a few minutes of CASA split per MS; "
    f"re-runs reuse the split MS)..."
)

record = reduce_target(
    spec,
    cache_root=CACHE_ROOT,      # archive tarballs would cache here (unused with a local ms_dir)
    output_root=OUTPUT_ROOT,    # products land at output/alma_g09v140/
)

out_dir = OUTPUT_ROOT / NAME
print(f"\nReduction complete. Products in {out_dir}")

"""
__Provenance__

The returned record (also written to `reduction.json` alongside the products) documents
every stage. Walk the highlights:

- `acquire`: which measurement sets were consumed and from where (`source: "local"` here;
  an archive-download run records tarball checksums instead).
- `split`: one block per (execution block, spw) with the resolved channel-averaging width.
- `assemble`: the visibility counts — including, importantly,
  `n_visibilities_dropped_invalid`: rows where *neither* polarization carried positive
  weight and finite data. **PyAutoReduce** drops these loudly and counts them, rather than
  silently zero-filling them (a zero-weight visibility written as data would poison the
  likelihood downstream).
- `package`: the product filenames and the contract they satisfy.
"""
print("\nacquire:")
print(f"    source            : {record['acquire']['source']}")
print(f"    measurement sets  : {record['acquire']['measurement_sets']}")

print("split:")
for block in record["split"]["blocks"]:
    print(
        f"    uid {block['uid']}  spw {block['spw']}  width {block['width']}"
    )

print("assemble:")
print(f"    n_visibilities    : {record['assemble']['n_visibilities']}")
for label, prov in record["assemble"]["blocks"].items():
    print(
        f"    {label}: {prov['n_visibilities']} visibilities "
        f"({prov['n_rows']} rows x {prov['n_channels']} channel(s), "
        f"{prov['n_visibilities_dropped_invalid']} dropped invalid, "
        f"{prov['n_antennas']} antennas, {prov['n_scans']} scans)"
    )

print("package:")
print(f"    products          : {record['package']['products']}")
print(f"    contract          : {record['package']['contract']}")

"""
__Stokes I And The Noise Map__

The split MS carries two parallel-hand polarizations (XX, YY). Continuum lens modeling
fits total intensity, so the assemble stage forms the weighted Stokes-I average per
visibility, with the MS weights (weight = 1/sigma^2 per complex visibility):

    I       = (w_xx * XX + w_yy * YY) / (w_xx + w_yy)
    sigma_I = 1 / sqrt(w_xx + w_yy)

The same sigma applies to the real and imaginary parts — the MS weight is per complex
visibility — which is exactly the `(Nvis, 2)` noise-map shape `al.Interferometer` expects.
Stacking the two polarizations as independent visibilities was considered and rejected:
it doubles the visibility count (and hence the NUFFT cost of every likelihood evaluation)
for zero information gain over the weighted average.

Two things are deliberately *absent* from this noise map:

- **No Casertano correlated-noise factor.** The imaging pipelines multiply their RMS maps
  by the Casertano R factor because drizzling resamples pixels and correlates their noise.
  Visibilities are not resampled pixels — each is an independent sample — so no such
  correction exists or is applied here.

- **No independent noise estimate.** sigma = 1/sqrt(WEIGHT) trusts the weight column of
  the calibrated MS. Whether that trust is justified depends on the CASA version history
  of your data and whether `statwt` was run — the single most important data-quality
  question for uv-plane modeling, treated in depth (with a verification you can run on
  your own data) in `step_by_step.py`.
"""

"""
__The Product Triplet__

Three FITS files, each a float64 array of shape (Nvis, 2), matching what
`al.Interferometer.from_fits` loads:

- `data.fits` — the visibilities: column 0 the real part, column 1 the imaginary part (Jy).
- `uv_wavelengths.fits` — the (u, v) coordinates of each visibility, in wavelengths.
- `noise_map.fits` — sigma on the real and imaginary parts (equal by convention, above).

Alongside them, per-(execution block, spw) diagnostic sidecars — `antennas_*.fits`,
`scans_*.fits`, `times_*.fits`, `frequencies_*.fits` — record which baselines, scans,
timestamps and channel frequencies each block contributed. They are not modeling inputs;
they are the paper trail for debugging (e.g. tracing a bad scan or antenna back through
the assembly).
"""
from astropy.io import fits

visibilities = fits.getdata(out_dir / "data.fits")
uv_wavelengths = fits.getdata(out_dir / "uv_wavelengths.fits")
noise_map = fits.getdata(out_dir / "noise_map.fits")

print(f"\ndata.fits           : {visibilities.shape} (real / imaginary, Jy)")
print(f"uv_wavelengths.fits : {uv_wavelengths.shape} (u / v, wavelengths)")
print(f"noise_map.fits      : {noise_map.shape} (sigma real / sigma imaginary)")
print(
    f"median |V| = {np.median(np.hypot(visibilities[:, 0], visibilities[:, 1])):.4f} Jy, "
    f"median sigma = {np.median(noise_map[:, 0]):.4f} Jy"
)

"""
__UV Coverage__

The uv-plane sampling determines what the dataset can constrain: the longest baselines
set the resolution, the shortest set the largest recoverable scale, and the gaps are where
the lens model is doing pure interpolation. Plotting it is the interferometric equivalent
of glancing at your image before fitting it. (Each visibility also implies its complex
conjugate at (-u, -v); we plot both to show the familiar symmetric coverage.)
"""
plot_dir = out_dir / "plots"
plot_dir.mkdir(exist_ok=True)

u_klambda = uv_wavelengths[:, 0] / 1e3
v_klambda = uv_wavelengths[:, 1] / 1e3

plt.figure(figsize=(6, 6))
plt.scatter(u_klambda, v_klambda, s=0.5, lw=0, alpha=0.5, label="sampled")
plt.scatter(-u_klambda, -v_klambda, s=0.5, lw=0, alpha=0.5, label="conjugate")
plt.xlabel(r"u [k$\lambda$]")
plt.ylabel(r"v [k$\lambda$]")
plt.title(f"{FIELD}: uv coverage ({visibilities.shape[0]} visibilities)")
plt.gca().set_aspect("equal")
plt.legend(markerscale=10, loc="upper right")
uv_png = plot_dir / "uv_coverage.png"
plt.savefig(uv_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved uv-coverage plot to {uv_png.resolve()}")

"""
__Load In PyAutoLens__

The consumer-side round trip. **PyAutoReduce** never imports **PyAutoLens** (a hard
boundary — the reducer stays releasable on its own), so this load is the workspace's job.
Two modeling-side choices enter here that are *not* reduction products:

- `real_space_mask` defines the sky region the source and lens are reconstructed within —
  its pixel scale and extent are yours to choose against the uv coverage above.
- `transformer_class` selects the Fourier transform: `al.TransformerNUFFT` scales to large
  visibility counts (the Powell et al. 2021 lesson, https://arxiv.org/abs/2005.03609).

The dirty image — the direct inverse transform of the visibilities, no deconvolution — is
the honest quick-look: if the lensed source is real and bright, its peak stands far above
the residual rms. This mirrors the pipeline's own validation check on this dataset.
"""
try:
    import autolens as al
except ImportError:
    al = None
    print(
        "\nPyAutoLens is not installed, so the final loading check is skipped "
        "(pip install autolens). The reduction itself is complete — the product "
        "triplet above is ready for any uv-plane modeling tool."
    )

if al is not None:
    real_space_mask = al.Mask2D.circular(
        shape_native=(256, 256),    # real-space grid the source/lens are evaluated on
        pixel_scales=0.05,          # arcsec / pixel — chosen for the ~0.1" scales the uv coverage reaches
        radius=4.0,                 # arcsec — comfortably contains the ~1.5" diameter ring
    )

    dataset = al.Interferometer.from_fits(
        data_path=out_dir / "data.fits",
        noise_map_path=out_dir / "noise_map.fits",
        uv_wavelengths_path=out_dir / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerNUFFT,
    )

    dirty = np.asarray(dataset.dirty_image.native)
    peak_over_rms = float(np.max(np.abs(dirty)) / np.std(dirty))
    print(
        f"\nLoaded {dataset.data.shape[0]} visibilities into al.Interferometer; "
        f"dirty-image peak/rms = {peak_over_rms:.1f} (expect >> 1 for a detected ring)."
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(dirty, origin="lower", cmap="magma")
    plt.colorbar(label="dirty-image intensity")
    plt.title(f"{FIELD}: dirty image (peak/rms = {peak_over_rms:.1f})")
    dirty_png = plot_dir / "dirty_image.png"
    plt.savefig(dirty_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved dirty-image plot to {dirty_png.resolve()}")

"""
__Wrap Up__

You reduced ALMA Band 6 observations of the lensed DSFG G09v1.40 from a calibrated
measurement set to a modeling-ready visibility dataset: field-split and channel-averaged
with CASA, extracted to numpy, combined to Stokes I with weights-derived noise, packaged
as the `(Nvis, 2)` triplet, and verified by a dirty-image quick look through
**PyAutoLens**.

Good places to checkout next:

- `scripts/alma/step_by_step.py` — every stage of the chain by hand: MS anatomy, the
  smearing limit on channel averaging, the weight-recalibration story and how to verify
  your weights against the visibility scatter.
- `scripts/alma/simulator.py` — simulate a realistic ALMA observation of a synthetic
  source with CASA's simobserve and push it through the identical chain.
- `scripts/surveys/start_here.py` — fetch optical survey cutouts for colour context on
  lens fields like this one, whose modeling data has no optical counterpart.
- `autolens_workspace/scripts/interferometer/` — model this dataset in the uv plane with
  **PyAutoLens**.
"""

"""
__Env__ (Developer Only)

ENV: network
"""
