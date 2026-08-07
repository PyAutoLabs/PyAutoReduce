"""
HST ACS: Step By Step
=====================

`start_here.py` reduced the SLACS lens SDSS J0008-0004 with one call. This script re-runs the
same reduction and takes it apart stage by stage: what the archive actually serves, how the
calibration reference files arrive, how the exposure cache makes re-runs cheap and offline,
what AstroDrizzle is told and why, where the noise-map numbers come from, and what evidence
each stage leaves behind in `reduction.json` and the `work/` directory.

**PyAutoReduce**'s pipeline stages are internal by design — you declare a `TargetSpec` and call
`reduce_target`, and the stage functions are not public API. So this script teaches each stage in
three honest ways: (a) explaining what happens, with links to the STScI documentation and the
papers behind each choice; (b) reading the evidence out of the provenance record and work
directory after the run; and (c) demonstrating the public helper functions (`query_exposures`,
`casertano_r`, `noise_map_from`, `weight_uniformity`, `registered_ratios`, ...) standalone on the
run's own products, so you can see the machinery with your own hands.

Expect the same runtime as `start_here.py` on a cold cache (~10-30 minutes); if you ran that
script first, the cache is warm and this one takes a few minutes.

__Contents__

- **Imports:** Import **PyAutoReduce**, its public helpers and the other libraries we need.
- **Paths:** Anchor the cache and output locations to the workspace root.
- **The Instrument Adapter:** Where ACS/WFC's identity lives — scales, products, defaults.
- **MAST Query Anatomy:** The `astroquery.mast` idiom, the product zoo, and HAP exclusion.
- **CRDS Reference Files:** Best-reference syncing and the offline dial.
- **The Exposure Cache:** Transient full-frame storage, the manifest, and eviction.
- **Run The Pipeline:** The reduction itself, so the later sections have evidence to read.
- **Footprint And Quality Filtering:** Which exposures were kept, skipped and rejected.
- **WCS And Alignment:** The a-priori Gaia-tied WCS, and the honest TweakReg story.
- **The Drizzle Stage:** Every AstroDrizzle keyword the pipeline set, one by one.
- **Sky Subtraction:** What `skymethod='globalmin+match'` does and why it suits lens fields.
- **Cosmic-Ray Flagging:** The median-combine flow, and the flux-loss caveat.
- **Noise Construction By Hand:** `casertano_r` + `noise_map_from` standalone on the mosaic.
- **The PSF Stage:** A summary, deferring to `psf.py` for the full story.
- **Packaging And The Strict Cutout:** Cutout2D strict mode, bad-pixel policy, uniformity.
- **Parity:** `registered_ratios` against a reference reduction, when one exists.
- **Wrap Up:** Where to go next in the workspace.

__Imports__

Alongside the two core names, we import the public helpers this script demonstrates standalone:
the instrument adapter registry, the MAST query helper, the exposure cache, the noise
constructors and the drizzle diagnostic.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from autoreduce import TargetSpec, reduce_target, instruments
from autoreduce.acquire.cache import ExposureCache
from autoreduce.acquire.mast import query_exposures
from autoreduce.drizzle.diagnostics import weight_uniformity, WEIGHT_UNIFORMITY_LIMIT
from autoreduce.noise.rms import casertano_r, noise_map_from, empirical_background_rms

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the folder
containing `scripts/`). **PyAutoReduce** requires absolute paths: its drizzle step changes the
working directory internally, so relative paths would break.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"      # Downloaded exposures + CRDS references (re-used across runs).
OUTPUT_ROOT = WORKSPACE / "output"    # Reduced datasets, one folder per target.

SPEC = TargetSpec(
    name="slacs0008-0004",  # The SLACS parity anchor, same as start_here.py — the cache is shared.
    ra=2.012333,  # Right ascension in degrees.
    dec=-0.068944,  # Declination in degrees.
    proposal_ids=("10886",),  # The SLACS ACS program (Bolton et al. 2008).
)

"""
__The Instrument Adapter__

Everything instrument-specific in **PyAutoReduce** lives in one place: the instrument adapter.
The pipeline stages are generic; the adapter tells them what ACS/WFC is — its native pixel scale,
which calibrated product to download, which CRDS environment variable the drizzle needs, and the
default AstroDrizzle keywords.

The registry is public, so we can inspect the ACS/WFC adapter directly:
"""
adapter = instruments.get("acs_wfc")

print("--- the acs_wfc adapter ---")
print(f"MAST instrument name : {adapter.mast_instrument_name}")
print(f"native pixel scale   : {adapter.native_scale} arcsec/pix")
print(f"calibrated product   : _{adapter.calibrated_suffix.lower()}.fits")
print(f"saturation level     : {adapter.saturation_dn:.0f} e-")
print(f"default drizzle kwargs: {adapter.default_drizzle_kwargs}")
print(f"all registered instruments: {instruments.registered_keys()}")

"""
Note `calibrated_suffix="FLC"` — the adapter, not the user, decides which archive product a
reduction is built from. For ACS that is the CTE-corrected exposure, for reasons the next section
unpacks.

__MAST Query Anatomy__

The acquire stage's query is the standard `astroquery.mast` idiom
(https://astroquery.readthedocs.io/en/latest/mast/mast.html): `Observations.query_criteria` by
coordinates, observation collection, instrument and filter, then `get_product_list` +
`filter_products` + `download_products` for the calibrated exposures. **PyAutoReduce** wraps this
in the public helper `query_exposures`, which adds two pieces of query hygiene you would
otherwise learn the hard way.

**The product zoo.** For any HST pointing, MAST serves a family of products per exposure
(ACS Data Handbook, https://hst-docs.stsci.edu/acsdhb):

- `_raw` — the untouched detector readout.
- `_flt` — calibrated by `calacs` (bias, dark, flat, DQ arrays, ERR arrays) but **without** the
  pixel-based CTE correction.
- `_flc` — the same, **with** the CTE correction of Anderson & Bedin 2010 (PASP 122, 1035,
  https://ui.adsabs.harvard.edu/abs/2010PASP..122.1035A) applied. This is what the ACS adapter
  downloads: CTE trailing systematically dims faint arcs on low backgrounds, exactly what a lens
  reduction cannot afford. (One caveat the handbook is honest about: the pixel-based correction
  amplifies read noise in low-S/N pixels.)
- `_drz` / `_drc` — MAST's own drizzled combinations. **PyAutoReduce** never uses these: the
  drizzle geometry (scale, pixfrac, orientation) is precisely what a lensing reduction must
  control, so combination happens locally from the `_flc` frames.

**HAP exclusion.** A plain coordinate query also matches Hubble Advanced Products
(https://outerspace.stsci.edu/spaces/HAdP/pages/54558799/Improvements+in+HST+Astrometry) —
skycell mosaics and visit-level associations whose obs-ids start with `hst_`. Their member lists
re-reference the same exposures many times over (for this very target, 31 matched products dedupe
to 7 actual files), and the visit-level `hst_*_flc.fits` files are renamed *copies* of the member
exposures MAST already serves directly. Ingesting both drizzles every exposure twice — doubled
IVM weights then suppress the computed noise by sqrt(2), a real bug the pipeline's validation
caught. `query_exposures` therefore keeps only direct calibration-level observations and filters
the product lists the same way.

The query below is the exact one the pipeline runs internally (network required):
"""
observations = query_exposures(
    ra=SPEC.ra,
    dec=SPEC.dec,
    adapter=adapter,  # Supplies obs_collection + instrument_name for the MAST query.
    filter_name=SPEC.filter_name,  # F814W — the SLACS filter.
    radius="0.5 arcmin",  # The default search radius around the target.
    proposal_ids=SPEC.proposal_ids,  # Keeps the query to the SLACS program only.
)

print("\n--- MAST query (direct observations only, HAP excluded) ---")
for row in observations:
    print(f"  obs_id={row['obs_id']}  proposal={row['proposal_id']}")

"""
__CRDS Reference Files__

Reference files are acquisition too. AstroDrizzle's IVM weighting resolves ACS calibration files
(darks, flats, bad-pixel tables) through the `jref$` environment prefix, served by the
Calibration Reference Data System (CRDS, https://hst-crds.stsci.edu). The acquire stage runs the
CRDS best-references sync (`crds.bestrefs --sync-references=1 --update-bestrefs`) for the
downloaded exposures and exports the environment the drizzle needs.

Two behaviours worth knowing:

- **The sync runs on every reduction, even over a warm exposure cache** — CRDS revises reference
  files independently of the exposures, so a cached re-run re-checks. When nothing is stale this
  is cheap.

- **`sync_references=False` is the explicit offline opt-out**: valid only over a
  previously-synced cache (it raises a `RuntimeError` on a cold one, rather than producing a
  drizzle with missing reference files). Use it for repeated offline experimentation once one
  networked run has warmed the cache:

      offline_spec = dataclasses.replace(SPEC, sync_references=False)

Reference files are shared across targets and are the one cache component never evicted.

__The Exposure Cache__

Full-frame exposures are transient by design: download per target, reduce, package, evict.
SLACS-like targets are single pointings (~0.5 GB); a whole-sample run streams one target at a
time so peak disk usage never grows with sample size. The machinery is the public
`ExposureCache`:

- `cache_root/<target>/` holds the exposures; `cache_root/crds/` the shared references.
- `cache_root/cache_manifest.json` records what was downloaded from where, so eviction never
  costs reproducibility — a re-run re-fetches deterministically.
- `reduce_target(..., size_cap_bytes=...)` enforces a size cap by evicting oldest completed
  targets; `evict_when_done=True` drops a target's exposures as soon as its products are written.
- A warm manifest is **the** offline mechanism: `exposures_for()` short-circuits MAST entirely
  when the files are already on disk.

__Run The Pipeline__

Now run the reduction, so the remaining sections have real evidence to read. Everything below
this call is *reading* — the run itself is identical to `start_here.py`.
"""
print(
    "\n"
    "Running the full reduction now (acquire -> align -> drizzle -> noise -> psf -> package).\n"
    "Cold cache: ~10-30 minutes (downloads ~0.5 GB). Warm cache: a few minutes.\n"
)

record = reduce_target(SPEC, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)

out_dir = OUTPUT_ROOT / SPEC.name
work_dir = out_dir / "work"

print("Reduction complete.")

cache = ExposureCache(root=CACHE_ROOT)
manifest = cache.read_manifest()
entry = manifest["targets"][SPEC.name]
print("\n--- cache manifest entry ---")
print(f"files downloaded : {len(entry['files'])}")
print(f"source           : {entry['source']}")
print(f"downloaded at    : {entry['downloaded_at']}")
print(f"evicted          : {entry['evicted']}")

"""
__Footprint And Quality Filtering__

Between download and drizzle, two screens run — and both leave evidence in the `acquire` block:

- **Usability screen.** MAST serves failed exposures (EXPFLAG "EXCESSIVE DOWNTIME", zero
  exposure time) alongside the good ones. They carry no science content and are dropped before
  combination — this target's archive set includes exactly such a failed exposure, which an
  early version of the pipeline combined before the screen existed.

- **Detector-footprint filter.** Only exposures whose detector footprint actually covers the
  target (with a margin of the cutout half-extent plus 15") enter combination. Survey visits can
  span many pointings that never touch the target; combining them wastes memory and time.
"""
acq = record["acquire"]

print("\n--- acquire block ---")
print(f"exposures kept       : {acq['n_exposures']}")
print(f"skipped off-target   : {acq['n_skipped_off_target']}")
print(f"rejected as unusable : {acq['n_skipped_unusable']}  {acq['unusable_exposures']}")
print(f"references synced    : {acq['references_synced']}")
print(f"exposure files       : {acq['exposures']}")

"""
__WCS And Alignment__

Frame-to-frame registration is a prerequisite for both cosmic-ray rejection and PSF fidelity —
misaligned frames produce elongated stars and CR-flagged galaxy cores. The DrizzlePac Handbook's
alignment chapter (https://hst-docs.stsci.edu/drizzpac) covers the classical fix, TweakReg, which
catalog-matches sources across frames and updates each WCS to few-milliarcsecond registration.

**PyAutoReduce**'s default is to *not* run TweakReg — and to be honest about what is and is not
wired:

- **Default: trust the a-priori WCS.** MAST now attaches Gaia-tied WCS solutions to HST
  exposures (the HAP astrometry programme — roughly 70% of ACS/WFC3 frames align to Gaia at
  ~10 mas). Within a single visit, relative alignment is normally already adequate for
  drizzling. The `align` block records which WCS solution each exposure carries, so you can see
  what you trusted.

- **The design intent** is TweakReg as a *deviation trigger*: run it only when a
  cross-correlation diagnostic shows residual misregistration above ~0.1 pixel. **In the current
  release that trigger is not wired** — the pipeline always records `tweakreg_run: False`, and
  the `alignment_tolerance_pix` spec field is read by nothing. If your field shows elongated
  stars in the mosaic, that is the symptom to look for; the per-frame registration residuals in
  `individual.py` are the quantitative check.
"""
print("\n--- align block ---")
print(f"tweakreg_run : {record['align']['tweakreg_run']}")
for exposure, wcsname in record["align"]["wcs_solutions"].items():
    print(f"  {exposure}: {wcsname}")

"""
__The Drizzle Stage__

The combine stage hands AstroDrizzle
(https://drizzlepac.readthedocs.io; algorithm: Fruchter & Hook 2002, PASP 114, 144,
https://ui.adsabs.harvard.edu/abs/2002PASP..114..144F) a fully explicit keyword set, recorded
verbatim in the provenance. Let's walk the keywords that matter, in the order they act
(DrizzlePac Handbook, https://hst-docs.stsci.edu/drizzpac):

- `skymethod='globalmin+match'` — sky subtraction before combination (next section).
- `median=True`, `blot=True`, `driz_cr=True` — the cosmic-ray flagging flow (section after).
- `final_scale=0.05` — output pixel scale in arcsec. The SLACS convention; equal to the ACS/WFC
  native scale, so the scale ratio s = 1.
- `final_pixfrac=0.8` — each input pixel is shrunk to 80% of its size before being "dripped"
  onto the output grid. The drizzle resolution/noise trade-off dial; `dials.py` is the full
  study.
- `final_kernel='square'` — the drop shape. STScI's default; SLACS IX used `gaussian`, one of
  the literature disagreements `dials.py` covers.
- `final_wht_type='IVM'` — inverse-variance weight maps, which the noise stage requires.
- `final_units='cps'` — electrons per second, the **PyAutoLens** flux convention.
- `final_rot=0.0` — north-up output.
- `preserve=False, build=False, clean=True` — housekeeping: no intermediate-file archaeology,
  separate `_sci`/`_wht` outputs, scratch files removed.

One operational quirk explains a workspace rule: drizzlepac lowercases output filenames
internally, so the combine runs inside a scratch directory (`work/`) — which is why **every path
handed to PyAutoReduce must be absolute**.
"""
print("\n--- drizzle kwargs (verbatim from provenance) ---")
for key, value in record["drizzle"]["drizzle_kwargs"].items():
    print(f"  {key} = {value}")

"""
__Sky Subtraction__

`skymethod='globalmin+match'` computes one global sky minimum across the exposure set and then
*matches* frame-to-frame offsets, rather than measuring the sky independently per frame
(`localmin`). This matters for lens fields: AstroDrizzle's per-frame sky estimate can be biased
by a large galaxy filling a small field — for a deflector at the field centre, an independent
per-frame sky would subtract galaxy light as "sky". The matched method estimates one consistent
pedestal and preserves relative photometry across the set. The DrizzlePac Handbook documents the
four `skymethod` options.

The subtraction is *virtual*: AstroDrizzle records each frame's sky in the `MDRIZSKY` header
keyword and subtracts during the drizzle, leaving the `_flc` files untouched. `individual.py`
shows where that pedestal must be handled explicitly (per-frame products).

__Cosmic-Ray Flagging__

With no shutter closed between reads, every HST exposure is peppered with cosmic rays. The
default flow is AstroDrizzle's median-combine flagging (DrizzlePac Handbook): drizzle each
exposure separately onto the common grid, median-combine those single drizzles into a clean
reference, "blot" the median back to each frame's geometry, and flag pixels that deviate beyond
`driz_cr_snr` — writing DQ bit 4096. It needs >= 2 overlapping exposures; on a single exposure
the pipeline automatically switches to its single-exposure branch (no median, no flagging —
recorded as `single_exposure_branch` in the provenance).

The honest caveat: on steep gradients (deflector cores, PSF star cores) the blotted median reads
systematically low when sub-pixel dither shifts smear the peak, so `driz_cr` can flag genuine
core flux as cosmic rays — measured at ~37% deflector-core flux loss on SLACS-like data at
pipeline thresholds. The `cr_method="deepcr"` dial (Zhang & Bloom 2020, ApJ 889, 24,
https://ui.adsabs.harvard.edu/abs/2020ApJ...889...24Z — a CNN trained largely on exactly this
ACS/F814W regime) is the per-frame alternative; `dials.py` tells the full story, including why
the default has not (yet) been flipped.
"""
print("\n--- drizzle CR evidence ---")
print(f"cr_method               : {record['drizzle']['cr_method']}")
print(f"single_exposure_branch  : {record['drizzle']['single_exposure_branch']}")

"""
__Noise Construction By Hand__

The noise stage is the most lensing-specific step, and its two ingredients — the science mosaic
and the IVM weight map — persist in `work/` after the run. So rather than just describing it, we
rebuild the noise-map ourselves with the same public functions the pipeline uses, and check we
get the shipped product.

The recipe (Bayer et al., https://arxiv.org/abs/1803.05952, section 3.1):

    sigma_i = R * sqrt( max(N_i, 0) / t_exp  +  1 / W_i )

- `1/W_i`: the IVM weight map is an inverse *background* variance per STScI semantics
  (DrizzlePac Handbook section 3.4) — read noise, dark and sky in one term.
- `max(N_i, 0)/t_exp`: the source Poisson term from the cps image and total exposure time,
  floored at zero.
- `R`: the Casertano et al. 2000 (AJ 120, 2747) correlated-noise factor. Drizzling shares each
  input pixel among neighbouring output pixels, so adjacent output pixels carry correlated
  noise; the per-pixel RMS then underestimates the noise a chi^2 (which assumes independent
  pixels) actually experiences. R is the scalar correction, a function of pixfrac p and scale
  ratio s only.
"""
mosaic_stem = f"{SPEC.name}_{SPEC.filter_name}".lower()
sci_path = sorted(work_dir.glob(f"{mosaic_stem}*_sci.fits"))[0]
wht_path = sorted(work_dir.glob(f"{mosaic_stem}*_wht.fits"))[0]

sci = fits.getdata(sci_path).astype(float)
wht = fits.getdata(wht_path).astype(float)
header = fits.getheader(sci_path)
exptime = float(header.get("EXPTIME", header.get("TEXPTIME")))

R = casertano_r(
    pixfrac=SPEC.final_pixfrac,  # 0.8 — the drizzle drop size.
    scale_ratio=adapter.scale_ratio(SPEC.final_scale),  # 1.0 — output scale / native scale.
)

noise_by_hand = noise_map_from(
    sci=sci,  # The cps science mosaic.
    wht=wht,  # The IVM weight map.
    exptime=exptime,  # Total exposure time, for the Poisson term.
    correlated_noise_factor=R,  # The Casertano correction.
)

print("\n--- noise construction by hand ---")
print(f"mosaic            : {sci_path.name}")
print(f"total exposure    : {exptime:.0f} s")
print(f"R (p={SPEC.final_pixfrac}, s=1) : {R:.3f}")
print(f"pipeline recorded R           : {record['drizzle']['correlated_noise_factor']:.3f}")

"""
To compare against the shipped `noise_map.fits` we cut the same 281x281 stamp out of our
hand-built full-mosaic map — using the same strict `Cutout2D` the packaging stage uses (more on
strict mode below).
"""
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

cutout = Cutout2D(
    data=noise_by_hand,
    position=WCS(header).world_to_pixel_values(SPEC.ra, SPEC.dec),
    size=SPEC.cutout_shape,
    mode="strict",  # Raises if the stamp leaves the mosaic — never pads silently.
)

shipped_noise = fits.getdata(out_dir / "noise_map.fits").astype(float)

# Compare away from masked-by-noise pixels (1e8) and any isolated bad pixels our raw map
# carries as NaN before the packaging policy handles them.
comparable = (shipped_noise < 1.0e7) & np.isfinite(cutout.data)
ratio = np.median(cutout.data[comparable] / shipped_noise[comparable])
print(f"hand-built / shipped noise (median): {ratio:.4f}")

sky_rms = empirical_background_rms(sci)
print(f"empirical blank-sky RMS of the mosaic : {sky_rms:.5f} e-/s")
print(f"pipeline recorded                     : {record['noise']['empirical_background_rms']:.5f} e-/s")

"""
The ratio should be 1.0000 to numerical precision — the shipped map *is* this construction. The
empirical blank-sky RMS is the validation closure the pipeline records with every run: the
sigma-clipped RMS of the mosaic background, which the noise-map's background term must track.

__The PSF Stage__

After the noise, the pipeline builds the PSF: field stars are selected on the mosaic
(unsaturated, uncrowded, away from edges and the target itself), and an effective PSF is built
from them following Anderson & King 2000 (PASP 112, 1360) via `photutils.EPSFBuilder`. The
delivered kernel honours the drizzled-PSF invariant — it is measured on the same
kernel/pixfrac/scale/orientation as the science mosaic.
"""
print("\n--- psf block (summary) ---")
print(f"method           : {record['psf']['method']}")
print(f"n_stars_used     : {record['psf']['n_stars_used']}")
print(f"fwhm_pix         : {record['psf']['fwhm_pix']:.2f}")
print(f"star_source_pass : {record['psf']['star_source_pass']}")

"""
The PSF deserves — and has — its own script: `psf.py` in this folder covers the selection cuts,
the `psf_star_pass` dial (what `star_source_pass` records and why), the STARRED backend, the
tier-2 honesty story and the quality diagnostics. We defer to it here.

__Packaging And The Strict Cutout__

Packaging cuts the 281x281 stamp with astropy's `Cutout2D` in `mode="strict"` — if the requested
stamp is not fully inside the mosaic, the reduction *raises* rather than shipping a padded or
truncated dataset. Size the cutout to the coverage, never the other way round.

Two more guards run at packaging time, both leaving provenance:

- **The bad-pixel policy**: isolated non-finite noise pixels (fully-rejected or dead pixels —
  routine in deep resampled stacks) are masked-by-noise (noise = 1e8, data zeroed) and counted.
  The failure stays loud where it matters: a *structured* defect (contiguous bad pixels), more
  than 0.5% of the cutout bad, or any bad pixel within 1.5" of the target centre is an error,
  not a masking opportunity — the lens region must reduce cleanly.

- **Weight uniformity over the cutout**: the same RMS/median diagnostic, re-measured on the
  shipped stamp specifically.
"""
print("\n--- package block ---")
print(f"products      : {record['package']['products']}")
print(f"cutout shape  : {record['package']['cutout_shape']}")
print(f"pixel scale   : {record['package']['pixel_scale']}")
print(f"data units    : {record['package']['data_units']}")

print("\n--- bad pixel policy ---")
print(json.dumps(record["bad_pixel_policy"], indent=2))

wht_cut = Cutout2D(
    data=wht,
    position=WCS(header).world_to_pixel_values(SPEC.ra, SPEC.dec),
    size=SPEC.cutout_shape,
    mode="strict",
)
print("\n--- weight uniformity, demonstrated standalone ---")
print(f"full mosaic : {weight_uniformity(wht):.3f}")
print(f"cutout      : {weight_uniformity(wht_cut.data):.3f}  (limit {WEIGHT_UNIFORMITY_LIMIT})")
print(f"pipeline recorded (cutout): "
      f"{record['drizzle']['weight_uniformity_cutout']['wht_rms_over_median']:.3f}")

"""
Finally, `reduction.json` on disk is the returned record plus an envelope — `written_at` and the
software versions of every package in the chain — so a dataset is auditable years later even if
this workspace is long gone.
"""
reduction = json.loads((out_dir / "reduction.json").read_text())
print("\n--- reduction.json top-level blocks ---")
print(sorted(reduction.keys()))

"""
__Parity__

The pipeline's acceptance method is parity: register a new reduction onto a reference
sub-pixel-accurately and compare bright-pixel data and noise ratios. The public helper
`registered_ratios` implements it — the same statistics behind the SLACS validation numbers
(data ratio ~0.96 against the legacy SLACS dataset; noise ~30% above the legacy maps *by
design*, because the legacy maps do not carry the correlated-noise correction R and ours do).

Parity needs a reference. If you have run `dials.py` (which writes a second reduction of this
target at different drizzle dials), we compare against it; otherwise this section skips —
honestly, rather than inventing a reference.
"""
reference_dir = OUTPUT_ROOT / "dials" / "pixfrac_0p6" / SPEC.name

if reference_dir.exists():
    from autoreduce.validation import registered_ratios

    ref_data = fits.getdata(reference_dir / "data.fits").astype(float)
    ref_noise = fits.getdata(reference_dir / "noise_map.fits").astype(float)
    new_data = fits.getdata(out_dir / "data.fits").astype(float)
    new_noise = fits.getdata(out_dir / "noise_map.fits").astype(float)

    parity = registered_ratios(new_data, new_noise, ref_data, ref_noise)
    print("\n--- parity vs the dials.py pixfrac-0.6 reduction ---")
    print(json.dumps(parity, indent=2))
    print(
        "\nExpect data_ratio_median ~= 1 (same photons, different resampling) and the noise\n"
        "ratio tracking the ratio of the two Casertano factors."
    )
else:
    print(
        "\n[parity] no reference reduction found at "
        f"{reference_dir}\n"
        "[parity] run scripts/hst_acs/dials.py first to create one — skipping the parity leg."
    )

"""
__Wrap Up__

You have now seen every stage of the default ACS pipeline with its evidence: the archive anatomy
and query hygiene, CRDS reference syncing and the offline dial, the transient cache, the explicit
AstroDrizzle keyword set, sky matching, cosmic-ray flagging and its caveat, the noise-map rebuilt
by hand from the run's own mosaic, the PSF summary, and the strict packaging guards.

The following locations of the workspace are good places to checkout next:

- `scripts/hst_acs/dials.py`: the trade study for the dials this script kept at their defaults —
  pixfrac, kernel, CR method.
- `scripts/hst_acs/psf.py`: the full PSF story this script deferred.
- `scripts/hst_acs/individual.py`: per-exposure frame products — the un-drizzled alternative.
- `scripts/hst_acs/simulator.py`: end-to-end validation by injecting a synthetic arc into the
  real exposures.
- `scripts/guides/noise_maps.py`: the noise recipes across all instruments, and why chi^2 cares.
- `scripts/guides/target_spec.py`: every `TargetSpec` dial in one place.

__Env__ (Developer Only)

ENV: network
"""
