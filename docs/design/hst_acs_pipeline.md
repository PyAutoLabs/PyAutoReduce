# HST/ACS reduction pipeline — design

Phase-1 design for reducing HST ACS/WFC imaging of strong lenses into
modeling-ready datasets. The governing principle, applied at every stage:
**run the instrument's default pipeline; deviate only where a lensing
calculation requires it, and document every deviation here.**

The quality bar is the SLACS reduction ([Bolton et al. 2008, SLACS
V](https://arxiv.org/abs/0805.1931)): ACS/WFC F814W exposures drizzled to a
0.05″/pix mosaic — the reductions behind the existing modeling datasets
(e.g. the 54-lens subhalo-scan sample). The weak-lensing-grade reference for
ACS reduction craft is the COSMOS pipeline ([Koekemoer et al.
2007](https://arxiv.org/abs/astro-ph/0703095)).

## Output contract

Per target, the pipeline emits the `al.Imaging.from_fits` product set:

| File | Shape (default) | Content |
|------|-----------------|---------|
| `data.fits` | 281×281 (≈14″ at 0.05″/pix) | drizzled science cutout, WCS intact |
| `noise_map.fits` | matches `data.fits` | per-pixel RMS (background + Poisson), correlated-noise corrected |
| `psf.fits` | 21×21, odd | compact PSF kernel for convolution |
| `psf_full.fits` | 61×61, odd | extended PSF (wings) for point-source work |
| `reduction.json` | — | provenance: program/visit/exposure IDs, filter, zero-point, exposure time, pixel scale, drizzle parameters, PSF method + diagnostics, software versions |

The existing SLACS modeling cutouts have **stripped FITS headers** — no WCS,
units or exposure metadata survive. `reduction.json` + intact cutout headers
restore that provenance permanently.

## Stage 0 — target specification

A reduction is declared, not scripted: a small per-target YAML (or a table row
for samples) giving target name, RA/Dec, instrument/filter, proposal ID(s) if
known, cutout size, and any per-target overrides. The pipeline is a pure
function of this spec plus the archive — re-running it reproduces the dataset
bit-for-bit (modulo upstream calibration reference-file updates, which
`reduction.json` records).

## Stage 1 — acquire (astroquery.mast)

**Default behaviour.** Query MAST by coordinates (`astroquery.mast.Observations`),
filter to the requested instrument/filter, and download the calibrated,
CTE-corrected exposures (`_flc.fits`) plus association tables (`_asn.fits`).
No re-calibration with `calacs` — MAST's on-the-fly reprocessing keeps
exposures current with the best reference files.

**Disk-space strategy** (a first-class requirement, not an optimization):

- **Per-target transient cache.** SLACS-like targets are single pointings —
  ~4 exposures ≈ 0.7 GB, needed only while the target reduces. The cache
  directory has a configurable size cap; exposures are evicted
  (oldest-completed-target first) once their products are written. A manifest
  (`cache_manifest.json`) records what was downloaded from where, so eviction
  never loses reproducibility — a re-run re-fetches deterministically.
- **Survey-mosaic targets.** For lenses inside large programs (COSMOS etc.),
  don't download tiles: use the MAST **HAPCut** cutout service
  (`astroquery.mast.Hapcut`) to fetch spatial cutouts of Hubble Advanced
  Products directly. This path skips stages 2–3 (the HAP mosaic is already
  drizzled) and enters at stage 4 with the HAP weight maps — a documented
  trade-off, since HAP drizzle parameters are not ours.
- Whole-sample runs stream: acquire → reduce → package → evict, one target at
  a time, so peak disk usage is one target's exposures regardless of sample
  size.

## Stage 2 — align

**Default behaviour.** Trust the a-priori WCS solutions MAST now attaches to
exposures (aligned to Gaia where possible). Within a single visit, relative
alignment is normally already adequate for drizzling.

**Deviation trigger, not default.** Run `tweakreg` relative alignment only
when the drizzle's cosmic-ray flagging or the final mosaic shows residual
misregistration (elongated stars, CR-flagged galaxy cores). The pipeline
computes an alignment diagnostic (cross-correlation of single-drizzled
exposures) and records it; sub-0.1-pixel residuals skip TweakReg. Absolute
astrometry only needs to be good enough for the WCS in the cutout header —
lens modeling works in relative arcseconds.

## Stage 3 — drizzle (drizzlepac.AstroDrizzle)

**Default behaviour.** AstroDrizzle with the STScI-recommended flow: sky
matching (`skymethod='globalmin+match'`), cosmic-ray rejection (`driz_cr`,
median-combine baseline), final drizzle of all exposures onto one grid.

**Lensing deviations (each justified):**

| Parameter | Value | Why we deviate |
|-----------|-------|----------------|
| `final_scale` | 0.05″/pix | SLACS convention; matches every existing modeling dataset and the source-plane resolution regime lens models are calibrated in |
| `final_rot` | 0 (north-up) | uniform orientation across samples simplifies masks, position angles, and cross-dataset comparison |
| `final_wht_type` | `IVM` | inverse-variance weights are what stage 4's noise map needs |
| `final_pixfrac` | start 0.8 | 4-exposure snapshots under-sample a 0.05″ grid with small pixfrac; exact value fixed by the SLACS parity study (below) |
| `final_kernel` | start `square`; evaluate `gaussian` | SLACS V used a Gaussian kernel; parity study decides whether matching it matters for the noise/PSF products |
| `final_units` | `cps` (e-/s) | counts/s + `EXPTIME` in provenance keeps the Poisson term computable while matching how the existing datasets are modeled — confirmed against SLACS parity |

Undrizzled artifacts (`_single_sci`, masks) stay in the transient cache; only
the mosaic + weight map proceed.

**Open question for the parity study:** whether the existing SLACS
`data.fits` are in e-/s or e-; the study measures this from the data/noise
relation rather than assuming.

## Stage 4 — noise map

The single most lensing-specific stage. Drizzle does not emit an RMS map;
the strong-lensing literature does ([Bayer et al.,
arXiv:1803.05952](https://arxiv.org/abs/1803.05952), §3.1, on exactly this
SLACS-style data):

```
sigma_i = sqrt( N_i / W_i + sigma_sky^2 )        # per pixel i
```

- **Background term** `sigma_sky = 1/sqrt(W_i)` from the IVM weight map —
  read noise, dark current and sky in one term, per STScI weight-map
  semantics.
- **Poisson term** `N_i / W_i` with `N_i` the source photo-electrons in pixel
  i (counts/s × effective exposure time from the weight/exposure maps).
  Negative-fluctuation pixels floor the Poisson term at zero rather than
  producing NaNs — and the pipeline **crashes loudly** on NaN/zero weights
  inside the cutout rather than silently patching them.
- **Correlated-noise correction.** Drizzling correlates neighbouring pixels;
  a naive per-pixel RMS underestimates the noise a lens-model chi^2 sees. Apply
  the DrizzlePac handbook / Casertano et al. (2000) scalar correction R
  (function of pixfrac p and scale ratio s) to the RMS map, and record R in
  `reduction.json`. Whether SLACS applied R is a parity-study question — if
  the legacy noise maps are uncorrected, we ship the *corrected* map and
  document the difference (deviate-when-lensing-requires cuts both ways).

**Validation.** Empirical noise in blank-sky patches of the mosaic must match
the noise-map prediction; the data/noise relation in bright unsaturated pixels
must show the Poisson slope.

## Stage 5 — PSF

Tiered strategy, informed by weak-lensing and AGN/quasar practice. The
non-negotiable invariant, whatever the tier: **the delivered PSF is the
drizzled PSF** — same kernel, pixfrac, scale and orientation as the science
mosaic. Never pair a native-frame PSF with a drizzled image.

- **Tier 1 — empirical ePSF from field stars** (default when the drizzled
  field contains enough usable stars): Anderson & King-style effective-PSF
  construction via `photutils.EPSFBuilder` on star cutouts from the final
  mosaic. Star selection: unsaturated, uncrowded, DQ-clean point sources;
  the count and fit residuals go into `reduction.json`.
- **Tier 2 — TinyTim + focus model** (fallback; SLACS elliptical snapshot
  fields are typically star-poor): model PSFs raytraced per exposure with
  TinyTim, focus ("breathing") estimated by matching whatever stars exist,
  per the COSMOS weak-lensing method ([Rhodes et al.
  2007](https://arxiv.org/abs/astro-ph/0701480) — focus recoverable to
  <1 μm rms from few stars). Per-exposure model PSFs are then **drizzled
  through the same AstroDrizzle footprint** as the science frames, which
  handles the drizzled-PSF invariant exactly. TinyTim's maintenance status is
  a risk; STScI's newer focus-diverse ePSF libraries for ACS are evaluated in
  the spike as a possible replacement.
- **Tier 3 — high-fidelity reconstruction back-ends (optional extra):**
  [STARRED](https://arxiv.org/abs/2402.08725) (wavelet-regularized
  two-channel PSF reconstruction; current state of the art in the TDCOSMO
  pipeline comparisons) and [PSFr](https://github.com/sibirrer/psfr)
  (Birrer's iterative reconstruction developed for lensed quasars). Not
  needed for SLACS-style galaxy-galaxy lenses; becomes the default tier for
  lensed quasars/AGN where the point source itself constrains the PSF.

Products: `psf.fits` 21×21 for fit convolution; `psf_full.fits` 61×61
capturing wings (diffraction spikes matter for quasar work later). Both odd,
centered, unit-normalized; construction method + diagnostics recorded.

## Stage 6 — package

Cut out `data.fits` and `noise_map.fits` around the target (default 281×281;
configurable), **preserving the cutout WCS and units in the headers** —
deviation from the legacy datasets' stripped headers, deliberately. Write the
PSFs and `reduction.json`. Optionally emit the auxiliary modeling-prep files
(`info.json` skeleton) but leave scientific annotations (positions, extra
galaxies) to the modeling workflow — reduction ends at the dataset.

## Validation — SLACS parity study

End-to-end on 2–3 SLACS lenses (e.g. `slacs0008-0004` plus one well-behaved
and one awkward system):

1. Reduce from MAST with this pipeline.
2. Compare against `/mnt/c/Users/Jammy/Science/subhalo/dataset/slacs/<lens>/`:
   data ratio map (photometric parity + units answer), noise ratio map
   (weight semantics + correlated-noise answer), PSF FWHM/ellipticity
   comparison.
3. Fit both reductions with the standard PyAutoLens SLaM-style model; the
   inferred lens parameters (Einstein radius, slope proxies) must agree
   within statistical errors. This is the acceptance test that matters —
   *the reduction is correct when the science is invariant under it.*
4. Discrepancies are documented in this file's parity appendix (to be added),
   never silently absorbed.

## Non-goals (phase 1)

WFC3, other filters, JWST, per-exposure `_flt` products, and Euclid (owned by
`euclid_strong_lens_modeling_pipeline`) — see `roadmap.md`. No GUI, no
database; per-target YAML + filesystem outputs only.
