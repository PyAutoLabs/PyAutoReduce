# Survey cutouts — ground-based colour context

Design for issue #50 (PyAutoMind prompt
`ground_based_instruments_optional_noise_psf.md`). Ground-based survey
imaging for **colour context** on lenses whose modeling data has no
optical counterpart (especially ALMA targets) — explicitly *not*
modeling inputs, which is why the RMS noise map and PSF are optional by
design.

## The domain decision

DES, SDSS, HSC and their peers deliver *pre-reduced* coadds through
public cutout services. Reducing raw DECam/SDSS frames would be a large
new pipeline for products the survey collaborations already made better;
so this is a third adapter domain (`cutout`) beside imaging and
visibility: **fetch + package, never reduce**. The branch mirrors the
visibility pattern — its own small pipeline behind the shared registry
and the `reduce_target` domain dispatch.

## Phase 1 services (endpoints verified live 2026-07-16, slacs0008 field)

| Adapter | Service | Bands | Scale | Noise | PSF |
|---|---|---|---|---|---|
| `legacy_surveys` | legacysurvey.org `fits-cutout` (DR10) | griz (default grz) | 0.262" | **yes** — `&invvar` appends an inverse-variance HDU to the same request | no |
| `sdss` | `astroquery.sdss` frames + WCS cutout | ugriz (default gri) | 0.396" | no (phase 1) | no |
| `panstarrs` | STScI `ps1filenames.py` + `fitscut.cgi` | grizy (default gri) | 0.25" | no (phase 1) | no |

**DES coverage note**: the DESI Legacy Imaging Surveys DR10 include the
DECam/DES footprint (and far more sky), so `legacy_surveys` *is* the DES
door — no separate DESDM service integration is needed.

Products per band: `<out>/<survey>/<band>/data.fits` (native scale,
survey WCS), `noise_map.fits` only where the service ships variance.
`reduction.json` carries a `products_optional` block stating what was
NOT produced and why — a survey cutout must never masquerade as a
modeling-ready `al.Imaging` dataset (no PSF, uncharacterised coadd
correlations).

## Assessed and deferred / recorded (the "what else is easy" ask)

- **HSC PDR** — cutout service is credential-gated (STARs account);
  deferred until a real need justifies the auth plumbing.
- **unWISE + GALEX** — served by the *same* Legacy viewer endpoint via
  the `layer=` parameter (`unwise-neo7`, `galex`); the cheapest
  multi-wavelength extension on the books — IR + UV colour for one
  small `layer` dial on the existing fetcher.
- **SDSS/PS1 variance** — SDSS frames carry gain/dark-variance/sky
  metadata and PS1 serves `.wt` stack files; both are recorded
  follow-ups if anyone actually needs ground-based noise maps.
- **PSF** — Legacy catalogs carry per-brick `psfsize_<band>`; a
  Gaussian-FWHM approximate kernel (clearly flagged approximate in
  provenance) is the recorded follow-up if colour context ever needs
  even a rough PSF.

## Boundaries

- The branch never imports the imaging stages; the only shared surfaces
  are the adapter registry, `TargetSpec` (`survey_bands` dial, reusing
  `cutout_shape` for size), and `reduction.json` provenance.
- Unit tests are numpy/astropy-only with fetchers monkeypatched; the
  real-network demonstration lives in `prototypes/survey_cutouts_spike.py`.
