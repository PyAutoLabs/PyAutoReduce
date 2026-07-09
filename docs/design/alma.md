# ALMA interferometer — the first visibility-domain reduction

Phase 5. The first instrument whose product is not an image: the output is
the `al.Interferometer.from_fits` triplet — `data.fits` (visibilities),
`uv_wavelengths.fits`, `noise_map.fits`, each shape `(Nvis, 2)` — plus
`reduction.json` provenance. The imaging stages (align, drizzle, sky, psf)
do not apply; the pipeline grows a **visibility branch**:

    acquire -> split -> extract -> assemble -> package

Reference practice is the working continuum-extraction recipe of an active
ALMA lens modeler (Aris; example project 2016.1.00282.S, field G09v1.40),
which this design generalises rather than re-derives. The one deliberate
extension over that recipe: the `WEIGHT` column is extracted too, because
`al.Interferometer` requires a per-visibility `noise_map` (the recipe's
outputs stop at visibilities + uv_wavelengths).

## Data model (how ALMA delivers)

- Data arrive as **measurement sets** (MS): a directory of tables, one MS
  per **execution block** (uid, e.g. `A002_Xb9b1b9_X3046`) — think of
  execution blocks as different exposures of the same scheduling block.
- Each MS carries several **spectral windows** (spw; typically 4). For
  continuum lens modelling every line-free spw is used; an spw containing
  an emission line is either dropped or extracted with more channels.
- **`width`** controls channel averaging at split time. The continuum
  default is to collapse the whole spw (`width` = the spw's channel count);
  for line work the observer chooses finer widths. Different spws may have
  different channel counts, so `width` is per-run, user-facing.
- Raw table shapes (after `casatools.table.getcol` + squeeze, single
  channel post-collapse): `DATA` `(n_pol=2, Nvis)` complex; `UVW`
  `(3, Nvis)` metres; `WEIGHT` `(2, Nvis)`; `SPECTRAL_WINDOW/CHAN_FREQ`
  scalar (collapsed) or `(n_chan,)`; `ANTENNA1`/`ANTENNA2`/`TIME`/
  `SCAN_NUMBER` `(Nvis,)`.

## Stage deltas vs the imaging design

| Stage | Delta |
|-------|-------|
| spec | `alma_uids` pins the execution blocks; `alma_field` names the science field inside the MS; `alma_spws` selects spectral windows; `alma_width` sets channel averaging (`0` = collapse each spw fully, the continuum default); `alma_ms_dir` points at already-calibrated MS when the archive step is done elsewhere. Imaging dials (cutout, drizzle, psf shapes) are ignored |
| acquire | **ALMA archive via `astroquery.alma`** (`acquire/alma.py`): query by member OUS / project code, download the product + raw tarballs into the exposure cache. **The canonical pipeline input is a calibrated MS directory** (`uid___<uid>.ms.split.cal` or equivalent), however obtained — see "Calibrated-MS acquisition" below. When `alma_ms_dir` is set, acquisition is a local-directory scan (Aris's workflow: calibrated MS delivered by an ARC) |
| split | **new stage** (`visibilities/split.py`), runs `casatasks.split` twice per Aris's recipe: (1) isolate the science field from each uid's MS, (2) per spw, average channels by `width` (`keepflags=False` so fully-flagged rows are dropped; `datacolumn="data"` — calibrated MS carry the calibrated data in `DATA`). Idempotent: an existing output MS is reused, matching the recipe's own re-run behaviour |
| extract | **new stage** (`visibilities/extract.py`): `casatools.table` reads of `DATA`, `UVW`, `WEIGHT`, `SPECTRAL_WINDOW/CHAN_FREQ`, `ANTENNA1`, `ANTENNA2`, `TIME`, `SCAN_NUMBER` per (uid, spw) — a direct port of the recipe's `getcol_wrapper` family, plus `WEIGHT`. casatools is imported inside functions (heavy-dep rule, as for drizzlepac/jwst) |
| assemble | **new stage** (`visibilities/assemble.py`), pure numpy: UVW metres -> wavelengths (`u * f / c` per channel frequency), polarization combine (below), noise from weights (below), then concatenation across uids × spws into the final `(Nvis, 2)` arrays |
| noise | σ per visibility per polarization = `1 / sqrt(WEIGHT)` (the MS weight convention: weight = 1/σ²; `split` re-scales weights through channel averaging). No Casertano factor — visibilities are uncorrelated samples, not resampled pixels |
| package | `package/interferometer.py` writes `data.fits`, `uv_wavelengths.fits`, `noise_map.fits` (each `(Nvis, 2)`, float64 — real/imag for data, u/v for wavelengths, σ_real/σ_imag for noise) + per-block diagnostic sidecars (`antennas_<uid>_spw_<spw>.fits`, `scans_…`, `times_…`, `frequencies_…` — the reference recipe's own exports) + `reduction.json`. Contract validated by loading with `al.Interferometer.from_fits` in the prototype — never imported by the library (boundary rule) |

## Polarization

`DATA` carries two parallel-hand correlations (XX, YY). Continuum lens
modelling fits Stokes I, so the assemble stage forms the weighted average

    I    = (w_xx·XX + w_yy·YY) / (w_xx + w_yy),      w = WEIGHT (= 1/σ²)
    σ_I  = 1 / sqrt(w_xx + w_yy)

per visibility (the same estimator CASA's own Stokes-I conversion uses).
The same σ_I applies to the real and imaginary parts — the MS weight is
per complex visibility. Rows where both polarizations carry zero/invalid
weight are dropped loudly (counted in provenance), never zero-filled.
Stacking both polarizations as independent visibilities was considered and
rejected: it doubles Nvis (NUFFT cost) for no information gain over the
weighted average.

## Calibrated-MS acquisition (the researched decision)

"Download the reduced data" (the modeler's modern workflow) resolves to
these archive paths, none of which is a plain anonymous file download of a
calibrated MS:

- **ARC on-demand services** — EU ARC "CalMS" service and EA/NA helpdesk
  requests deliver calibrated MS out-of-band; NRAO **SRDP** serves restored,
  pipeline-calibrated MS for Cycle 5+ data.
- **Local restore** — download the product + raw tarballs (this is what
  `astroquery.alma` automates) and run `scriptForPI.py` under the CASA
  version recorded in the QA2 README. Version pinning makes full automation
  of the restore a separate concern.

Design consequence: `acquire/alma.py` automates the archive **download**
(query by project code / member OUS, fetch tarballs into the cache,
checksums into provenance) and the pipeline consumes a **calibrated MS
directory** from any of the paths above (`alma_ms_dir` for delivered/
restored MS is the expected common case today). Automating the scriptForPI
restore inside the pipeline is an **open item** — it requires matching
monolithic CASA versions per cycle, exactly the constraint the modular
tooling avoids for extraction.

## Headless CASA (the second researched decision)

The recipe historically ran inside the monolithic `casa` shell (`tb` and
`split` as injected globals) — the modeler never got a plain-Python
invocation working. The modular pip packages solve this: **`casatools`**
provides the `table` tool and **`casatasks`** provides `split`, both
pip-installable (wheels through Python 3.13) and proven in headless
environments. Extraction of an already-calibrated MS has no CASA-version
coupling (that constraint binds only the scriptForPI restore), so any
recent modular CASA works. Both packages are heavy deps: imported inside
functions, never at module level, never in unit tests — the same rule as
drizzlepac / the jwst stack. Fallback if modular CASA is unavailable on a
host: `casa --nogui --agg -c <script>` against a generated script, not
implemented until needed.

## Validation anchor

Project **2016.1.00282.S**, field **G09v1.40** (uids `A002_Xb9b1b9_X3046`,
`A002_Xb99cbd_X2456`; spws 1, 2; width 240): `prototypes/alma_g09v140.py`
runs the visibility branch end-to-end on the calibrated MS and compares
visibilities and uv_wavelengths **numerically** against the modeler's own
exported files (he has offered them), then loads the packaged products with
`al.Interferometer.from_fits` and checks a dirty-image reconstruction shows
the source. The prototype accepts either an archive download or a local MS
directory.

## Open items

- scriptForPI restore automation (CASA-version pinning per cycle).
- Emission-line / cube extraction (per-channel Interferometer lists) — the
  modelling side shipped separately (the `alma-datacube` task); the
  reduction side reuses `split` with finer `width` and per-channel assemble,
  deferred until a line-modelling dataset needs it.
- `SIGMA` column cross-check: for calibrated data `WEIGHT` is authoritative;
  a σ-vs-weight consistency diagnostic could be added to extraction.
- Time/baseline averaging beyond channel collapse (further Nvis reduction
  for very large configurations) — a `casatasks.split`/`mstransform` dial,
  not needed for the anchor dataset.
