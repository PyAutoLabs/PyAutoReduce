# PyAutoReduce

Data reduction of Hubble Space Telescope (and, in future, JWST and other
instrument) imaging into **modeling-ready datasets** for
[PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens) and
[PyAutoGalaxy](https://github.com/PyAutoLabs/PyAutoGalaxy).

Given a strong lens (or galaxy) target, PyAutoReduce downloads the archival
exposures, reduces them with the instrument's standard pipeline tooling, and
emits the exact products the modeling stack loads via `al.Imaging.from_fits`:

| Product | Description |
|---------|-------------|
| `data.fits` | Science cutout, drizzled to the modeling pixel scale |
| `noise_map.fits` | Per-pixel RMS: drizzle weight-map background term + Poisson source term, correlated-noise corrected |
| `psf.fits` / `psf_full.fits` | Drizzle-consistent PSF estimate (compact + extended) |
| `reduction.json` | Full provenance: program IDs, exposures, zero-point, exposure time, pixel scale, pipeline versions |

## Design principles

- **Default pipelines first.** Each stage uses the instrument's standard
  tooling (`astroquery.mast`, `drizzlepac`, `photutils`) with its recommended
  settings; deviations exist only where lens modeling requires them, and each
  one is documented in the design docs.
- **Disk-frugal.** Full-frame exposures are transient: download per target,
  reduce, package the cutouts, evict. Survey-mosaic targets use MAST cutout
  services instead of tile downloads.
- **Standalone.** PyAutoReduce emits the PyAutoLens/PyAutoGalaxy input format
  but does not import them; it sits directly on the astropy ecosystem.

## Status

**Design phase.** The HST/ACS pipeline design lives in
[`docs/design/hst_acs_pipeline.md`](docs/design/hst_acs_pipeline.md); the
longer-term roadmap (WFC3, JWST, per-exposure frame products) in
[`docs/design/roadmap.md`](docs/design/roadmap.md).

## Installation

```bash
pip install autoreduce            # core (outputs + packaging only)
pip install "autoreduce[hst]"     # + the STScI HST reduction stack (drizzlepac)
pip install "autoreduce[psf]"     # + high-fidelity PSF reconstruction back-ends
```
