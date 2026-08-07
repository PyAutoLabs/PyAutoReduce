# PyAutoReduce Workspace

[PyAutoReduce](https://github.com/PyAutoLabs/PyAutoReduce) |
[PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens) |
[autolens_workspace](https://github.com/PyAutoLabs/autolens_workspace) |
[PyAutoGalaxy](https://github.com/PyAutoLabs/PyAutoGalaxy)

Welcome to the **PyAutoReduce** Workspace!

**PyAutoReduce** reduces raw archival telescope data into **modeling-ready datasets** for
[PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens) and
[PyAutoGalaxy](https://github.com/PyAutoLabs/PyAutoGalaxy). Given a target (a strong lens, a
galaxy), it downloads the archive exposures, reduces them with the instrument's standard pipeline
tooling, and emits the exact products the modeling stack loads via `al.Imaging.from_fits`:

| Product | Description |
|---------|-------------|
| `data.fits` | Science cutout at the modeling pixel scale |
| `noise_map.fits` | Per-pixel RMS noise, correlated-noise corrected |
| `psf.fits` / `psf_full.fits` | Reduction-consistent PSF estimate (compact + extended) |
| `reduction.json` | Full provenance: exposures, pipeline dials, diagnostics, software versions |

A reduction is *declared*, not scripted: you build a `TargetSpec`, call `reduce_target`, and the
pipeline is a pure function of that spec plus the archive. Every script in this workspace is a
narrated, runnable walk through one instrument's reduction.

## Instrument Coverage

Each instrument folder is anchored to a real target reduced end-to-end and validated against a
published literature reduction — the workspace's quality bar:

| Folder | Instrument | Validation anchor | Literature quality bar |
|--------|-----------|-------------------|------------------------|
| `scripts/hst_acs` | HST ACS/WFC | SLACS J0008-0004 (prop. 10886) | SLACS, Bolton et al. 2008 |
| `scripts/hst_wfc3_uvis` | HST WFC3/UVIS | SDSS J0252+0039 F390W | Bayer et al. (arXiv:1803.05952) |
| `scripts/hst_wfc3_ir` | HST WFC3/IR | J0252+0039 IR snapshot | WFC3 Data Handbook conventions |
| `scripts/jwst_nircam` | JWST NIRCam (SW + LW) | COSMOS-Web ring (prop. 1727) | Mercier et al. 2024 (arXiv:2309.15986) |
| `scripts/keck_nirc2` | Keck NIRC2 LGS-AO | B1938+666 | SHARP, Lagattuta et al. 2012; Chen et al. 2019 |
| `scripts/alma` | ALMA (visibilities) | G09v1.40 (2016.1.00282.S) | uv-plane practice, e.g. Hezaveh et al. 2016 |
| `scripts/surveys` | Legacy Surveys / SDSS / Pan-STARRS cutouts | SLACS J0008-0004 field | Colour context only — **not** modeling data |

## Getting Started

Install the core package, then the extras for the instrument(s) you reduce:

```bash
pip install autoreduce            # core (numpy, astropy, astroquery, photutils, PyYAML)
```

| Extra | Installs | Needed for |
|-------|----------|------------|
| `autoreduce[hst]` | drizzlepac, drizzle | HST ACS + WFC3 (the AstroDrizzle combine) |
| `autoreduce[keck]` | pykoa, drizzle | Keck NIRC2 (KOA acquisition + native combine) |
| `autoreduce[psf]` | psfr, stpsf | High-fidelity PSF back-ends |
| `autoreduce[starred]` | starred-astro | The STARRED super-sampled ePSF back-end (GPL + JAX) |
| `autoreduce[frames]` | deepCR | Per-frame cosmic-ray masking (`cr_method="deepcr"`, frame products) |

Two stacks are deliberately **not** pip extras and are installed separately:

- **JWST**: `pip install jwst==1.14.0` (the pipeline is pinned to this version) plus `crds`,
  with `CRDS_PATH` / `CRDS_SERVER_URL` set for lazy reference syncing.
- **ALMA**: modular CASA — `pip install casatools casatasks` — for the visibility branch.

**Be honest with yourself about what running these scripts involves**: every reduction downloads
real exposures from an archive (MAST, KOA, the ALMA archive) and runs the instrument's heavy
reduction stack. Scripts need network access, several GB of cache space, and minutes-to-hours of
runtime on the first run of a target. They are written to *read* as documentation even when you
cannot run them.

## New Users

New users should read `scripts/start_here.py`. It explains why reduction quality matters for lens
modeling, walks a complete HST/ACS reduction of a SLACS lens end-to-end, and routes you to the
right instrument folder for your data.

The `scripts/guides` folder then covers the pieces every instrument shares: the output contract
(`output_contract.py`), noise-map construction (`noise_maps.py`) and the `TargetSpec` declaration
(`target_spec.py`) — the last two run offline.

## Workspace Structure

The workspace includes the following main directories:

- `scripts`: **PyAutoReduce** examples written as Python scripts, one folder per instrument.
- `notebooks`: Jupyter notebook versions, generated from `scripts` at release time.
- `config`: Build/CI configuration only — **PyAutoReduce** itself needs no config files.
- `dataset`: Per-target `TargetSpec` YAML files you may add (no FITS is ever committed).
- `output`: Where reduced datasets land, one folder per target (not committed).
- `cache`: Downloaded exposures + CRDS references, re-used across runs (not committed).

The instrument packages include the following types of example:

- `start_here`: The default pipeline end-to-end on the instrument's validation anchor.
- `step_by_step`: Every reduction step in as much granularity as the public API allows, taught
  with the instrument handbook and the literature.
- `individual`: Per-exposure frame products — modeling native frames instead of a mosaic.
- `psf`: The PSF story for that instrument — star selection, tiers, back-ends, diagnostics.
- `simulator`: Synthetic-source injection into real frames, and flux-recovery closure tests.

The `README.md` files distributed throughout the workspace describe what is in each folder.

## Community & Support

Support for **PyAutoReduce** is available via the PyAuto Slack workspace, where the community
shares updates and helps troubleshoot problems. Slack is invitation-only: if you'd like to join,
please send an email requesting an invite.

For installation issues, bug reports, or feature requests, please raise an issue on the
[GitHub issues page](https://github.com/PyAutoLabs/PyAutoReduce/issues).

## Contribution

To make changes to the examples, edit the corresponding Python files (`.py`) in the `scripts`
folder — never the notebooks, which are generated from the scripts at release time. The marker
`# %%` alternates between code cells and markdown cells in the generated notebooks.

## The Reduction Domain Ladder: Mosaics, Frames, Visibilities and Cutouts

**PyAutoReduce** organises reductions into a ladder of four data domains. What changes as you
climb is not the instrument but the *form* the modeling-ready dataset takes:

- An **imaging mosaic** (`hst_acs`, `hst_wfc3_uvis`, `hst_wfc3_ir`, `jwst_nircam`, `keck_nirc2`)
  is the default rung: dithered exposures combined onto a single resampled grid, with a matching
  noise map and a PSF built through the identical resampling. Resampling correlates the noise —
  which is why the noise map carries the Casertano correlated-noise correction.

- **Per-exposure frames** (`frame_products=True`; the `individual.py` scripts) sidestep the
  mosaic: every calibrated exposure ships as its own native-pixel dataset with per-frame PSFs and
  a manifest. Modeling the frames jointly costs more bookkeeping but the noise in each frame is
  *uncorrelated* — no drizzle, no correction factor.

- **Visibilities** (`alma`) skip images entirely: the dataset is the calibrated interferometer
  visibilities themselves — `data.fits` / `uv_wavelengths.fits` / `noise_map.fits`, each an
  `(N_vis, 2)` array loaded via `al.Interferometer.from_fits` — because fitting in the uv-plane
  keeps the noise independent and the likelihood well-defined.

- **Survey cutouts** (`surveys`) are the context rung: pre-reduced coadd cutouts fetched from
  Legacy Surveys / SDSS / Pan-STARRS services for colour context around a target. They ship no
  PSF and are **not** modeling data.

## Build Configuration

The `config/build/` directory contains files used by the automated build and test system (CI,
smoke tests). These are not relevant to normal workspace usage — see `config/README.md`. Because
every script here needs network access and a heavy instrument stack, no script currently runs in
CI smoke tests (see `smoke_tests.txt` for the per-script reasons).
