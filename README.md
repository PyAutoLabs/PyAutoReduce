# PyAutoReduce: Telescope Data for Galaxy and Lens Modeling

**PyAutoReduce** prepares archival telescope observations for modeling with **PyAutoLens** and **PyAutoGalaxy**. Define your target and reduction settings, then produce the data, noise estimates and point spread functions needed for your analysis.

## Getting Started

Install the core package:

```bash
pip install pyautoreduce
```

For Hubble imaging reductions, install the HST tools:

```bash
pip install "pyautoreduce[hst]"
```

- **[Start with an example](https://github.com/PyAutoLabs/autoreduce_workspace/blob/main/scripts/start_here.py)** — follow an HST imaging reduction from archival exposures to a dataset ready for modeling.
- **[Find your instrument](https://github.com/PyAutoLabs/autoreduce_workspace#find-your-examples)** — explore the workspace examples for HST, JWST, Keck and ALMA.
- **[Understand the settings](https://github.com/PyAutoLabs/autoreduce_workspace/blob/main/scripts/guides/target_spec.py)** — learn how to describe your target and configure a reduction.

Each instrument needs its own reduction software. Running the examples also requires network access and space for downloaded exposures; you can read them without running them.

## What You Get

For imaging reductions, the main outputs are:

| File | What it contains |
|---|---|
| `data.fits` | The reduced science image |
| `noise_map.fits` | An estimate of the uncertainty in each pixel |
| `psf.fits` / `psf_full.fits` | Point spread functions describing the image’s blurring |
| `reduction.json` | Processing settings, diagnostics and provenance |

The [output guide](https://github.com/PyAutoLabs/autoreduce_workspace/blob/main/scripts/guides/output_contract.py) explains these files and the different outputs for individual exposures and interferometer data.

PyAutoReduce runs independently of PyAutoLens and PyAutoGalaxy; its outputs are prepared for use with those libraries.

## Explore Further

- **[Noise maps](https://github.com/PyAutoLabs/autoreduce_workspace/blob/main/scripts/guides/noise_maps.py)** — understand how reduction affects your uncertainty estimates.
- **[Reduction design](https://github.com/PyAutoLabs/PyAutoReduce/tree/main/docs/design)** — explore the methods and choices behind each instrument’s processing.

## Community and Contributing

Ask questions, suggest improvements and share your work in [PyAutoLabs Discussions](https://github.com/orgs/PyAutoLabs/discussions), or chat with us on [Slack](https://join.slack.com/t/pyautolens/shared_invite/zt-2cufp4eyf-fXfgMxRGuvg~bMrI3uOAxg).

## License

See the [license](https://github.com/PyAutoLabs/PyAutoReduce/blob/main/LICENSE).
