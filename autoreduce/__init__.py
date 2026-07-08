"""
autoreduce — data reduction of HST (and future JWST/other) imaging into
modeling-ready datasets for PyAutoLens and PyAutoGalaxy.

The pipeline turns archival exposures into the per-lens products the modeling
stack loads via ``al.Imaging.from_fits``: ``data.fits``, ``noise_map.fits``,
``psf.fits`` and ``psf_full.fits``, plus a ``reduction.json`` provenance record.

Status: design phase — see ``docs/design/hst_acs_pipeline.md`` for the HST/ACS
pipeline design and ``docs/design/roadmap.md`` for what comes after.
"""

from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("autoreduce")
except PackageNotFoundError:
    __version__ = "0.0.dev0"
