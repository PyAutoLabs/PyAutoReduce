"""
The stage orchestrator: TargetSpec in, modeling-ready dataset out.

    acquire -> align -> drizzle -> noise -> psf -> package

Each stage contributes to the provenance record; `reduction.json` is written
alongside the data products. Heavy dependencies (astroquery, drizzlepac,
photutils) are imported inside stages so the package imports without them.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import instruments
from .acquire import cache as cache_mod
from .acquire import crds as crds_mod
from .acquire import mast as mast_mod
from .align import diagnostics as align_mod
from .drizzle import combine as combine_mod
from .noise import rms as rms_mod
from .package import cutout as cutout_mod
from .package import provenance as provenance_mod
from .psf import epsf as epsf_mod
from .psf import stars as stars_mod
from .target import TargetSpec


def reduce_target(
    spec: TargetSpec,
    cache_root: Path,
    output_root: Path,
    size_cap_bytes: Optional[int] = None,
    evict_when_done: bool = False,
) -> Dict:
    """Run the full pipeline for one target; returns the provenance record."""
    adapter = instruments.get(spec.instrument)
    cache = cache_mod.ExposureCache(Path(cache_root), size_cap_bytes=size_cap_bytes)
    out_dir = Path(output_root) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_dir.mkdir(exist_ok=True)

    record: Dict = {"target": spec.as_dict(), "instrument": adapter.key}

    # -- acquire ------------------------------------------------------------
    crds_mod.configure_environment(cache.references_dir, adapter)
    exposures = cache.exposures_for(spec.name)
    downloaded = False
    if not exposures:
        observations = mast_mod.query_exposures(
            spec.ra,
            spec.dec,
            adapter,
            spec.filter_name,
            proposal_ids=spec.proposal_ids,
        )
        exposures = mast_mod.download_exposures(
            observations, adapter, cache.target_dir(spec.name)
        )
        cache.record_download(
            spec.name, [str(p) for p in exposures], source="mast"
        )
        downloaded = True
    # Fully-cached re-runs with references already synced stay offline.
    refs_synced = False
    if downloaded or not crds_mod.references_present(cache.references_dir, adapter):
        crds_mod.sync_best_references(exposures)
        refs_synced = True
    record["acquire"] = {
        "n_exposures": len(exposures),
        "exposures": [Path(p).name for p in exposures],
        "downloaded": downloaded,
        "references_synced": refs_synced,
    }

    # -- align ----------------------------------------------------------------
    record["align"] = {
        "wcs_solutions": align_mod.wcs_solution_names(exposures),
        "tweakreg_run": False,  # a-priori WCS accepted by default (stage 2)
    }

    # -- drizzle ---------------------------------------------------------------
    sci_path, wht_path, drizzle_prov = combine_mod.combine(
        exposures, spec, adapter, work_dir
    )
    record["drizzle"] = drizzle_prov

    from astropy.io import fits

    with fits.open(sci_path) as hdul:
        sci = hdul[0].data.astype(float)
        header = hdul[0].header.copy()
    wht = fits.getdata(wht_path)
    exptime = header.get("EXPTIME", header.get("TEXPTIME"))
    if exptime is None or exptime <= 0:
        raise ValueError(f"mosaic header carries no positive EXPTIME: {exptime}")

    # -- noise -----------------------------------------------------------------
    noise = rms_mod.noise_map_from(
        sci,
        wht,
        exptime=float(exptime),
        correlated_noise_factor=drizzle_prov["correlated_noise_factor"],
    )
    record["noise"] = {
        "recipe": "R * sqrt(max(sci,0)/exptime + 1/wht)",
        "correlated_noise_factor": drizzle_prov["correlated_noise_factor"],
        "exptime": float(exptime),
        "empirical_background_rms": rms_mod.empirical_background_rms(
            sci[np.isfinite(noise)]
        ),
    }

    # -- psf -------------------------------------------------------------------
    from astropy.wcs import WCS

    target_xy = WCS(header).world_to_pixel_values(spec.ra, spec.dec)
    selection = stars_mod.StarSelection()
    stars = stars_mod.find_stars(
        sci,
        selection,
        target_xy=(float(target_xy[0]), float(target_xy[1])),
        peak_max=selection.saturation_fraction * adapter.saturation_dn / float(exptime),
    )
    psf, psf_full, psf_diag = epsf_mod.build_epsf(
        sci, stars, spec.psf_shape, spec.psf_full_shape
    )
    record["psf"] = psf_diag

    # -- package ----------------------------------------------------------------
    data_cut = cutout_mod.cutout_to_fits(
        sci, header, spec.ra, spec.dec, spec.cutout_shape, out_dir / "data.fits"
    )
    noise_cut = cutout_mod.cutout_to_fits(
        noise, header, spec.ra, spec.dec, spec.cutout_shape, out_dir / "noise_map.fits"
    )
    rms_mod.assert_finite_within(noise_cut, f"{spec.name} cutout")

    fits.PrimaryHDU(psf.astype(np.float32)).writeto(
        out_dir / "psf.fits", overwrite=True
    )
    fits.PrimaryHDU(psf_full.astype(np.float32)).writeto(
        out_dir / "psf_full.fits", overwrite=True
    )
    record["package"] = {
        "products": ["data.fits", "noise_map.fits", "psf.fits", "psf_full.fits"],
        "cutout_shape": list(spec.cutout_shape),
        "pixel_scale": spec.final_scale,
        "data_units": drizzle_prov["drizzle_kwargs"]["final_units"],
    }

    provenance_mod.write_reduction_json(out_dir, record)

    # -- evict --------------------------------------------------------------------
    cache.mark_completed(spec.name)
    if evict_when_done:
        cache.evict(spec.name)
    cache.enforce_cap()

    return record
