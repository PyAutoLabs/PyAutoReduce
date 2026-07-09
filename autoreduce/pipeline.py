"""
The stage orchestrator: TargetSpec in, modeling-ready dataset out.

    acquire -> align -> drizzle -> noise -> psf -> package

Each stage contributes to the provenance record; `reduction.json` is written
alongside the data products. Heavy dependencies (astroquery, drizzlepac,
photutils) are imported inside stages so the package imports without them.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import instruments
from .acquire import cache as cache_mod
from .acquire import crds as crds_mod
from .acquire import footprint as footprint_mod
from .acquire import mast as mast_mod
from .align import diagnostics as align_mod
from .drizzle import combine as combine_mod
from .drizzle.diagnostics import check_weight_uniformity
from .instruments import InstrumentAdapter
from .noise import rms as rms_mod
from .package import cutout as cutout_mod
from .package import provenance as provenance_mod
from .psf import epsf as epsf_mod
from .psf import stars as stars_mod
from .target import TargetSpec


@dataclass
class _StageContext:
    """Everything the stages share; `record` is the growing provenance."""

    spec: TargetSpec
    adapter: InstrumentAdapter
    cache: cache_mod.ExposureCache
    out_dir: Path
    work_dir: Path
    record: Dict = field(default_factory=dict)
    exposures: List[Path] = field(default_factory=list)


def _acquire(ctx: _StageContext) -> None:
    """Download (or reuse) exposures, sync references, footprint-filter."""
    spec, adapter, cache = ctx.spec, ctx.adapter, ctx.cache
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
    # The jwst pipeline syncs its own references lazily through CRDS_PATH,
    # so explicit bestrefs is an HST-observatory step.
    refs_synced = False
    if adapter.observatory == "hst" and (
        downloaded
        or not crds_mod.references_present(cache.references_dir, adapter)
    ):
        crds_mod.sync_best_references(exposures)
        refs_synced = True
    # Detector-footprint filter: only exposures covering the target enter
    # combination — survey visits span many detectors that never touch it,
    # and combining them wastes memory (image3 OOM) and time.
    cutout_extent = 0.5 * max(spec.cutout_shape) * spec.final_scale
    exposures, skipped = footprint_mod.filter_to_target(
        exposures, spec.ra, spec.dec, margin_arcsec=cutout_extent + 15.0
    )
    ctx.exposures = exposures
    ctx.record["acquire"] = {
        "n_exposures": len(exposures),
        "exposures": [Path(p).name for p in exposures],
        "n_skipped_off_target": len(skipped),
        "downloaded": downloaded,
        "references_synced": refs_synced,
    }


def _align(ctx: _StageContext) -> None:
    ctx.record["align"] = {
        "wcs_solutions": align_mod.wcs_solution_names(ctx.exposures),
        "tweakreg_run": False,  # a-priori WCS accepted by default (stage 2)
    }


def _combine(ctx: _StageContext):
    """Run the backend combine; load the mosaic; return (sci, header, wht, exptime)."""
    from astropy.io import fits

    sci_path, wht_path, drizzle_prov = combine_mod.combine(
        ctx.exposures, ctx.spec, ctx.adapter, ctx.work_dir
    )
    ctx.record["drizzle"] = drizzle_prov

    with fits.open(sci_path) as hdul:
        sci = hdul[0].data.astype(float)
        header = hdul[0].header.copy()
    wht = fits.getdata(wht_path)
    exptime = header.get("EXPTIME", header.get("TEXPTIME"))
    # Only the HST noise construction divides by exposure time; the JWST path
    # reads propagated ERR and needs no exptime — record it if present, but
    # never hard-fail a reduction that doesn't use it.
    if ctx.adapter.combine_backend != "jwst_image3" and (
        exptime is None or exptime <= 0
    ):
        raise ValueError(f"mosaic header carries no positive EXPTIME: {exptime}")
    return sci, header, wht, float(exptime) if exptime else 0.0


def _noise(ctx: _StageContext, sci, wht, exptime: float) -> np.ndarray:
    drizzle_prov = ctx.record["drizzle"]
    if ctx.adapter.combine_backend == "jwst_image3":
        from astropy.io import fits

        from .noise import jwst_rms as jwst_rms_mod

        err = fits.getdata(drizzle_prov["err_path"]).astype(float)
        noise, consistency = jwst_rms_mod.noise_map_from_error(
            err,
            sci,
            correlated_noise_factor=drizzle_prov["correlated_noise_factor"],
        )
        ctx.record["noise"] = {
            "recipe": "R * ERR (propagated by calwebb_image3 resample)",
            "correlated_noise_factor": drizzle_prov["correlated_noise_factor"],
            "exptime": float(exptime),
            **consistency,
        }
    else:
        noise = rms_mod.noise_map_from(
            sci,
            wht,
            exptime=float(exptime),
            correlated_noise_factor=drizzle_prov["correlated_noise_factor"],
        )
        ctx.record["noise"] = {
            "recipe": "R * sqrt(max(sci,0)/exptime + 1/wht)",
            "correlated_noise_factor": drizzle_prov["correlated_noise_factor"],
            "exptime": float(exptime),
            "empirical_background_rms": rms_mod.empirical_background_rms(
                sci[np.isfinite(noise)]
            ),
        }
    return noise


def _psf(ctx: _StageContext, sci, header):
    from astropy.io import fits
    from astropy.wcs import WCS

    spec, adapter = ctx.spec, ctx.adapter
    target_xy = WCS(header).world_to_pixel_values(spec.ra, spec.dec)
    selection = stars_mod.StarSelection()
    if adapter.observatory == "hst":
        # Saturation is per exposure, not per stack: a star saturates when its
        # rate fills the well within one exposure, so the cps cap divides the
        # full well by the longest single-exposure time — never the mosaic
        # total.
        max_single_exptime = max(
            float(fits.getheader(p).get("EXPTIME", 0.0)) for p in ctx.exposures
        )
        if max_single_exptime <= 0.0:
            raise ValueError("no exposure carries a positive EXPTIME header")
        peak_max = (
            selection.saturation_fraction
            * adapter.saturation_dn
            / max_single_exptime
        )
    else:
        # JWST mosaics are in surface-brightness units (MJy/sr) where a
        # full-well cut is meaningless; saturated cores arrive as NaN/DQ-blank
        # from the level-2 pipeline, so no peak cut is applied. Refinement
        # (unit-converted cap) tracked in docs/design/jwst.md open items.
        peak_max = None
    stars = stars_mod.find_stars(
        sci,
        selection,
        target_xy=(float(target_xy[0]), float(target_xy[1])),
        peak_max=peak_max,
    )
    psf, psf_full, psf_diag = epsf_mod.build_epsf(
        sci, stars, spec.psf_shape, spec.psf_full_shape
    )
    ctx.record["psf"] = psf_diag
    return psf, psf_full


def _package(ctx: _StageContext, sci, header, wht, noise, psf, psf_full) -> None:
    from astropy.io import fits

    spec, out_dir = ctx.spec, ctx.out_dir
    data_cut, data_header, center_xy = cutout_mod.make_cutout(
        sci, header, spec.ra, spec.dec, spec.cutout_shape
    )
    noise_cut, noise_header, _ = cutout_mod.make_cutout(
        noise, header, spec.ra, spec.dec, spec.cutout_shape
    )
    # Isolated dead/rejected pixels are masked-by-noise (recorded); the loud
    # failure remains for excessive masking or bad pixels near the lens.
    data_cut, noise_cut, mask_diag = rms_mod.mask_isolated_bad_pixels(
        data_cut,
        noise_cut,
        center_xy=center_xy,
        pixel_scale=spec.final_scale,
        region_name=f"{spec.name} cutout",
    )
    rms_mod.assert_finite_within(noise_cut, f"{spec.name} cutout")
    cutout_mod.write_fits(data_cut, data_header, out_dir / "data.fits")
    cutout_mod.write_fits(noise_cut, noise_header, out_dir / "noise_map.fits")
    ctx.record["bad_pixel_policy"] = mask_diag

    # The mosaic-wide WHT uniformity mixes coverage tiers across the full
    # union footprint; the science verdict belongs to the cutout region.
    wht_cut, _, _ = cutout_mod.make_cutout(
        wht, header, spec.ra, spec.dec, spec.cutout_shape
    )
    ctx.record["drizzle"]["weight_uniformity_cutout"] = check_weight_uniformity(
        wht_cut
    )

    fits.PrimaryHDU(psf.astype(np.float32)).writeto(
        out_dir / "psf.fits", overwrite=True
    )
    fits.PrimaryHDU(psf_full.astype(np.float32)).writeto(
        out_dir / "psf_full.fits", overwrite=True
    )
    ctx.record["package"] = {
        "products": ["data.fits", "noise_map.fits", "psf.fits", "psf_full.fits"],
        "cutout_shape": list(spec.cutout_shape),
        "pixel_scale": spec.final_scale,
        # Backend-agnostic: both backends stamp BUNIT on the mosaic header.
        "data_units": str(header.get("BUNIT", "unknown")),
    }


def _evict(ctx: _StageContext, evict_when_done: bool) -> None:
    ctx.cache.mark_completed(ctx.spec.name)
    if evict_when_done:
        ctx.cache.evict(ctx.spec.name)
    ctx.cache.enforce_cap()


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

    ctx = _StageContext(
        spec=spec,
        adapter=adapter,
        cache=cache,
        out_dir=out_dir,
        work_dir=work_dir,
        record={"target": spec.as_dict(), "instrument": adapter.key},
    )

    _acquire(ctx)
    _align(ctx)
    sci, header, wht, exptime = _combine(ctx)
    noise = _noise(ctx, sci, wht, exptime)
    psf, psf_full = _psf(ctx, sci, header)
    _package(ctx, sci, header, wht, noise, psf, psf_full)
    provenance_mod.write_reduction_json(out_dir, ctx.record)
    _evict(ctx, evict_when_done)

    return ctx.record
