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
from .acquire import quality as quality_mod
from .align import diagnostics as align_mod
from .drizzle import combine as combine_mod
from .drizzle.diagnostics import check_weight_uniformity
from .instruments import InstrumentAdapter
from .noise import rms as rms_mod
from .package import cutout as cutout_mod
from .package import frames as frames_mod
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
    # Ground-based extras (KOA path): calibration/PSF-star frame paths and
    # the prepared PSF-star products the psf stage consumes.
    ground: Dict = field(default_factory=dict)


def _acquire_koa(ctx: _StageContext) -> None:
    """
    KOA acquisition (keck_ao.md stage 1): raw science frames, the night's
    calibrations, PSF-star frames, and the epoch-matched distortion solution.
    No footprint filter — NIRC2 observations are pointed and raw-header WCS
    is approximate; the frame set is pinned by ids/program instead.
    """
    from .acquire import koa as koa_mod

    spec, adapter, cache = ctx.spec, ctx.adapter, ctx.cache
    target_dir = cache.target_dir(spec.name)
    downloaded = []

    # Each component self-heals independently: a run interrupted after the
    # science download resumes by fetching only what is missing.
    exposures = cache.exposures_for(spec.name)
    if not exposures:
        science_table = koa_mod.query_science_frames(
            spec.ra,
            spec.dec,
            adapter,
            spec.filter_name,
            ctx.work_dir,
            proposal_ids=spec.proposal_ids,
            koa_ids=spec.koa_science_ids,
        )
        exposures = koa_mod.download_frames(science_table, target_dir, "science")
        cache.record_download(
            spec.name, [str(p) for p in exposures], source="koa"
        )
        downloaded.append("science")
    facts = koa_mod.frame_facts_from_headers(exposures)

    cal_paths = sorted((target_dir / "cals").rglob("*.fits*"))
    if not cal_paths:
        calib_table = koa_mod.query_night_calibrations(
            dates=sorted({f["date_obs"] for f in facts}),
            setups=sorted({(f["itime"], f["coadds"]) for f in facts}),
            adapter=adapter,
            filter_name=spec.filter_name,
            work_dir=ctx.work_dir,
        )
        cal_paths = koa_mod.download_frames(calib_table, target_dir, "cals")
        downloaded.append("cals")

    # The cached psf/ directory is consulted only when the *current* spec
    # names PSF-star frames — a spec that dropped its ids must not inherit
    # another run's stars (provenance would no longer match the products).
    psf_paths = []
    if spec.koa_psf_star_ids:
        psf_paths = sorted((target_dir / "psf").rglob("*.fits*"))
        if not psf_paths:
            star_table = koa_mod.query_science_frames(
                spec.ra,
                spec.dec,
                adapter,
                spec.filter_name,
                ctx.work_dir,
                koa_ids=spec.koa_psf_star_ids,
            )
            psf_paths = koa_mod.download_frames(star_table, target_dir, "psf")
            downloaded.append("psf")

    # Pointing coherence (the KOA analogue of the MAST footprint filter):
    # header pointings must cluster — an unpinned cone query can pull in
    # frames pointed at a nearby star or neighbouring target (the SHARP
    # PSF-star pointings sit only ~20" from the lens).
    _assert_pointing_coherence(exposures, spec)

    star_facts = (
        koa_mod.frame_facts_from_headers(psf_paths) if psf_paths else []
    )
    all_mjds = [f["mjd"] for f in facts] + [f["mjd"] for f in star_facts]
    epochs = {
        koa_mod.distortion_solution_for_mjd(m) for m in all_mjds
    }
    if len(epochs) != 1:
        raise ValueError(
            f"frame set spans the 2015-04-13 distortion-epoch boundary "
            f"({sorted(epochs)}); reduce each epoch as its own target spec"
        )
    distortion_prov = koa_mod.sync_distortion_solution(
        cache.references_dir, adapter, mjd=facts[0]["mjd"]
    )
    # Local absolute paths are working state (the prepared-frame headers
    # need them), never provenance — reduction.json must stay host-portable.
    ctx.ground["distortion"] = distortion_prov
    portable_prov = {
        k: v for k, v in distortion_prov.items() if k != "distortion_paths"
    }
    ctx.exposures = [f["path"] for f in facts]
    ctx.ground["cal_paths"] = cal_paths
    ctx.ground["psf_raw_paths"] = psf_paths
    ctx.record["acquire"] = {
        "n_exposures": len(ctx.exposures),
        "exposures": [Path(p).name for p in ctx.exposures],
        "n_calibration_frames": len(cal_paths),
        "n_psf_star_frames": len(psf_paths),
        "downloaded": downloaded,
        **portable_prov,
    }


def _assert_pointing_coherence(
    exposures, spec: TargetSpec, max_scatter_arcsec: float = 10.0
) -> None:
    """
    All science frames must point at the same field: loud on outliers from
    the median pointing beyond the dither budget (`max_scatter_arcsec` plus
    the cutout extent). Raw-header pointing is arcsecond-grade, which is
    exactly good enough to catch a different-pointing contaminant.
    """
    from astropy.io import fits

    ras, decs, names = [], [], []
    for path in exposures:
        header = fits.getheader(path)
        ras.append(float(header["RA"]))
        decs.append(float(header["DEC"]))
        names.append(Path(path).name)
    ra0, dec0 = np.median(ras), np.median(decs)
    cos_dec = np.cos(np.radians(dec0))
    sep = 3600.0 * np.hypot(
        (np.asarray(ras) - ra0) * cos_dec, np.asarray(decs) - dec0
    )
    budget = max_scatter_arcsec + 0.5 * max(spec.cutout_shape) * spec.final_scale
    outliers = [n for n, s in zip(names, sep) if s > budget]
    if outliers:
        raise ValueError(
            f"{len(outliers)} science frame(s) point > {budget:.0f}\" from "
            f"the median pointing ({outliers[:5]}...) — a cone query likely "
            f"caught a different pointing (PSF star / neighbour); pin the "
            f"frame set with koa_science_ids"
        )


def _acquire(ctx: _StageContext) -> None:
    """Download (or reuse) exposures, sync references, footprint-filter."""
    if ctx.adapter.archive == "koa":
        _acquire_koa(ctx)
        return
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
    # Usability screen: MAST serves failed exposures (EXPFLAG "EXCESSIVE
    # DOWNTIME", EXPTIME 0) alongside the good ones — no science content,
    # never combined or packaged. Runs on cached lists too.
    exposures, unusable = quality_mod.filter_usable_exposures(exposures)
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
        "n_skipped_unusable": len(unusable),
        "unusable_exposures": [Path(p).name for p in unusable],
        "downloaded": downloaded,
        "references_synced": refs_synced,
    }


def _inject(ctx: _StageContext) -> None:
    """
    Opt-in synthetic-source injection (docs/design/simulate.md phase 1):
    everything downstream of acquire sees work-dir frame copies carrying
    the injected source; the exposure cache is never mutated.
    """
    if not ctx.spec.inject_image:
        return
    from .inject import imaging as inject_mod

    injected, fragment = inject_mod.inject_into_exposures(
        ctx.exposures, ctx.spec, ctx.adapter, ctx.work_dir
    )
    ctx.exposures = injected
    ctx.record["inject"] = fragment


def _align(ctx: _StageContext) -> None:
    if ctx.adapter.observatory == "keck":
        # Raw NIRC2 header WCS is approximate; relative registration is
        # phase cross-correlation inside the nirc2_native combine.
        ctx.record["align"] = {
            "wcs_solutions": "raw NIRC2 headers (approximate)",
            "method": "phase cross-correlation at combine",
        }
        return
    ctx.record["align"] = {
        "wcs_solutions": align_mod.wcs_solution_names(ctx.exposures),
        "tweakreg_run": False,  # a-priori WCS accepted by default (stage 2)
    }


def _prepare_keck_frames(
    ctx: _StageContext, raw_paths, calib, sky_window: int, tag: str,
    sky_group_gap_s: float = 3600.0,
):
    """
    Calibrate + sky-subtract one raw frame set; write prepared FITS.

    The sky is estimated within temporally contiguous groups only
    (``sky_group_gap_s`` splits nights and interleaved PSF-star visits) —
    window adjacency across a gap would borrow sky from a different
    night/visit, which the K'-band variability timescale forbids.
    """
    from astropy.io import fits

    from .acquire import koa as koa_mod
    from .calibrate import calibrate_frame
    from .sky import group_by_time_gaps, running_sky_subtract

    detector = ctx.adapter.ground_detector()
    facts = koa_mod.frame_facts_from_headers(raw_paths)
    frames = [
        calibrate_frame(
            fits.getdata(f["path"]).astype(float),
            calib,
            detector.gain_e_per_dn,
            f["coadds"],
        )
        for f in facts
    ]
    subtracted = [None] * len(frames)
    sky_levels = [None] * len(frames)
    group_recipes = []
    groups = group_by_time_gaps([f["mjd"] for f in facts], gap_s=sky_group_gap_s)
    for group in groups:
        group_sub, group_prov = running_sky_subtract(
            [frames[j] for j in group],
            window=min(sky_window, max(1, len(group) - 1)),
        )
        for j, sub, level in zip(
            group, group_sub, group_prov["sky_levels_e"]
        ):
            subtracted[j] = sub
            sky_levels[j] = level
        group_recipes.append(
            {"n_frames": len(group), "recipe": group_prov["recipe"]}
        )
    sky_prov = {
        "recipe": "per-contiguous-group scaled running sky "
        f"(gap > {sky_group_gap_s:.0f}s starts a new group)",
        "groups": group_recipes,
        "window": int(sky_window),
        "sky_levels_e": sky_levels,
    }

    prepared_dir = ctx.work_dir / f"prepared_{tag}"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    distortion = ctx.ground["distortion"]
    prepared_paths, mjds = [], []
    for fact, frame, sky_level in zip(
        facts, subtracted, sky_prov["sky_levels_e"]
    ):
        header = fits.Header()
        header["ITIME"] = fact["itime"]
        header["COADDS"] = fact["coadds"]
        header["MJD-OBS"] = fact["mjd"]
        header["SAMPMODE"] = fact["sampmode"]
        header["MULTISAM"] = fact["multisam"]
        header["SKYLEV"] = sky_level
        header["DISTX"] = distortion["distortion_paths"][0]
        header["DISTY"] = distortion["distortion_paths"][1]
        header["BUNIT"] = "ELECTRONS"
        out_path = prepared_dir / f"{fact['path'].stem}_prep.fits"
        fits.PrimaryHDU(frame.astype(np.float32), header=header).writeto(
            out_path, overwrite=True
        )
        prepared_paths.append(out_path)
        mjds.append(fact["mjd"])
    return prepared_paths, mjds, sky_prov


def _ground_prepare(ctx: _StageContext) -> None:
    """The ground-based pre-combine stages (keck_ao.md stages 2-3)."""
    if ctx.adapter.observatory != "keck":
        return
    from .acquire import koa as koa_mod
    from .calibrate import build_calibrations, load_calibration_sets

    facts = koa_mod.frame_facts_from_headers(ctx.exposures)
    setups = {(f["itime"], f["coadds"]) for f in facts}
    if len(setups) != 1:
        raise ValueError(
            f"science frames span {len(setups)} ITIME/COADDS setups "
            f"({sorted(setups)}); reduce one setup per target spec"
        )
    itime, coadds = setups.pop()
    darks, flat_on, flat_off = load_calibration_sets(
        ctx.ground["cal_paths"], science_itime=itime, science_coadds=coadds
    )
    calib = build_calibrations(darks, flat_on, flat_off)
    ctx.record["calibrate"] = {
        "gain_e_per_dn": ctx.adapter.ground_detector().gain_e_per_dn,
        **calib.provenance,
    }

    prepared, _, sky_prov = _prepare_keck_frames(
        ctx, ctx.exposures, calib, ctx.spec.sky_window, tag="science"
    )
    ctx.exposures = prepared
    ctx.record["sky"] = sky_prov

    if ctx.ground.get("psf_raw_paths"):
        import dataclasses

        # PSF-star visits use their own (shorter) ITIME; the science-matched
        # master dark does not apply — the running sky carries their dark.
        # Visits minutes apart are distinct AO/sky epochs: the tighter gap
        # matches nirc2_star.EPOCH_GAP_S so sky groups align with PSF epochs.
        from .psf.nirc2_star import EPOCH_GAP_S

        star_calib = dataclasses.replace(calib, master_dark=None)
        star_prepared, star_mjds, star_sky_prov = _prepare_keck_frames(
            ctx,
            ctx.ground["psf_raw_paths"],
            star_calib,
            ctx.spec.sky_window,
            tag="psfstar",
            sky_group_gap_s=EPOCH_GAP_S,
        )
        ctx.ground["psf_prepared"] = star_prepared
        ctx.ground["psf_mjds"] = star_mjds
        ctx.record["sky"]["psf_star_sky"] = star_sky_prov


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


def _psf(ctx: _StageContext, sci, header, noise=None):
    from astropy.io import fits
    from astropy.wcs import WCS

    spec, adapter = ctx.spec, ctx.adapter
    if adapter.observatory == "keck":
        # Any keck re-run clears candidates from previous runs first — a run
        # with fewer surviving epochs (or a tier-B fallback) must not leave
        # stale kernels behind for candidate-globbing consumers.
        for stale in ctx.out_dir.glob("psf_candidate_*.fits"):
            stale.unlink()
    if adapter.observatory == "keck" and ctx.ground.get("psf_prepared"):
        # Tier A (keck_ao.md stage 6): PSF-star epochs reduced
        # pipeline-identically; every epoch ships as a candidate.
        from .psf import nirc2_star

        psf, psf_full, candidates, diag = nirc2_star.build_candidates(
            ctx.ground["psf_prepared"],
            ctx.ground["psf_mjds"],
            spec,
            adapter,
            ctx.work_dir,
        )
        for i, candidate in enumerate(candidates):
            fits.PrimaryHDU(candidate.astype(np.float32)).writeto(
                ctx.out_dir / f"psf_candidate_{i}.fits", overwrite=True
            )
        ctx.record["psf"] = diag
        return psf, psf_full
    if spec.psf_from_frames:
        # The alternative mosaic PSF: combine the per-frame tier-1 ePSFs
        # through each frame's drizzle geometry instead of measuring stars
        # on the resampled mosaic (issue #21).
        from .psf import frame_combine as frame_combine_mod

        psf, psf_full, diag = frame_combine_mod.combined_mosaic_psf(
            ctx.exposures, spec, adapter, header
        )
        ctx.record["psf"] = diag
        return psf, psf_full
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
    elif adapter.observatory == "keck":
        # Tier B in-field ePSF on an e-/s mosaic: the same per-exposure
        # full-well cap as HST applies, from the prepared frames' own
        # ITIME x COADDS (the longest single frame bounds the star rate
        # that stays linear).
        max_single_exptime = max(
            float(fits.getheader(p)["ITIME"]) * int(fits.getheader(p)["COADDS"])
            for p in ctx.exposures
        )
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
    if spec.psf_backend == "starred":
        # Tier-1b: STARRED super-sampled ePSF from the same field stars
        # (hst_acs_pipeline.md Stage 5 Tier 1b, #35). Optional GPL/JAX extra;
        # raises loudly if unavailable — never silently falls back to Tier 1.
        from .psf import starred_epsf as starred_mod

        if noise is None:
            raise ValueError(
                "psf_backend='starred' needs the noise map (STARRED weights "
                "stars by per-pixel noise); pipeline did not pass it"
            )
        psf, psf_full, psf_diag = starred_mod.build_starred_epsf(
            sci, noise, stars, spec.psf_shape, spec.psf_full_shape
        )
    else:
        psf, psf_full, psf_diag = epsf_mod.build_epsf(
            sci, stars, spec.psf_shape, spec.psf_full_shape
        )
    if adapter.observatory == "keck":
        # Tier B: in-field ePSF. Usable, but an AO PSF from field stars at a
        # different anisoplanatic angle is still provisional by contract.
        psf_diag = {"psf_provisional": True, **psf_diag}
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
    products = ["data.fits", "noise_map.fits", "psf.fits", "psf_full.fits"]
    # Tier-A keck reductions also ship the PSF-star epoch candidates.
    products += sorted(p.name for p in out_dir.glob("psf_candidate_*.fits"))
    ctx.record["package"] = {
        "products": products,
        "cutout_shape": list(spec.cutout_shape),
        "pixel_scale": spec.final_scale,
        # Backend-agnostic: both backends stamp BUNIT on the mosaic header.
        "data_units": str(header.get("BUNIT", "unknown")),
    }


def _package_frames(ctx: _StageContext) -> None:
    """
    Opt-in per-exposure frame products (roadmap "Per-exposure frame
    products") — a packaging mode over the already-calibrated exposures.
    Runs after _package (driz_cr DQ flags and any tweakreg WCS refinement
    exist by then) and before eviction can delete the cached frames.
    """
    if not ctx.spec.frame_products:
        return
    if ctx.adapter.observatory == "keck":
        # The ground path packages the pipeline's own prepared frames
        # (ctx.exposures after _ground_prepare) — offset-based registration,
        # constructed noise, frame-vs-stack outlier masks, native tier-A
        # PSF-star stamps (issue #33).
        from .package import keck_frames as keck_frames_mod

        fragment = keck_frames_mod.package_keck_frame_products(
            ctx.exposures,
            ctx.spec,
            ctx.adapter,
            ctx.out_dir,
            drizzle_prov=ctx.record["drizzle"],
            psf_star_frames=ctx.ground.get("psf_prepared") or [],
            psf_star_mjds=ctx.ground.get("psf_mjds") or [],
            psf_record=ctx.record.get("psf") or {},
        )
        ctx.record["frames"] = fragment
        ctx.record["package"]["products"].append("frames/manifest.json")
        return
    exposures = ctx.exposures
    driz_cr_run = not ctx.record["drizzle"]["single_exposure_branch"]
    if ctx.adapter.observatory == "jwst":
        # The _crf products are the JWST analogue of driz_cr-flagged _flc
        # files: image3's outlier flags (DO_NOT_USE) and tweakreg-updated
        # WCS live there, not in the on-disk _cal inputs.
        crf_paths = ctx.record["drizzle"].get("crf_paths") or []
        if crf_paths:
            exposures = [Path(p) for p in crf_paths]
            source_note = "image3 _crf products (outlier-flagged, tweakreg WCS)"
        else:
            driz_cr_run = False
            source_note = (
                "_cal products — no _crf captured, stack outlier flags and "
                "tweakreg WCS updates absent"
            )
    else:
        source_note = "driz_cr-flagged _flc/_flt exposures (flags in place)"
    fragment = frames_mod.package_frame_products(
        exposures,
        ctx.spec,
        ctx.adapter,
        ctx.out_dir,
        driz_cr_run=driz_cr_run,
        source_note=source_note,
    )
    ctx.record["frames"] = fragment
    ctx.record["package"]["products"].append("frames/manifest.json")


def _evict(cache: cache_mod.ExposureCache, name: str, evict_when_done: bool) -> None:
    cache.mark_completed(name)
    if evict_when_done:
        cache.evict(name)
    cache.enforce_cap()


def reduce_target(
    spec: TargetSpec,
    cache_root: Path,
    output_root: Path,
    size_cap_bytes: Optional[int] = None,
    evict_when_done: bool = False,
) -> Dict:
    """Run the full pipeline for one target; returns the provenance record."""
    adapter = instruments.get(spec.instrument)
    observatory = getattr(adapter, "observatory", None)
    # Fail before any download — unsupported combinations are loud design
    # boundaries, not silently-skipped options.
    if spec.frame_products and observatory not in ("hst", "jwst", "keck"):
        raise ValueError(
            "frame_products supports HST, JWST and Keck only "
            f"(instrument {spec.instrument!r})"
        )
    if spec.psf_from_frames and observatory not in ("hst", "jwst"):
        # The AO path's mosaic PSF is the tier-A epoch design
        # (keck_ao.md); combining native star stamps into a drizzled
        # kernel is not that design.
        raise ValueError(
            "psf_from_frames supports HST and JWST only "
            f"(instrument {spec.instrument!r})"
        )
    if spec.inject_image and (
        observatory != "hst" or adapter.combine_backend != "astrodrizzle"
    ):
        # Phase 1 of docs/design/simulate.md; JWST/Keck injection and the
        # ALMA simobserve path are later phases.
        raise ValueError(
            "inject_image supports the HST astrodrizzle path only "
            f"(instrument {spec.instrument!r}; docs/design/simulate.md phase 1)"
        )
    cache = cache_mod.ExposureCache(Path(cache_root), size_cap_bytes=size_cap_bytes)
    out_dir = Path(output_root) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_dir.mkdir(exist_ok=True)

    if adapter.domain == "visibility":
        # The visibility branch (docs/design/alma.md): its own stages, the
        # shared cache/provenance machinery, the same eviction contract.
        from .visibilities import pipeline as visibility_pipeline

        record = visibility_pipeline.reduce_visibility_target(
            spec, adapter, cache, out_dir, work_dir
        )
        # The cache lifecycle applies only when acquisition went through the
        # cache — a local alma_ms_dir reduction never creates a manifest
        # entry, and mark_completed on a missing entry is (rightly) loud.
        if record["acquire"]["source"] == "alma-archive":
            _evict(cache, spec.name, evict_when_done)
        return record

    ctx = _StageContext(
        spec=spec,
        adapter=adapter,
        cache=cache,
        out_dir=out_dir,
        work_dir=work_dir,
        record={"target": spec.as_dict(), "instrument": adapter.key},
    )

    _acquire(ctx)
    _inject(ctx)
    _align(ctx)
    _ground_prepare(ctx)
    sci, header, wht, exptime = _combine(ctx)
    noise = _noise(ctx, sci, wht, exptime)
    psf, psf_full = _psf(ctx, sci, header, noise)
    _package(ctx, sci, header, wht, noise, psf, psf_full)
    _package_frames(ctx)
    provenance_mod.write_reduction_json(out_dir, ctx.record)
    _evict(ctx.cache, ctx.spec.name, evict_when_done)

    return ctx.record
