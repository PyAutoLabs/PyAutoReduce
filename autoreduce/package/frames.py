"""
Per-exposure frame products (roadmap "Per-exposure frame products") — the
opt-in packaging mode that ships every calibrated ``_flc``/``_flt`` chip as a
modeling-ready native-pixel product set alongside the drizzled mosaic:

    frames/manifest.json
    frames/<rootname>_chip<EXTVER>/{data.fits, noise_map.fits,
                                    dq.fits, cr_mask.fits}

A packaging mode, not a new pipeline: it consumes the exposures the standard
stages already prepared (CTE-corrected units from MAST, driz_cr CR flags in
DQ, tweakreg-refined WCS in the headers) and never touches the mosaic path.

Per-frame noise is the calacs/calwf3-propagated ERR extension — native-pixel
Poisson + read noise, no correlated-noise factor (nothing has been
resampled). Bad pixels (any DQ bit, the deepCR cosmic-ray mask, off-chip or
non-finite coverage) use the mosaic's masked-by-noise convention, so each
frame's ``data.fits`` + ``noise_map.fits`` load directly as an imaging
dataset and drop out of any chi^2. The raw ``dq.fits`` and ``cr_mask.fits``
keep the full bit information for consumers wanting a different policy.

The cutout header carries the frame's SIP WCS (``relax=True``); the NPOL /
D2IM lookup-table distortion is not FITS-serializable (~0.1 px), so the
manifest also records ``target_pixel`` — the target projected through the
full distortion model — as the exact per-frame registration anchor.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..instruments import InstrumentAdapter
from ..noise import rms as rms_mod
from ..target import TargetSpec
from . import cosmic_rays as cr_mod

MANIFEST_VERSION = 2


class _ChipSkip(Exception):
    """A chip that cannot ship a cutout (no overlap with the target region)."""


def frame_cutout_shape(
    cutout_shape: Tuple[int, int], final_scale: float, native_scale: float
) -> Tuple[int, int]:
    """
    Native-pixel cutout shape covering the mosaic cutout's sky footprint.

    Derived from the existing dials (no new user-facing shape): the mosaic
    cutout spans ``cutout_shape * final_scale`` arcsec, so the frame cutout
    spans the same sky at ``native_scale``, odd-forced so the target has a
    centre pixel like every other product.
    """
    if final_scale <= 0.0 or native_scale <= 0.0:
        raise ValueError(
            f"scales must be positive: final {final_scale}, native {native_scale}"
        )

    def _one(n: int) -> int:
        m = int(np.ceil(n * final_scale / native_scale))
        return m if m % 2 == 1 else m + 1

    return (_one(cutout_shape[0]), _one(cutout_shape[1]))


def _units_to_cps(bunit: str, exptime: float) -> Tuple[float, str, str]:
    """(factor, provenance note, output BUNIT) for the frame's SCI/ERR.

    HST frames are taken to e-/s so all frames + mosaic share the cps flux
    scale; JWST frames stay in their native surface-brightness units
    (defaults-first — MJy/sr is what calwebb delivers and what the mosaic
    keeps).
    """
    unit = bunit.strip().upper()
    if unit in ("ELECTRONS/S", "ELECTRON/S", "ELECTRONS/SEC"):
        return 1.0, "none (already e-/s)", "ELECTRONS/S"
    if unit in ("ELECTRONS", "ELECTRON"):
        if not np.isfinite(exptime) or exptime <= 0.0:
            raise ValueError(
                f"cannot convert ELECTRONS to e-/s without a positive EXPTIME: "
                f"{exptime}"
            )
        return 1.0 / exptime, "SCI,ERR / EXPTIME", "ELECTRONS/S"
    if unit == "MJY/SR":
        return 1.0, "none (native MJy/sr)", "MJy/sr"
    raise ValueError(
        f"unrecognised frame BUNIT {bunit!r} — expected ELECTRONS[/S] (HST) "
        "or MJy/sr (JWST) calibrated products"
    )


def _exposure_time(primary) -> float:
    """Exposure time across observatories: HST EXPTIME, JWST XPOSURE/EFFEXPTM."""
    for key in ("EXPTIME", "XPOSURE", "EFFEXPTM"):
        if key in primary:
            return float(primary[key])
    return 0.0


def _sky_level(primary, hdr) -> Tuple[float, Optional[str]]:
    """(sky level, keyword) — HST MDRIZSKY / JWST skymatch BKGLEVEL.

    Both pipelines record the matched sky rather than subtracting it from
    the calibrated frames; absence means sky matching did not run (a
    legitimate configuration), recorded as 0.0 with no keyword.
    """
    for source in (hdr, primary):
        for key in ("MDRIZSKY", "BKGLEVEL"):
            if key in source:
                return float(source[key]), key
    return 0.0, None


def _sip_only_header(hdr):
    """
    Copy of a SCI header with the lookup-table distortion keywords removed.

    HST calibrated headers carry CPDIS/DP (NPOL) and D2IM lookup-table
    distortion that only resolves against the open HDUList; astropy's WCS
    raises if they are parsed without it. The frame cutout ships the SIP
    part only, so strip the lookup keywords rather than carrying dead
    references to extensions the cutout file does not have.
    """
    out = hdr.copy()
    for card in list(out.cards):
        if str(card.keyword).startswith(("CPDIS", "CPERR", "DP1", "DP2", "D2IM")):
            out.remove(card.keyword, ignore_missing=True, remove_all=True)
    return out


def _write_product(data: np.ndarray, header, out_path: Path, dtype) -> None:
    from astropy.io import fits

    fits.PrimaryHDU(data.astype(dtype), header=header).writeto(
        out_path, overwrite=True
    )


def _package_one_chip(
    hdul, extver: int, spec: TargetSpec, adapter: InstrumentAdapter,
    shape: Tuple[int, int], chip_dir: Path, cr_masker,
) -> Dict:
    """One SCI chip -> data/noise/dq/cr_mask FITS + its manifest entry."""
    from astropy.nddata import Cutout2D
    from astropy.nddata.utils import NoOverlapError
    from astropy.wcs import WCS

    primary = hdul[0].header
    sci_hdu = hdul["SCI", extver]
    err = hdul["ERR", extver].data
    dq = hdul["DQ", extver].data
    hdr = sci_hdu.header

    # Project the target through the full distortion model (SIP + NPOL/D2IM
    # lookup tables — resolvable only with the open HDUList, exactly as the
    # footprint filter does); this is the registration anchor.
    wcs_full = WCS(hdr, fobj=hdul, naxis=2)
    x, y = wcs_full.world_to_pixel_values(spec.ra, spec.dec)
    if not (np.isfinite(x) and np.isfinite(y)):
        raise _ChipSkip("target does not project onto the chip")

    # The written header uses the SIP-only WCS (FITS-serializable); the
    # lookup-table residual (~0.1 px) is recorded in the manifest note.
    # The NPOL/D2IM keywords must be stripped first: astropy raises on a
    # CPDIS/D2IM header parsed without the open HDUList (fobj).
    wcs_sip = WCS(_sip_only_header(hdr), naxis=2)
    try:
        cut = Cutout2D(
            sci_hdu.data.astype(float),
            position=(float(x), float(y)),
            size=shape,
            wcs=wcs_sip,
            mode="partial",
            fill_value=np.nan,
        )
    except NoOverlapError:
        raise _ChipSkip("no overlap with the target cutout region")
    cut_slices = cut.slices_original
    err_cut = np.full(shape, np.nan)
    dq_cut = np.zeros(shape, dtype=np.int32)
    err_cut[cut.slices_cutout] = err[cut_slices].astype(float)
    dq_cut[cut.slices_cutout] = dq[cut_slices].astype(np.int32)
    sci_cut_native = cut.data
    offchip = ~np.isfinite(sci_cut_native)

    # deepCR sees the frame as trained on: native units, sky pedestal in.
    cr_mask = cr_masker(sci_cut_native)

    # Both pipelines record the matched sky rather than subtracting it from
    # the calibrated frames (AstroDrizzle: MDRIZSKY; image3 skymatch:
    # BKGLEVEL) — subtract it so frames share the mosaic's zero level.
    sky, sky_keyword = _sky_level(primary, hdr)
    exptime = _exposure_time(primary)
    factor, conversion, out_bunit = _units_to_cps(
        str(hdr.get("BUNIT", "")), exptime
    )
    sci_cps = (sci_cut_native - sky) * factor
    err_cps = err_cut * factor

    # DQ policy diverges by observatory: HST DQ bits all mark suspect data
    # (any nonzero -> masked). JWST ramps *remove* cosmic rays during slope
    # fitting, so informational bits (JUMP_DET etc.) ride good pixels — only
    # DO_NOT_USE (bit 0, also set by image3's outlier_detection in _crf
    # products) means bad.
    if adapter.observatory == "jwst":
        dq_bad = (dq_cut & 1) != 0
    else:
        dq_bad = dq_cut != 0
    bad = (
        dq_bad
        | cr_mask
        | offchip
        | ~np.isfinite(err_cps)
        | (err_cps <= 0.0)
        | ~np.isfinite(sci_cps)
    )
    data_out = np.where(bad, 0.0, sci_cps)
    noise_out = np.where(bad, rms_mod.MASKED_NOISE_VALUE, err_cps)

    out_header = cut.wcs.to_header(relax=True)
    for key in ("INSTRUME", "TELESCOP", "EXPTIME", "EXPSTART", "ROOTNAME"):
        if key in primary:
            out_header[key] = primary[key]
    if "CCDCHIP" in hdr:
        out_header["CCDCHIP"] = hdr["CCDCHIP"]
    data_header = out_header.copy()
    data_header["BUNIT"] = out_bunit

    chip_dir.mkdir(parents=True)
    _write_product(data_out, data_header, chip_dir / "data.fits", np.float32)
    _write_product(noise_out, data_header, chip_dir / "noise_map.fits", np.float32)
    _write_product(dq_cut, out_header, chip_dir / "dq.fits", np.int32)
    _write_product(cr_mask, out_header, chip_dir / "cr_mask.fits", np.uint8)

    target_cut_x, target_cut_y = cut.to_cutout_position((float(x), float(y)))
    return {
        "chip": int(extver),
        "ccdchip": int(hdr["CCDCHIP"]) if "CCDCHIP" in hdr else None,
        "dir": chip_dir.name,
        "exptime": exptime,
        "expstart": float(primary["EXPSTART"]) if "EXPSTART" in primary else None,
        "filter": spec.filter_name,
        "unit_conversion": conversion,
        "data_units": out_bunit,
        "sky_subtracted": sky,
        "sky_keyword": sky_keyword,
        "target_pixel": [float(target_cut_x), float(target_cut_y)],
        "n_masked_pixels": int(bad.sum()),
        "n_cr_pixels": int(cr_mask.sum()),
        "offchip_fraction": float(offchip.mean()),
        # The astrometric solution behind this frame's WCS. The RMS/NMATCHES
        # keywords state the *group's absolute* alignment to the external
        # catalog; the frame-to-frame relative residual modeling consumes is
        # measured separately (see _relative_registration).
        "registration": {
            "wcsname": str(hdr.get("WCSNAME", "unknown")),
            "wcstype": str(hdr.get("WCSTYPE", "unknown")),
            "rms_ra_mas": float(hdr["RMS_RA"]) if "RMS_RA" in hdr else None,
            "rms_dec_mas": float(hdr["RMS_DEC"]) if "RMS_DEC" in hdr else None,
            "nmatches": int(hdr["NMATCHES"]) if "NMATCHES" in hdr else None,
        },
        "psf": _frame_psf(hdul, extver, spec, adapter, chip_dir),
    }


def _frame_psf(hdul, extver, spec, adapter, chip_dir: Path) -> Dict:
    """Tier-1 native ePSF for one chip; writes psf files when viable."""
    from ..psf import frame_epsf as frame_epsf_mod

    psf, psf_full, diag = frame_epsf_mod.build_frame_epsf(
        hdul, extver, spec, adapter
    )
    if psf is not None:
        # Plain kernels, mirroring the mosaic PSF products (no WCS — the
        # kernel is defined on the frame's native pixel grid).
        from astropy.io import fits

        fits.PrimaryHDU(psf.astype(np.float32)).writeto(
            chip_dir / "psf.fits", overwrite=True
        )
        fits.PrimaryHDU(psf_full.astype(np.float32)).writeto(
            chip_dir / "psf_full.fits", overwrite=True
        )
    return diag


def _relative_registration(frames_dir: Path, entries: List[Dict]) -> None:
    """
    Measure every frame's registration residual against the first written
    frame and record it in each entry's ``registration`` block.

    The residual is what a multi-frame modeler actually needs to know: each
    frame is resampled onto the reference frame's pixel grid through *both*
    shipped cutout WCS (a perfect WCS pair leaves zero shift) and the
    remaining offset is phase-correlated. It bounds the error of treating
    the inter-exposure shifts as perfectly known — including the SIP-only
    serialization term the manifest's WCS note describes. The header
    RMS_RA/RMS_DEC record the group's absolute catalog alignment instead,
    which is not the modeling-relevant quantity (issue #19).
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy.ndimage import map_coordinates

    from ..align.registration import phase_offset

    # A correlation between mostly-masked cutouts locks onto the mask
    # geometry, not the source (JWST validation, issue #27: dithers put the
    # target near detector edges, off-chip fractions up to ~0.6, "residuals"
    # of ~200 px). The reference is the best-covered frame, and a pair is
    # only *reliable* when both frames are mostly unmasked; unreliable
    # residuals are still recorded but flagged and excluded from the
    # headline maximum.
    max_masked_fraction = 0.2
    ref_entry = min(entries, key=lambda e: e["n_masked_pixels"])
    with fits.open(frames_dir / ref_entry["dir"] / "data.fits") as hdul:
        ref = hdul[0].data.astype(float)
        ref_wcs = WCS(hdul[0].header)
    ny, nx = ref.shape
    npix = float(ny * nx)
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra, dec = ref_wcs.pixel_to_world_values(xx, yy)
    ref_ok = ref_entry["n_masked_pixels"] / npix <= max_masked_fraction

    # The resample's boundary blends toward cval at the cutout border; a
    # trimmed margin keeps the correlation locked on source structure
    # rather than border artifacts (which dominate on flat backgrounds).
    margin = 2

    ref_entry["registration"]["reference"] = None
    ref_entry["registration"]["residual_dy_px"] = 0.0
    ref_entry["registration"]["residual_dx_px"] = 0.0
    ref_entry["registration"]["residual_reliable"] = bool(ref_ok)
    for entry in entries:
        if entry is ref_entry:
            continue
        with fits.open(frames_dir / entry["dir"] / "data.fits") as hdul:
            data = hdul[0].data.astype(float)
            wcs = WCS(hdul[0].header)
        xk, yk = wcs.world_to_pixel_values(ra, dec)
        resampled = map_coordinates(
            data, [yk, xk], order=1, mode="constant", cval=0.0
        )
        # whiten=True: on real (noisy) frames phase correlation is the
        # hole-robust estimator — the per-frame CR masks punch different
        # zeros into each cutout, which bias plain correlation and centroids
        # by ~0.2-0.3 px but hit the whitened phase spectrum incoherently
        # (measured on slacs0008, issue #19).
        dy, dx = phase_offset(
            ref[margin:-margin, margin:-margin],
            resampled[margin:-margin, margin:-margin],
        )
        entry["registration"]["reference"] = ref_entry["dir"]
        entry["registration"]["residual_dy_px"] = float(dy)
        entry["registration"]["residual_dx_px"] = float(dx)
        entry["registration"]["residual_reliable"] = bool(
            ref_ok and entry["n_masked_pixels"] / npix <= max_masked_fraction
        )


def package_frame_products(
    exposures: List[Path],
    spec: TargetSpec,
    adapter: InstrumentAdapter,
    out_dir: Path,
    driz_cr_run: bool,
    source_note: Optional[str] = None,
) -> Dict:
    """
    Write ``out_dir/frames/`` for every (exposure, SCI chip) covering the
    target; returns the ``reduction.json`` provenance fragment.

    ``driz_cr_run`` states whether stack-based rejection flagged the frames
    (HST driz_cr into the _flc DQ; JWST image3 outlier_detection into the
    _crf DQ). ``source_note`` describes the input family for the manifest.
    """
    from astropy.io import fits

    frames_dir = Path(out_dir) / "frames"
    # Idempotent re-runs: a rerun with a different exposure set must not
    # leave orphan frame directories behind (the keck stale-candidate rule).
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    shape = frame_cutout_shape(
        spec.cutout_shape, spec.final_scale, adapter.native_scale
    )
    cr_method = cr_mod.cr_method_record(adapter.key)
    if cr_method["method"] == "deepCR":
        cr_masker = cr_mod.masker_for(adapter.key)
    else:
        # IR: cosmic rays already flagged in DQ by ramp fitting.
        def cr_masker(sci):
            return np.zeros(sci.shape, dtype=bool)

    entries, skipped = [], []
    seen_rootnames: Dict[str, str] = {}
    for path in exposures:
        with fits.open(path) as hdul:
            primary = hdul[0].header
            # HST primaries carry ROOTNAME; JWST files don't — their stem
            # minus the product suffix (jw..._nrcb1) is the exposure+detector
            # identity. The naive split('_')[0] would collide across a JWST
            # visit and trip the duplicate guard below.
            stem = Path(path).stem
            for suffix in ("_cal", "_crf", "_flc", "_flt"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            rootname = str(primary.get("ROOTNAME", stem)).strip().lower()
            # One file per exposure is an identity assumption, not a hope:
            # a repeated ROOTNAME means the same exposure was ingested twice
            # (e.g. a direct FLC plus its renamed HAP copy) — the mosaic
            # would be drizzling it twice too. Fail loudly at the source.
            if rootname in seen_rootnames:
                raise ValueError(
                    f"duplicate ROOTNAME {rootname!r}: {seen_rootnames[rootname]} "
                    f"and {Path(path).name} are the same exposure ingested "
                    "twice — fix the acquire exposure list"
                )
            seen_rootnames[rootname] = Path(path).name
            extvers = [
                int(hdu.header.get("EXTVER", 1))
                for hdu in hdul
                if hdu.name == "SCI"
            ]
            for extver in extvers:
                chip_dir = frames_dir / f"{rootname}_chip{extver}"
                try:
                    entry = _package_one_chip(
                        hdul, extver, spec, adapter, shape, chip_dir, cr_masker
                    )
                except _ChipSkip as skip:
                    skipped.append(
                        {"rootname": rootname, "chip": extver, "reason": str(skip)}
                    )
                    continue
                entries.append(
                    {"rootname": rootname, "source_file": Path(path).name, **entry}
                )

    if not entries:
        # The acquire footprint filter guarantees every exposure covers the
        # target, so writing zero chips means a geometry bug — fail loudly.
        raise ValueError(
            f"frame_products wrote no chips from {len(exposures)} exposures "
            f"covering {spec.name} — footprint/cutout geometry bug"
        )

    _relative_registration(frames_dir, entries)
    reliable = [
        float(
            np.hypot(
                e["registration"]["residual_dy_px"],
                e["registration"]["residual_dx_px"],
            )
        )
        for e in entries
        # A measurement needs a PAIR of well-covered frames: the reference's
        # own zero-by-construction residual is not one.
        if e["registration"]["residual_reliable"]
        and e["registration"]["reference"] is not None
    ]
    # None when no sufficiently-covered frame pair exists (e.g. every dither
    # puts the target near a detector edge) — an honest "unmeasured", never
    # a mask-geometry artifact presented as a shift.
    max_residual = max(reliable) if reliable else None

    units = {e["data_units"] for e in entries}
    if len(units) != 1:
        raise ValueError(
            f"heterogeneous frame units {sorted(units)} for {spec.name} — "
            "mixed calibration families in one exposure list"
        )
    data_units = units.pop()

    if adapter.observatory == "jwst":
        dq_semantics = {
            "policy": (
                "DO_NOT_USE bit, off-chip or non-finite pixel -> "
                "masked-by-noise (noise=MASKED_NOISE_VALUE, data=0); "
                "dq.fits keeps all bits"
            ),
            "masked_noise_value": rms_mod.MASKED_NOISE_VALUE,
            "1": (
                "DO_NOT_USE — calwebb's bad-pixel verdict; image3 "
                "outlier_detection also sets it in _crf products"
            ),
            "4": (
                "JUMP_DET — informational: the cosmic ray was removed "
                "during ramp fitting, the pixel is good data"
            ),
            "reference": "jwst.datamodels dqflags table",
            **(
                {}
                if driz_cr_run
                else {
                    "driz_cr_note": (
                        "single-exposure reduction: no image3 stack "
                        "rejection; ramp-level jump detection is the only "
                        "cosmic-ray rejection"
                    )
                }
            ),
        }
    else:
        dq_semantics = {
            "policy": (
                "any nonzero DQ bit, deepCR CR pixel, off-chip or non-finite "
                "pixel -> masked-by-noise (noise=MASKED_NOISE_VALUE, data=0); "
                "dq.fits/cr_mask.fits keep the raw bits"
            ),
            "masked_noise_value": rms_mod.MASKED_NOISE_VALUE,
            "4096": "AstroDrizzle driz_cr cosmic ray",
            "reference": "instrument DQ flag table (ACS/WFC3 handbooks)",
            **(
                {}
                if driz_cr_run
                else {
                    "driz_cr_note": (
                        "single-exposure reduction: DQ carries no driz_cr "
                        "cosmic-ray flags; the per-frame CR mask is the only "
                        "cosmic-ray rejection"
                    )
                }
            ),
        }

    manifest = {
        "version": MANIFEST_VERSION,
        "target": {"name": spec.name, "ra": spec.ra, "dec": spec.dec},
        "data_units": data_units,
        "source": source_note or "as-delivered calibrated exposures",
        "frame_cutout_shape": list(shape),
        "native_scale": adapter.native_scale,
        "driz_cr_run": bool(driz_cr_run),
        "cr_method": cr_method,
        "dq_semantics": dq_semantics,
        "wcs": (
            "SIP WCS in each data.fits header; NPOL/D2IM lookup distortion is "
            "not FITS-serializable (~0.1 px residual); target_pixel is "
            "computed through the full distortion model"
        ),
        "registration_note": (
            "per-frame registration block: wcsname/wcstype/rms/nmatches state "
            "the group's ABSOLUTE catalog alignment; residual_dy_px/"
            "residual_dx_px are the measured frame-to-frame RELATIVE "
            "registration errors through the shipped cutout WCS (phase "
            "correlation vs the reference frame) — the quantity multi-frame "
            "modeling consumes. The measurement itself is limited to "
            "~0.1-0.3 px where CR-masked pixels bite the source, so sub-0.1 "
            "px values are consistent with zero. Treating shifts as "
            "known-perfect is safe when these residuals are far below the "
            "modeling scale; otherwise free per-frame (dy, dx) with priors "
            "of this width"
        ),
        "max_registration_residual_px": max_residual,
        "frames": entries,
        "skipped_chips": skipped,
    }
    with open(frames_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Deliberately loud for now (user request, issue #19): the registration
    # story is easy to forget and the header RMS keywords invite misreading.
    if max_residual is None:
        headline = (
            "no reliable frame pair (every frame is heavily masked/off-chip "
            "at the target) — relative registration UNMEASURED, per-frame "
            "flags in the manifest"
        )
    else:
        headline = f"max relative residual {max_residual:.3f} native px"
    print(
        f"[frames] inter-exposure registration ({spec.name}): {headline} "
        f"across {len(entries)} chips. "
        "Measurement floor is ~0.1-0.3 px where CR-masked pixels bite the "
        "source — sub-0.1 px values are consistent with zero. NOTE: the "
        "header RMS_RA/RMS_DEC keywords state the group's ABSOLUTE catalog "
        "alignment (slacs0008: GSC-2.4.2, ~44 mas), NOT this relative "
        "number — the relative registration is measured, and lives in "
        "frames/manifest.json 'registration' per frame. Default stance: "
        "shifts are known; free per-frame (dy, dx) with priors of the "
        "recorded width for precision work."
    )

    without_psf = [e["dir"] for e in entries if e["psf"]["method"] == "none"]
    if without_psf:
        # Loud by design: a frame without a PSF is not modelable until the
        # tier-2 model PSF exists (issue #21) — say so at reduction time.
        print(
            f"[frames] per-frame ePSF NOT viable for {len(without_psf)}/"
            f"{len(entries)} chips ({', '.join(without_psf)}) — too few "
            "usable stars on the frame; these frames ship without psf.fits "
            "and are not modelable until the tier-2 model PSF lands. "
            "Reasons per frame: frames/manifest.json 'psf'."
        )

    return {
        "n_exposures": len(exposures),
        "n_chips_written": len(entries),
        "n_chips_skipped": len(skipped),
        "n_frames_with_psf": len(entries) - len(without_psf),
        "driz_cr_run": bool(driz_cr_run),
        "cr_method": cr_method,
        "data_units": data_units,
        "max_registration_residual_px": max_residual,
        "manifest": "frames/manifest.json",
    }
