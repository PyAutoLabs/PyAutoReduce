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
from typing import Dict, List, Tuple

import numpy as np

from ..instruments import InstrumentAdapter
from ..noise import rms as rms_mod
from ..target import TargetSpec
from . import cosmic_rays as cr_mod

MANIFEST_VERSION = 1


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


def _units_to_cps(bunit: str, exptime: float) -> Tuple[float, str]:
    """(multiplicative factor, provenance note) taking SCI/ERR to e-/s."""
    unit = bunit.strip().upper()
    if unit in ("ELECTRONS/S", "ELECTRON/S", "ELECTRONS/SEC"):
        return 1.0, "none (already e-/s)"
    if unit in ("ELECTRONS", "ELECTRON"):
        if not np.isfinite(exptime) or exptime <= 0.0:
            raise ValueError(
                f"cannot convert ELECTRONS to e-/s without a positive EXPTIME: "
                f"{exptime}"
            )
        return 1.0 / exptime, "SCI,ERR / EXPTIME"
    raise ValueError(
        f"unrecognised frame BUNIT {bunit!r} — expected ELECTRONS or "
        f"ELECTRONS/S for HST calibrated products"
    )


def _write_product(data: np.ndarray, header, out_path: Path, dtype) -> None:
    from astropy.io import fits

    fits.PrimaryHDU(data.astype(dtype), header=header).writeto(
        out_path, overwrite=True
    )


def _package_one_chip(
    hdul, extver: int, spec: TargetSpec, shape: Tuple[int, int], chip_dir: Path,
    cr_masker,
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
    wcs_sip = WCS(hdr, naxis=2)
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

    # globalmin+match sky is only *virtually* subtracted during drizzle —
    # AstroDrizzle stores it as MDRIZSKY; subtract it so frames share the
    # mosaic's zero level (absent keyword = sky subtraction not run = 0).
    mdrizsky = float(hdr.get("MDRIZSKY", 0.0))
    exptime = float(primary.get("EXPTIME", 0.0))
    factor, conversion = _units_to_cps(str(hdr.get("BUNIT", "")), exptime)
    sci_cps = (sci_cut_native - mdrizsky) * factor
    err_cps = err_cut * factor

    bad = (
        (dq_cut != 0)
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
    data_header["BUNIT"] = "ELECTRONS/S"

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
        "mdrizsky_subtracted": mdrizsky,
        "target_pixel": [float(target_cut_x), float(target_cut_y)],
        "n_masked_pixels": int(bad.sum()),
        "n_cr_pixels": int(cr_mask.sum()),
        "offchip_fraction": float(offchip.mean()),
    }


def package_frame_products(
    exposures: List[Path],
    spec: TargetSpec,
    adapter: InstrumentAdapter,
    out_dir: Path,
    driz_cr_run: bool,
) -> Dict:
    """
    Write ``out_dir/frames/`` for every (exposure, SCI chip) covering the
    target; returns the ``reduction.json`` provenance fragment.
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
    for path in exposures:
        with fits.open(path) as hdul:
            primary = hdul[0].header
            rootname = (
                str(primary.get("ROOTNAME", Path(path).name.split("_")[0]))
                .strip()
                .lower()
            )
            extvers = [
                int(hdu.header.get("EXTVER", 1))
                for hdu in hdul
                if hdu.name == "SCI"
            ]
            for extver in extvers:
                chip_dir = frames_dir / f"{rootname}_chip{extver}"
                try:
                    entry = _package_one_chip(
                        hdul, extver, spec, shape, chip_dir, cr_masker
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

    manifest = {
        "version": MANIFEST_VERSION,
        "target": {"name": spec.name, "ra": spec.ra, "dec": spec.dec},
        "data_units": "ELECTRONS/S",
        "frame_cutout_shape": list(shape),
        "native_scale": adapter.native_scale,
        "driz_cr_run": bool(driz_cr_run),
        "cr_method": cr_method,
        "dq_semantics": {
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
        },
        "wcs": (
            "SIP WCS in each data.fits header; NPOL/D2IM lookup distortion is "
            "not FITS-serializable (~0.1 px residual); target_pixel is "
            "computed through the full distortion model"
        ),
        "frames": entries,
        "skipped_chips": skipped,
    }
    with open(frames_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "n_exposures": len(exposures),
        "n_chips_written": len(entries),
        "n_chips_skipped": len(skipped),
        "driz_cr_run": bool(driz_cr_run),
        "cr_method": cr_method,
        "data_units": "ELECTRONS/S",
        "manifest": "frames/manifest.json",
    }
