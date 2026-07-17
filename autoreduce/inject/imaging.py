"""
Imaging injection (docs/design/simulate.md — HST astrodrizzle + JWST
jwst_image3 paths).

Balrog-style synthetic-source injection (Suchyta et al. 2016; DES Y6,
Anbajagane et al. 2025): the input image is rendered onto each real
calibrated frame's native grid through the frame's own WCS, convolved with
that frame's PSF, converted to native units, given its own Poisson counts,
and added to work-dir *copies* of the frames — the exposure cache is never
mutated. Everything downstream (align, drizzle + driz_cr, noise, psf,
package) then runs unchanged on frames whose cosmic rays, sky and
correlated noise are real.

Input contract: a plain FITS image whose pixel values are flux per input
pixel in the adapter's ``inject_units`` — e-/s for HST's electron-referred
frames, Jy for JWST's MJy/sr frames (total source flux = array sum) — at
`inject_pixel_scale` arcsec/pix, placed north-up at `inject_position`
(default: the target). Values must be non-negative — the injected
source's Poisson realisation is drawn from them. The input must not be
PSF-convolved already; the stage convolves with each frame's PSF itself.
Phase 1 covered the HST astrodrizzle path; phase 2a (this) adds JWST
jwst_image3.
"""

import shutil
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..instruments import InstrumentAdapter
from ..target import TargetSpec

# Chips whose rendered injection totals less than this fraction of the
# input flux are untouched — the source does not overlap their footprint.
MIN_OVERLAP_FRACTION = 1e-9


def input_wcs(shape: Tuple[int, int], pixel_scale: float, ra: float, dec: float):
    """North-up TAN WCS centred on the injection position."""
    from astropy.wcs import WCS

    if pixel_scale <= 0.0:
        raise ValueError(f"inject pixel scale must be positive: {pixel_scale}")
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [(shape[1] + 1) / 2.0, (shape[0] + 1) / 2.0]
    scale_deg = pixel_scale / 3600.0
    wcs.wcs.cd = np.array([[-scale_deg, 0.0], [0.0, scale_deg]])
    return wcs


def load_input_image(spec: TargetSpec) -> Tuple[np.ndarray, object]:
    """The input image and its synthetic WCS, contract-checked."""
    from astropy.io import fits

    data = fits.getdata(spec.inject_image).astype(np.float64)
    if data.ndim != 2:
        raise ValueError(
            f"inject_image must be a 2D image: {spec.inject_image} has "
            f"shape {data.shape}"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError(f"inject_image contains non-finite pixels: {spec.inject_image}")
    if np.any(data < 0.0):
        raise ValueError(
            f"inject_image contains negative pixels: {spec.inject_image} — "
            "values are flux per pixel and seed a Poisson draw, so they "
            "must be non-negative"
        )
    ra, dec = spec.inject_position or (spec.ra, spec.dec)
    return data, input_wcs(data.shape, spec.inject_pixel_scale, ra, dec)


def render_to_chip(
    input_cps: np.ndarray,
    in_wcs,
    chip_wcs,
    chip_shape: Tuple[int, int],
    pixfrac: float = 1.0,
) -> np.ndarray:
    """
    Flux-conserving resample of the input onto one chip's native grid.

    The pixmap runs over the (small) input grid — input pixel centres to
    world through the synthetic TAN WCS, world to chip pixels through the
    frame's own solution — so the full lookup-table distortion costs one
    array-shaped ``all_world2pix``, never a per-frame-pixel loop.
    """
    ny, nx = input_cps.shape
    yy, xx = np.mgrid[0.0:ny, 0.0:nx]
    world = in_wcs.all_pix2world(
        np.column_stack([xx.ravel(), yy.ravel()]), 0
    )
    # quiet=True: off-chip positions are expected for footprints larger
    # than one chip; drizzle drops non-finite map entries itself.
    chip_xy = chip_wcs.all_world2pix(world, 0, quiet=True)
    pixmap = np.empty((ny, nx, 2), dtype=np.float64)
    pixmap[..., 0] = chip_xy[:, 0].reshape(ny, nx)
    pixmap[..., 1] = chip_xy[:, 1].reshape(ny, nx)

    # Nominal (reference-pixel) areas — the few-percent per-pixel
    # distortion variation across a chip is the same PAM effect
    # uncorrected FLC photometry carries.
    from astropy.wcs.utils import proj_plane_pixel_area

    area_ratio = proj_plane_pixel_area(chip_wcs) / proj_plane_pixel_area(in_wcs)
    return render_via_pixmap(
        input_cps, pixmap, tuple(chip_shape), area_ratio, pixfrac=pixfrac
    )


def render_via_pixmap(
    input_image: np.ndarray,
    pixmap: np.ndarray,
    out_shape: Tuple[int, int],
    area_ratio: float,
    pixfrac: float = 1.0,
) -> np.ndarray:
    """
    The shared drizzle-render core: deposit ``input_image`` onto
    ``out_shape`` through an arbitrary ``(ny, nx, 2)`` pixmap. The drizzle
    kernel preserves surface brightness, so the output sums to
    input_flux x (input pixel area / output pixel area); ``area_ratio``
    (output/input pixel area) restores flux-per-pixel semantics.
    """
    from drizzle.resample import Drizzle

    driz = Drizzle(kernel="square", out_shape=tuple(out_shape), fillval=0.0)
    driz.add_image(
        input_image,
        exptime=1.0,
        pixmap=pixmap,
        pixfrac=pixfrac,
        in_units="cps",
    )
    return driz.out_img.astype(np.float64) * area_ratio


def convolve_with_psf(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Same-shape FFT convolution with a unit-sum kernel (numpy-only)."""
    ky, kx = kernel.shape
    if ky % 2 == 0 or kx % 2 == 0:
        raise ValueError(f"PSF kernel shape must be odd: {kernel.shape}")
    ny, nx = image.shape
    fy, fx = ny + ky - 1, nx + kx - 1
    out = np.fft.irfft2(
        np.fft.rfft2(image, s=(fy, fx)) * np.fft.rfft2(kernel, s=(fy, fx)),
        s=(fy, fx),
    )
    oy, ox = ky // 2, kx // 2
    return out[oy : oy + ny, ox : ox + nx]


def _chip_units(
    bunit: str, sci_header, exptime: float, adapter: InstrumentAdapter
) -> Tuple[float, float, str]:
    """
    ``(electrons_per_input, sci_per_electron, note)`` for one chip — the
    two directions of the chip's photometry (the reading contract's
    counterpart, `package.frames._units_to_cps`):

    - ``model_e = rendered_input x electrons_per_input`` sizes the
      injected source's Poisson draw;
    - ``SCI += counts x sci_per_electron`` (and
      ``ERR^2 += model_e x sci_per_electron^2``) converts realised
      electrons back to the chip's SCI units.

    HST ELECTRONS[/S] frames take the adapter's e-/s input directly.
    JWST MJy/sr frames take Jy-per-pixel input through the frame's own
    PIXAR_SR — electrons_per_input x sci_per_electron reduces to
    1/(PIXAR_SR x 1e6), so the injected MEAN is flux-exact and the
    nominal detector gain (adapter e_per_dn) shapes only the Poisson
    width.
    """
    unit = bunit.strip().upper()
    if unit in ("ELECTRONS", "ELECTRON"):
        if not np.isfinite(exptime) or exptime <= 0.0:
            raise ValueError(
                f"cannot inject into ELECTRONS without a positive EXPTIME: {exptime}"
            )
        return exptime, 1.0, "e-/s x EXPTIME"
    if unit in ("ELECTRONS/S", "ELECTRON/S", "ELECTRONS/SEC"):
        if not np.isfinite(exptime) or exptime <= 0.0:
            raise ValueError(
                f"cannot inject into ELECTRONS/S without a positive EXPTIME: {exptime}"
            )
        return exptime, 1.0 / exptime, "e-/s x EXPTIME; SCI back in cps"
    if unit == "MJY/SR":
        photmjsr = sci_header.get("PHOTMJSR")
        pixar_sr = sci_header.get("PIXAR_SR")
        gain = adapter.e_per_dn
        if not photmjsr or not pixar_sr:
            raise ValueError(
                "MJy/sr injection needs PHOTMJSR and PIXAR_SR in the SCI "
                f"header (got {photmjsr!r}, {pixar_sr!r})"
            )
        if not gain:
            raise ValueError(
                f"MJy/sr injection needs adapter e_per_dn to size the "
                f"Poisson draw (instrument {adapter.key!r})"
            )
        if not np.isfinite(exptime) or exptime <= 0.0:
            raise ValueError(
                f"cannot inject into MJy/sr without a positive exposure time: {exptime}"
            )
        electrons_per_jy = gain * exptime / (
            float(pixar_sr) * 1.0e6 * float(photmjsr)
        )
        sci_per_electron = float(photmjsr) / (gain * exptime)
        return (
            electrons_per_jy,
            sci_per_electron,
            f"Jy -> MJy/sr via PIXAR_SR; Poisson width via nominal "
            f"e_per_dn={gain} (approximate; mean is gain-free)",
        )
    raise ValueError(
        f"unrecognised SCI BUNIT {bunit!r} for injection — expected "
        "ELECTRONS[/S] (HST) or MJY/SR (JWST) calibrated products"
    )


def _frame_kernel(
    hdul, extver: int, spec: TargetSpec, adapter: InstrumentAdapter
) -> Tuple[np.ndarray, str]:
    """(unit-sum kernel, provenance note) for one chip."""
    from astropy.io import fits

    from ..psf import epsf as epsf_mod

    if spec.inject_psf:
        kernel = fits.getdata(spec.inject_psf).astype(np.float64)
        kernel = epsf_mod.normalise_kernel(kernel, kernel.shape)
        return kernel, f"inject_psf {Path(spec.inject_psf).name}"
    from ..psf import frame_epsf as frame_epsf_mod

    _psf, psf_full, diag = frame_epsf_mod.build_frame_epsf(hdul, extver, spec, adapter)
    if psf_full is None:
        raise epsf_mod.InsufficientStarsError(
            f"frame ePSF not viable for injection ({diag.get('tier1', {}).get('reason')}); "
            "provide TargetSpec.inject_psf explicitly"
        )
    return epsf_mod.normalise_kernel(psf_full, psf_full.shape), "tier-1 frame ePSF"


def inject_into_exposures(
    exposures: Sequence[Path],
    spec: TargetSpec,
    adapter: InstrumentAdapter,
    work_dir: Path,
) -> Tuple[List[Path], Dict]:
    """
    Inject the input image into work-dir copies of every exposure.

    Returns the new exposure paths and the provenance fragment for
    ``reduction.json``'s ``inject`` block.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    input_cps, in_wcs = load_input_image(spec)
    input_flux = float(input_cps.sum())
    injected_dir = Path(work_dir) / "injected"
    injected_dir.mkdir(parents=True, exist_ok=True)

    new_paths: List[Path] = []
    frame_records: List[Dict] = []
    psf_note: Optional[str] = None
    units_note: Optional[str] = None
    for path in exposures:
        path = Path(path)
        out_path = injected_dir / path.name
        shutil.copy2(path, out_path)
        seed_offset = zlib.crc32(path.name.encode())
        rng = np.random.default_rng((spec.inject_seed, seed_offset))
        chip_records: List[Dict] = []
        with fits.open(out_path, mode="update") as hdul:
            from ..package.frames import _exposure_time

            exptime = _exposure_time(hdul[0].header)
            extvers = [
                hdu.ver for hdu in hdul if hdu.name == "SCI"
            ]
            if not extvers:
                raise ValueError(f"exposure carries no SCI extensions: {path.name}")
            for extver in extvers:
                sci_hdu = hdul["SCI", extver]
                chip_wcs = WCS(sci_hdu.header, fobj=hdul, naxis=2)
                model_cps = render_to_chip(
                    input_cps, in_wcs, chip_wcs, sci_hdu.data.shape
                )
                overlap = float(model_cps.sum())
                if overlap <= MIN_OVERLAP_FRACTION * max(input_flux, 1.0):
                    continue
                kernel, psf_note = _frame_kernel(hdul, extver, spec, adapter)
                model_input = convolve_with_psf(model_cps, kernel)
                electrons_per_input, sci_per_electron, units_note = _chip_units(
                    str(sci_hdu.header.get("BUNIT", "")),
                    sci_hdu.header,
                    exptime,
                    adapter,
                )
                # Convolution ringing can leave tiny negatives; they are
                # not source flux and cannot seed a Poisson draw.
                model_e = np.clip(model_input, 0.0, None) * electrons_per_input
                counts = rng.poisson(model_e).astype(np.float64)
                sci_hdu.data = sci_hdu.data + counts * sci_per_electron
                try:
                    err_hdu = hdul["ERR", extver]
                except KeyError:
                    raise ValueError(
                        f"exposure has no ERR extension for chip {extver}: "
                        f"{path.name} — injection must propagate its variance"
                    )
                err_hdu.data = np.sqrt(
                    err_hdu.data.astype(np.float64) ** 2
                    + model_e * sci_per_electron**2
                )
                chip_records.append(
                    {"extver": int(extver), "injected_e": float(counts.sum())}
                )
            hdul[0].header["INJECTED"] = (True, "synthetic source injected (simulate.md)")
            hdul[0].header["INJIMG"] = (
                Path(spec.inject_image).name,
                "injected input image",
            )
            hdul[0].header["INJSEED"] = (spec.inject_seed, "injection Poisson seed")
        new_paths.append(out_path)
        frame_records.append(
            {
                "exposure": path.name,
                "seed_offset": int(seed_offset),
                "chips": chip_records,
            }
        )

    fragment = {
        "input_image": Path(spec.inject_image).name,
        "input_units": adapter.inject_units,
        "units_note": units_note,
        "input_pixel_scale": spec.inject_pixel_scale,
        "input_flux_cps": input_flux,
        "position": list(spec.inject_position or (spec.ra, spec.dec)),
        "psf_source": psf_note or "no frame overlapped the injection footprint",
        "seed": spec.inject_seed,
        "total_injected_e": float(
            sum(c["injected_e"] for f in frame_records for c in f["chips"])
        ),
        "frames": frame_records,
    }
    return new_paths, fragment
