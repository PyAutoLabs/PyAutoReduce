"""
Keck injection (docs/design/simulate.md, phase 2b — the nirc2_native path).

Placement never touches the raw header WCS (arcsecond-grade — tens of
native pixels): it rides the same measured-offsets arithmetic the combine
and the frame-products packager use. A pre-pass measures
`offsets_to_reference` on the prepared frames and reproduces the combine's
deterministic mosaic geometry; the target sits at the mosaic centre (the
keck path's existing WCS convention), `inject_position` is honoured as an
offset from the target, and each frame's injection centre comes from the
frame-products fixed-point pixmap inversion. Because the injected content
is placed consistently with the measured offsets, the combine's
re-measured registration is unchanged in expectation — the recovery spike
checks this empirically.

Prepared frames are single-HDU electron images (running-sky-subtracted,
NaN bad pixels, ITIME/COADDS headers, no ERR extension): injection adds
Poisson counts to finite pixels only, and the injected source's noise
reaches the packaged noise map through the mosaic counts themselves —
the keck noise stage constructs, never reads, per-frame errors.

Phase 2b requires ``TargetSpec.inject_psf`` (the epoch-matched tier-A
candidate as an automatic source needs candidates built before the
combine — a recorded follow-up on issue #54).
"""

import shutil
import zlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..instruments import InstrumentAdapter
from ..target import TargetSpec
from . import imaging as imaging_mod


def _affine_pixmap(
    input_shape: Tuple[int, int],
    centre_yx: Tuple[float, float],
    scale_ratio_in_to_native: float,
) -> np.ndarray:
    """Input grid -> frame pixels: pure scale + shift around the centre."""
    ny, nx = input_shape
    yy, xx = np.mgrid[0.0:ny, 0.0:nx]
    cy_in, cx_in = (ny - 1) / 2.0, (nx - 1) / 2.0
    pixmap = np.empty((ny, nx, 2), dtype=np.float64)
    pixmap[..., 0] = centre_yx[1] + (xx - cx_in) * scale_ratio_in_to_native
    pixmap[..., 1] = centre_yx[0] + (yy - cy_in) * scale_ratio_in_to_native
    return pixmap


def inject_into_prepared(
    prepared_paths: Sequence[Path],
    spec: TargetSpec,
    adapter: InstrumentAdapter,
    work_dir: Path,
    distortion: np.ndarray,
) -> Tuple[List[Path], Dict]:
    """
    Inject into work-dir copies of the prepared frames; returns the new
    paths and the provenance fragment.
    """
    from astropy.io import fits

    from ..drizzle.nirc2_combine import mosaic_geometry
    from ..package.keck_frames import _target_position_in_frame
    from ..psf import epsf as epsf_mod
    from .imaging import convolve_with_psf, render_via_pixmap

    if not spec.inject_psf:
        raise ValueError(
            "keck injection requires TargetSpec.inject_psf — the AO PSF "
            "varies per epoch and no automatic per-frame source exists "
            "before the combine (issue #54 follow-up)"
        )
    kernel = fits.getdata(spec.inject_psf).astype(np.float64)
    kernel = epsf_mod.normalise_kernel(kernel, kernel.shape)

    input_flux, in_wcs = imaging_mod.load_input_image(spec)
    del in_wcs  # keck placement is offset-based; the WCS is unused

    frames = [
        fits.getdata(p).astype(np.float64) for p in prepared_paths
    ]
    from ..align.registration import offsets_to_reference

    offsets = offsets_to_reference(
        [np.nan_to_num(f, nan=0.0) for f in frames]
    )
    scale_ratio = adapter.scale_ratio(spec.final_scale)  # final / native
    mosaic_scale_ratio = 1.0 / scale_ratio  # native -> final, as combine uses
    origin, out_shape = mosaic_geometry(
        frames[0].shape, offsets, mosaic_scale_ratio, distortion
    )
    # The keck mosaic WCS convention: the target sits at the mosaic centre.
    target_mosaic = ((out_shape[0] - 1) / 2.0, (out_shape[1] - 1) / 2.0)
    if spec.inject_position is not None:
        d_ra, d_dec = (
            spec.inject_position[0] - spec.ra,
            spec.inject_position[1] - spec.dec,
        )
        # Mosaic axes follow the detector frame; the recorded convention
        # (nirc2_combine output WCS) is x ~ -RA, y ~ +Dec at final_scale.
        cos_dec = np.cos(np.radians(spec.dec))
        target_mosaic = (
            target_mosaic[0] + (d_dec * 3600.0) / spec.final_scale,
            target_mosaic[1] - (d_ra * 3600.0 * cos_dec) / spec.final_scale,
        )

    injected_dir = Path(work_dir) / "injected"
    injected_dir.mkdir(parents=True, exist_ok=True)
    input_scale_ratio = spec.inject_pixel_scale / adapter.native_scale

    new_paths: List[Path] = []
    frame_records: List[Dict] = []
    for path, frame, offset in zip(prepared_paths, frames, offsets):
        path = Path(path)
        out_path = injected_dir / path.name
        shutil.copy2(path, out_path)
        rng = np.random.default_rng(
            (spec.inject_seed, zlib.crc32(path.name.encode()))
        )
        with fits.open(out_path, mode="update") as hdul:
            header = hdul[0].header
            t_frame = float(header["ITIME"]) * int(header["COADDS"])
            if t_frame <= 0.0:
                raise ValueError(
                    f"non-positive frame exposure time in {path.name}: {t_frame}"
                )
            centre = _target_position_in_frame(
                target_mosaic,
                offset,
                origin,
                mosaic_scale_ratio,
                distortion,
                frame.shape,
            )
            pixmap = _affine_pixmap(input_flux.shape, centre, input_scale_ratio)
            rendered = render_via_pixmap(
                input_flux,
                pixmap,
                frame.shape,
                area_ratio=1.0 / input_scale_ratio**2,
            )
            rendered = convolve_with_psf(rendered, kernel)
            model_e = np.clip(rendered, 0.0, None) * t_frame
            counts = rng.poisson(model_e).astype(np.float64)
            # NaN bad pixels stay NaN — a real source's photons land there
            # too, and are lost the same way.
            finite = np.isfinite(hdul[0].data)
            data = hdul[0].data.astype(np.float64)
            data[finite] += counts[finite]
            hdul[0].data = data.astype(np.float32)
            header["INJECTED"] = (True, "synthetic source injected (simulate.md)")
            header["INJIMG"] = (Path(spec.inject_image).name, "injected input image")
            header["INJSEED"] = (spec.inject_seed, "injection Poisson seed")
        new_paths.append(out_path)
        frame_records.append(
            {
                "exposure": path.name,
                "seed_offset": int(zlib.crc32(path.name.encode())),
                "centre_yx": [float(centre[0]), float(centre[1])],
                "injected_e": float(counts[finite].sum()),
            }
        )

    fragment = {
        "input_image": Path(spec.inject_image).name,
        "input_units": adapter.inject_units,
        "units_note": "e-/s x ITIME x COADDS (prepared electron frames)",
        "input_pixel_scale": spec.inject_pixel_scale,
        "input_flux_cps": float(input_flux.sum()),
        "placement": (
            "measured-offsets arithmetic (offsets_to_reference pre-pass, "
            "target-at-mosaic-centre convention); inject_position honoured "
            "as an offset from the target, not absolute"
        ),
        "position": list(spec.inject_position or (spec.ra, spec.dec)),
        "psf_source": f"inject_psf {Path(spec.inject_psf).name}",
        "seed": spec.inject_seed,
        "total_injected_e": float(sum(f["injected_e"] for f in frame_records)),
        "frames": frame_records,
    }
    return new_paths, fragment
