"""
Exposure combination via AstroDrizzle (design doc stage 3).

Defaults-first: the adapter supplies the STScI-recommended keywords; the
lensing deviations (final scale, orientation, units, weight type) and the
user-facing ``pixfrac``/``kernel`` dials come from the `TargetSpec`. The
single-exposure branch (SLACS-V caveat) drizzles the lone frame without
CR rejection — cosmic rays are flagged from the DQ array downstream instead
of median-combining, and the provenance records the branch taken.
"""

import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..instruments import InstrumentAdapter
from ..target import TargetSpec

# AstroDrizzle's cosmic-ray DQ bit. The cr_method="deepcr" route writes its
# per-frame masks into this same bit, so every downstream consumer of CR
# flags (final drizzle bit masking, frame products) reads one convention.
CR_DQ_BIT = 4096


def drizzle_kwargs_for(spec: TargetSpec, adapter: InstrumentAdapter, n_exposures: int) -> Dict:
    """Assemble the AstroDrizzle keyword set; pure function, unit-testable."""
    if n_exposures < 1:
        raise ValueError("need at least one exposure")
    multi = n_exposures > 1
    kwargs = dict(adapter.default_drizzle_kwargs)
    kwargs.update(
        preserve=False,
        build=False,
        clean=True,
        final_scale=spec.final_scale,
        final_pixfrac=spec.final_pixfrac,
        final_kernel=spec.final_kernel,
        # CR rejection needs >= 2 exposures; the single-exposure branch
        # (SLACS-V caveat) skips median/blot/driz_cr.
        driz_cr=multi,
        median=multi,
        blot=multi,
    )
    if multi and spec.cr_method == "deepcr":
        # Per-frame route (issue #61): CR masks are already in the DQ arrays
        # (apply_per_frame_cr_masks), so the stack rejection is off and the
        # combine is a plain weighted mean. resetbits=0 is load-bearing —
        # AstroDrizzle's default resetbits (4096) clears exactly the DQ bit
        # the masks were written into, silently producing an unmasked mosaic.
        kwargs.update(driz_cr=False, median=False, blot=False, resetbits=0)
    return kwargs


def star_pass_kwargs_for(
    spec: TargetSpec, adapter: InstrumentAdapter, n_exposures: int
) -> Dict:
    """
    Keyword set for the dedicated PSF-star drizzle pass
    (``psf_star_pass="no_cr"``, issue #62): the science kwargs with every
    stack CR-rejection step off and previously written CR DQ flags treated
    as good (``final_bits`` gains the CR bit). ``resetbits=0`` so the
    science pass's DQ flags survive in the input exposures — frame products
    and re-runs still see them. Pure function, unit-testable.
    """
    kwargs = drizzle_kwargs_for(spec, adapter, n_exposures)
    kwargs.update(
        driz_cr=False,
        median=False,
        blot=False,
        resetbits=0,
        final_bits=int(kwargs.get("final_bits", 0)) | CR_DQ_BIT,
    )
    return kwargs


def dq_with_cr_flags(dq: np.ndarray, cr_mask: np.ndarray, cr_bit: int = CR_DQ_BIT) -> np.ndarray:
    """
    One chip's DQ array with the CR bit rewritten from ``cr_mask``: the bit
    is cleared everywhere first (so re-runs are idempotent, mirroring
    AstroDrizzle's own resetbits behaviour) then set where the mask flags a
    cosmic ray. Pure function; preserves the DQ dtype.
    """
    dq = np.asarray(dq)
    cr_mask = np.asarray(cr_mask, dtype=bool)
    if dq.shape != cr_mask.shape:
        raise ValueError(f"DQ/mask shape mismatch: {dq.shape} vs {cr_mask.shape}")
    cleared = (dq & ~cr_bit).astype(dq.dtype)
    return np.where(cr_mask, cleared | cr_bit, cleared).astype(dq.dtype)


def apply_per_frame_cr_masks(exposures: List[Path], adapter: InstrumentAdapter) -> Dict:
    """
    The ``cr_method="deepcr"`` masking step (issue #61): per-frame deepCR
    cosmic-ray masks written into each exposure's DQ arrays as the
    AstroDrizzle CR bit, replacing driz_cr's blotted-median stack rejection
    (which reads systematically low on steep gradients and flags genuine
    core flux). Mutates the exposures' DQ in place — exactly as driz_cr
    itself does — idempotently (the CR bit is cleared before rewriting).
    Returns the provenance fragment.
    """
    from astropy.io import fits

    from ..package import cosmic_rays as cr_mod

    masker = cr_mod.masker_for(adapter.key)
    n_cr_pixels = {}
    for path in exposures:
        with fits.open(path, mode="update") as hdul:
            total = 0
            for hdu in hdul:
                if hdu.name != "SCI":
                    continue
                mask = masker(hdu.data)
                dq_hdu = hdul["DQ", hdu.ver]
                dq_hdu.data = dq_with_cr_flags(dq_hdu.data, mask)
                total += int(mask.sum())
            n_cr_pixels[Path(path).name] = total
    return {
        **cr_mod.cr_method_record(adapter.key),
        "cr_bit": CR_DQ_BIT,
        "n_cr_pixels": n_cr_pixels,
    }


def combine(
    exposures: List[Path],
    spec: TargetSpec,
    adapter: InstrumentAdapter,
    output_dir: Path,
) -> Tuple[Path, Path, Dict]:
    """
    Combine exposures via the adapter's backend; return
    (sci_path, wht_path, provenance_fragment).

    Requires the CRDS environment configured (acquire.crds) beforehand.
    """
    if adapter.combine_backend == "jwst_image3":
        from . import jwst_combine

        return jwst_combine.combine(exposures, spec, adapter, output_dir)
    if adapter.combine_backend == "nirc2_native":
        from . import nirc2_combine

        return nirc2_combine.combine(exposures, spec, adapter, output_dir)
    if adapter.combine_backend != "astrodrizzle":
        raise ValueError(
            f"unknown combine backend {adapter.combine_backend!r} "
            f"for instrument {adapter.key}"
        )

    from astropy.io import fits
    from drizzlepac import astrodrizzle

    from ._common import chdir_scratch, combine_provenance

    # Drizzlepac lowercases output filenames internally, which breaks absolute
    # paths containing capitals on case-sensitive filesystems — so run inside
    # the scratch dir with a relative, already-lowercase output root. This
    # also keeps AstroDrizzle's cwd scratch files contained.
    output_name = f"{spec.name}_{spec.filter_name}".lower()
    per_frame_cr = None
    if len(exposures) > 1 and spec.cr_method == "deepcr":
        per_frame_cr = apply_per_frame_cr_masks(exposures, adapter)
    kwargs = drizzle_kwargs_for(spec, adapter, len(exposures))
    with chdir_scratch(output_dir) as output_dir:
        astrodrizzle.AstroDrizzle(
            input=[str(p) for p in exposures],
            output=output_name,
            **kwargs,
        )
    output_root = str(output_dir / output_name)

    def _one(suffix: str) -> Path:
        hits = sorted(glob.glob(f"{output_root}*{suffix}"))
        if len(hits) != 1:
            raise FileNotFoundError(
                f"expected exactly one {suffix} for {output_root}, got {hits}"
            )
        return Path(hits[0])

    sci = _one("_sci.fits")
    wht = _one("_wht.fits")

    tail = {"cr_method": spec.cr_method}
    if per_frame_cr is not None:
        tail["per_frame_cr"] = per_frame_cr
    provenance = combine_provenance(
        spec,
        adapter,
        exposures,
        fits.getdata(wht),
        kwargs_key="drizzle_kwargs",
        kwargs={k: kwargs[k] for k in sorted(kwargs)},
        tail=tail,
    )
    return sci, wht, provenance


def combine_star_pass(
    exposures: List[Path],
    spec: TargetSpec,
    adapter: InstrumentAdapter,
    output_dir: Path,
) -> Tuple[Path, Dict]:
    """
    Drizzle the dedicated CR-flag-ignoring PSF-star mosaic
    (``psf_star_pass="no_cr"``, issue #62) onto the same grid/geometry as
    the science combine; returns (sci_path, kwargs). AstroDrizzle backend
    only — the caller gates on ``adapter.combine_backend``.
    """
    if adapter.combine_backend != "astrodrizzle":
        raise ValueError(
            "combine_star_pass supports the astrodrizzle backend only "
            f"(instrument {adapter.key!r})"
        )

    from drizzlepac import astrodrizzle

    from ._common import chdir_scratch

    output_name = f"{spec.name}_{spec.filter_name}_starpass".lower()
    kwargs = star_pass_kwargs_for(spec, adapter, len(exposures))
    with chdir_scratch(output_dir) as output_dir:
        astrodrizzle.AstroDrizzle(
            input=[str(p) for p in exposures],
            output=output_name,
            **kwargs,
        )
    hits = sorted(glob.glob(f"{output_dir / output_name}*_sci.fits"))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one star-pass _sci.fits for "
            f"{output_dir / output_name}, got {hits}"
        )
    return Path(hits[0]), kwargs
