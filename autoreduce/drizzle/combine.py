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

from ..instruments import InstrumentAdapter
from ..target import TargetSpec
from .diagnostics import check_weight_uniformity
from ..noise.rms import casertano_r


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
    return kwargs


def combine(
    exposures: List[Path],
    spec: TargetSpec,
    adapter: InstrumentAdapter,
    output_dir: Path,
) -> Tuple[Path, Path, Dict]:
    """
    Run AstroDrizzle; return (sci_path, wht_path, provenance_fragment).

    Requires the CRDS environment configured (acquire.crds) beforehand.
    """
    from astropy.io import fits
    from drizzlepac import astrodrizzle

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Drizzlepac lowercases output filenames internally, which breaks absolute
    # paths containing capitals on case-sensitive filesystems — so chdir into
    # the work dir and pass a relative, already-lowercase output root. This
    # also keeps AstroDrizzle's cwd scratch files contained.
    output_name = f"{spec.name}_{spec.filter_name}".lower()
    output_root = str(output_dir / output_name)

    kwargs = drizzle_kwargs_for(spec, adapter, len(exposures))
    import os

    cwd = os.getcwd()
    os.chdir(output_dir)
    try:
        astrodrizzle.AstroDrizzle(
            input=[str(p) for p in exposures],
            output=output_name,
            **kwargs,
        )
    finally:
        os.chdir(cwd)

    def _one(suffix: str) -> Path:
        hits = sorted(glob.glob(f"{output_root}*{suffix}"))
        if len(hits) != 1:
            raise FileNotFoundError(
                f"expected exactly one {suffix} for {output_root}, got {hits}"
            )
        return Path(hits[0])

    sci = _one("_sci.fits")
    wht = _one("_wht.fits")

    wht_data = fits.getdata(wht)
    provenance = {
        "n_exposures": len(exposures),
        "exposures": [Path(p).name for p in exposures],
        "single_exposure_branch": len(exposures) == 1,
        "drizzle_kwargs": {k: kwargs[k] for k in sorted(kwargs)},
        "correlated_noise_factor": casertano_r(
            spec.final_pixfrac, adapter.scale_ratio(spec.final_scale)
        ),
        "weight_uniformity": check_weight_uniformity(wht_data),
    }
    return sci, wht, provenance
