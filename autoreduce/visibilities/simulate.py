"""
The simobserve acquire-alternative (docs/design/simulate.md, phase 3).

Fully-synthetic ALMA: the one instrument where raw-data simulation is
cheap and observatory-supported. The inject input (Jy per pixel — the
same flux contract as JWST injection) becomes a 4-axis sky model; CASA
``simobserve`` turns it into a MeasurementSet with thermal + atmospheric
noise for the chosen array configuration; the existing
``split -> extract -> assemble -> package`` stages then run unchanged.

Mode 2 — uv-plane injection into a *real* MS (the true Balrog analogue,
real calibration systematics for free) — is deferred: it needs
FT-at-uv-points and phase-centre machinery this mode doesn't, and this
mode fully answers the original ask. Recorded in simulate.md.

Everything numeric here is pure and unit-tested; only ``simulate_ms``
touches casatasks (loud ImportError guidance when absent, matching the
extract stage's contract).
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from ..target import TargetSpec


def skymodel_fits(
    input_image: np.ndarray,
    pixel_scale: float,
    ra: float,
    dec: float,
    freq_ghz: float,
    out_path: Path,
) -> Path:
    """
    The 4-axis (RA---SIN / DEC--SIN / STOKES / FREQ) Jy/pixel sky model
    simobserve consumes. Total flux is the array sum, unchanged.
    """
    from astropy.io import fits

    if pixel_scale <= 0.0:
        raise ValueError(f"sky-model pixel scale must be positive: {pixel_scale}")
    ny, nx = input_image.shape
    data = input_image.astype(np.float32)[np.newaxis, np.newaxis, :, :]
    header = fits.Header()
    header["BUNIT"] = "Jy/pixel"
    header["BTYPE"] = "Intensity"
    scale_deg = pixel_scale / 3600.0
    for i, (ctype, crval, cdelt, crpix) in enumerate(
        [
            ("RA---SIN", ra, -scale_deg, (nx + 1) / 2.0),
            ("DEC--SIN", dec, scale_deg, (ny + 1) / 2.0),
            ("STOKES", 1.0, 1.0, 1.0),
            ("FREQ", freq_ghz * 1e9, 2.0e9, 1.0),
        ],
        start=1,
    ):
        header[f"CTYPE{i}"] = ctype
        header[f"CRVAL{i}"] = crval
        header[f"CDELT{i}"] = cdelt
        header[f"CRPIX{i}"] = crpix
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CUNIT4"] = "Hz"
    header["RESTFRQ"] = freq_ghz * 1e9
    header["RADESYS"] = "ICRS"
    fits.PrimaryHDU(data, header=header).writeto(out_path, overwrite=True)
    return Path(out_path)


def simobserve_kwargs(spec: TargetSpec, skymodel: Path) -> Dict:
    """The headless simobserve call, as a pure inspectable dict."""
    thermal = spec.alma_sim_pwv_mm > 0.0
    kwargs = {
        "project": f"{spec.name}_sim",
        "skymodel": str(skymodel),
        "setpointings": True,
        "obsmode": "int",
        "antennalist": spec.alma_sim_antennalist,
        "integration": f"{spec.alma_sim_integration_s:g}s",
        "totaltime": f"{spec.alma_sim_totaltime_s:g}s",
        "thermalnoise": "tsys-atm" if thermal else "",
        "graphics": "none",
        "overwrite": True,
    }
    if thermal:
        kwargs["user_pwv"] = spec.alma_sim_pwv_mm
    return kwargs


def simulated_ms_path(project_dir: Path, antennalist: str, noisy: bool) -> Path:
    """simobserve's output naming: <project>/<project>.<cfg>.[noisy.]ms."""
    cfg = Path(antennalist).stem
    suffix = "noisy.ms" if noisy else "ms"
    return Path(project_dir) / f"{Path(project_dir).name}.{cfg}.{suffix}"


def simulate_ms(spec: TargetSpec, work_dir: Path) -> Tuple[Path, Dict]:
    """
    Run simobserve in a work-dir scratch (it writes its project directory
    into the cwd); returns the simulated MS path and provenance.
    """
    try:
        from casatasks import simobserve
    except ImportError as err:
        raise ImportError(
            "the simobserve acquire-alternative needs casatasks "
            "(pip install casatasks casatools — the same modular-CASA "
            "stack the extract stage uses; docs/design/alma.md)"
        ) from err

    from astropy.io import fits

    from ..drizzle._common import chdir_scratch

    input_image = fits.getdata(spec.inject_image).astype(np.float64)
    if np.any(~np.isfinite(input_image)) or np.any(input_image < 0.0):
        raise ValueError(
            f"sky model must be finite and non-negative: {spec.inject_image}"
        )
    ra, dec = spec.inject_position or (spec.ra, spec.dec)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    skymodel = skymodel_fits(
        input_image,
        spec.inject_pixel_scale,
        ra,
        dec,
        spec.alma_sim_freq_ghz,
        work_dir / "skymodel.fits",
    )
    kwargs = simobserve_kwargs(spec, skymodel.resolve())
    with chdir_scratch(work_dir):
        simobserve(**kwargs)
    ms_path = simulated_ms_path(
        work_dir / kwargs["project"],
        spec.alma_sim_antennalist,
        noisy=bool(kwargs["thermalnoise"]),
    )
    if not ms_path.exists():
        raise FileNotFoundError(
            f"simobserve completed but the expected MS is missing: {ms_path}"
        )
    provenance = {
        "source": "simobserve",
        "skymodel": skymodel.name,
        "input_flux_jy": float(input_image.sum()),
        "antennalist": spec.alma_sim_antennalist,
        "totaltime_s": spec.alma_sim_totaltime_s,
        "integration_s": spec.alma_sim_integration_s,
        "freq_ghz": spec.alma_sim_freq_ghz,
        "thermal_noise": (
            f"tsys-atm, pwv={spec.alma_sim_pwv_mm}mm"
            if kwargs["thermalnoise"]
            else "none (alma_sim_pwv_mm=0)"
        ),
    }
    return ms_path, provenance
