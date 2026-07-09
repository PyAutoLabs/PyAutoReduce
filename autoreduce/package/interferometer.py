"""
Interferometer packaging (design doc alma.md, package stage): the
`al.Interferometer.from_fits` product triplet — `data.fits`,
`uv_wavelengths.fits`, `noise_map.fits`, each `(Nvis, 2)` float64 — plus
diagnostic sidecars. The autolens contract is emitted, never imported
(boundary rule); the validation prototype does the `from_fits` round-trip.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PRODUCTS = ("data.fits", "uv_wavelengths.fits", "noise_map.fits")


def write_products(
    out_dir: Path,
    visibilities: np.ndarray,
    uv_wavelengths: np.ndarray,
    noise_map: np.ndarray,
    sidecars: Optional[Dict[str, np.ndarray]] = None,
) -> List[str]:
    """Write the triplet (+ named sidecars); returns the product filenames."""
    from astropy.io import fits

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "data.fits": visibilities,
        "uv_wavelengths.fits": uv_wavelengths,
        "noise_map.fits": noise_map,
    }
    n = np.asarray(visibilities).shape[0]
    for name, array in arrays.items():
        array = np.asarray(array, dtype=np.float64)
        if array.shape != (n, 2):
            raise ValueError(
                f"{name} has shape {array.shape}, expected ({n}, 2) — the "
                f"al.Interferometer contract"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(noise_map <= 0.0):
        raise ValueError("noise_map must be positive everywhere")

    products = []
    for name, array in arrays.items():
        fits.PrimaryHDU(np.asarray(array, dtype=np.float64)).writeto(
            out_dir / name, overwrite=True
        )
        products.append(name)
    for name, array in (sidecars or {}).items():
        filename = f"{name}.fits"
        fits.PrimaryHDU(np.asarray(array)).writeto(
            out_dir / filename, overwrite=True
        )
        products.append(filename)
    return products
