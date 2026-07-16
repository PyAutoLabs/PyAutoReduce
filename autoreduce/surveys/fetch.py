"""
Per-service cutout fetchers. URL construction is pure (unit-tested);
the network step is a thin `astropy` download, loud on empty coverage.

All service endpoints were verified live against slacs0008-0004
(2026-07-16): Legacy `fits-cutout` returns a (nbands, ny, nx) image cube
with an inverse-variance cube as a second HDU under ``&invvar``; the PS1
`ps1filenames.py` table + `fitscut.cgi` two-step serves stack cutouts;
SDSS frames come through `astroquery.sdss`.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

LEGACY_LAYER = "ls-dr10"
LEGACY_CUTOUT_URL = "https://www.legacysurvey.org/viewer/fits-cutout"
PS1_FILENAMES_URL = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
PS1_FITSCUT_URL = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"


def legacy_cutout_url(
    ra: float, dec: float, size: int, bands: Tuple[str, ...], invvar: bool
) -> str:
    """One request serves all bands as a cube (+ invvar HDU when asked)."""
    url = (
        f"{LEGACY_CUTOUT_URL}?ra={ra}&dec={dec}&layer={LEGACY_LAYER}"
        f"&pixscale=0.262&bands={''.join(bands)}&size={size}"
    )
    return url + "&invvar" if invvar else url


def ps1_filenames_url(ra: float, dec: float, bands: Tuple[str, ...]) -> str:
    return f"{PS1_FILENAMES_URL}?ra={ra}&dec={dec}&filters={''.join(bands)}"


def ps1_fitscut_url(filename: str, ra: float, dec: float, size: int) -> str:
    return (
        f"{PS1_FITSCUT_URL}?red={filename}&ra={ra}&dec={dec}"
        f"&size={size}&format=fits"
    )


def _download(url: str) -> Path:
    from astropy.utils.data import download_file

    return Path(download_file(url, cache=False, show_progress=False))


def fetch_legacy(
    ra: float, dec: float, size: int, bands: Tuple[str, ...]
) -> Dict[str, Dict]:
    """{band: {data, ivar, header}} from one Legacy cube request."""
    from astropy.io import fits

    path = _download(legacy_cutout_url(ra, dec, size, bands, invvar=True))
    with fits.open(path) as hdul:
        cube = hdul[0].data.astype(np.float64)
        ivar = hdul[1].data.astype(np.float64)
        header = hdul[0].header.copy()
    served = tuple(str(header.get("BANDS", "")).replace(" ", ""))
    if cube.ndim != 3 or len(served) != cube.shape[0]:
        raise ValueError(
            f"unexpected Legacy cutout shape {cube.shape} for bands {served!r}"
        )
    out = {}
    for i, band in enumerate(served):
        if band not in bands:
            continue
        if not np.any(cube[i]):
            # An all-zero plane is the service's "no coverage" for a band.
            continue
        out[band] = {"data": cube[i], "ivar": ivar[i], "header": header}
    if not out:
        raise ValueError(
            f"Legacy Surveys has no {''.join(bands)} coverage at "
            f"({ra:.5f}, {dec:.5f})"
        )
    return out


def fetch_ps1(
    ra: float, dec: float, size: int, bands: Tuple[str, ...]
) -> Dict[str, Dict]:
    """{band: {data, header}} via the filenames-table + fitscut two-step."""
    from astropy.io import fits

    table_path = _download(ps1_filenames_url(ra, dec, bands))
    rows = [
        line.split()
        for line in Path(table_path).read_text().splitlines()[1:]
        if line.strip()
    ]
    by_band = {row[4]: row[7] for row in rows}
    out = {}
    for band in bands:
        if band not in by_band:
            continue
        path = _download(ps1_fitscut_url(by_band[band], ra, dec, size))
        with fits.open(path) as hdul:
            out[band] = {
                "data": hdul[0].data.astype(np.float64),
                "header": hdul[0].header.copy(),
            }
    if not out:
        raise ValueError(
            f"Pan-STARRS has no {''.join(bands)} coverage at ({ra:.5f}, {dec:.5f})"
        )
    return out


def fetch_sdss(
    ra: float, dec: float, size: int, bands: Tuple[str, ...]
) -> Dict[str, Dict]:
    """{band: {data, header}} — astroquery frame, cut to size around the target."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    from astroquery.sdss import SDSS as sdss_service

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    out = {}
    for band in bands:
        frames = sdss_service.get_images(coordinates=coord, band=band)
        if not frames:
            continue
        hdu = frames[0][0]
        wcs = WCS(hdu.header)
        cut = Cutout2D(hdu.data.astype(np.float64), coord, (size, size), wcs=wcs)
        header = hdu.header.copy()
        header.update(cut.wcs.to_header())
        out[band] = {"data": cut.data, "header": header}
    if not out:
        raise ValueError(
            f"SDSS has no {''.join(bands)} coverage at ({ra:.5f}, {dec:.5f})"
        )
    return out


FETCHERS = {"legacy": fetch_legacy, "ps1": fetch_ps1, "sdss": fetch_sdss}
