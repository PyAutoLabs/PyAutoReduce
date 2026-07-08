"""
MAST acquisition (design doc stage 1).

Query hygiene (spike finding): plain coordinate queries also match HAP
skycell products, whose member lists re-reference the same exposures many
times over and pull in neighbouring pointings. We therefore keep only
*direct* calibration-level observations (numeric proposal IDs, obs_id not a
``hst_skycell`` product) and optionally filter by proposal, then download the
adapter's calibrated exposure products.
"""

from pathlib import Path
from typing import List, Optional, Sequence

from ..instruments import InstrumentAdapter


def is_direct_observation(obs_id: str, proposal_id: str) -> bool:
    """True for a direct program observation, False for HAP skycell products."""
    if str(obs_id).startswith("hst_skycell"):
        return False
    proposal = str(proposal_id).strip()
    return proposal.isdigit()


def select_observations(
    obs_table,
    proposal_ids: Optional[Sequence[str]] = None,
):
    """Filter a MAST observation table to direct program observations."""
    keep = []
    for row in obs_table:
        if not is_direct_observation(row["obs_id"], row["proposal_id"]):
            continue
        if proposal_ids is not None and str(row["proposal_id"]) not in set(
            str(p) for p in proposal_ids
        ):
            continue
        keep.append(row)
    return keep


def query_exposures(
    ra: float,
    dec: float,
    adapter: InstrumentAdapter,
    filter_name: str,
    radius: str = "0.5 arcmin",
    proposal_ids: Optional[Sequence[str]] = None,
):
    """Query MAST for direct observations of the target. Network."""
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Observations

    coord = SkyCoord(ra, dec, unit="deg")
    obs = Observations.query_criteria(
        coordinates=coord,
        radius=radius,
        obs_collection="HST",
        instrument_name=adapter.mast_instrument_name,
        filters=filter_name,
        dataproduct_type="image",
    )
    selected = select_observations(obs, proposal_ids=proposal_ids)
    if not selected:
        raise LookupError(
            f"no direct {adapter.mast_instrument_name} {filter_name} observations "
            f"at ({ra}, {dec}) within {radius}"
            + (f" for proposals {list(proposal_ids)}" if proposal_ids else "")
        )
    return selected


def download_exposures(
    observations,
    adapter: InstrumentAdapter,
    download_dir: Path,
) -> List[Path]:
    """Download the calibrated exposure products for the observations. Network."""
    from astropy.table import vstack
    from astroquery.mast import Observations

    products = vstack([Observations.get_product_list(row) for row in observations])
    calibrated = Observations.filter_products(
        products,
        productSubGroupDescription=[adapter.calibrated_suffix],
        mrp_only=False,
    )
    if len(calibrated) == 0:
        raise LookupError(
            f"observations carry no {adapter.calibrated_suffix} products"
        )
    Observations.download_products(calibrated, download_dir=str(download_dir))
    suffix = f"_{adapter.calibrated_suffix.lower()}.fits"
    paths = sorted(set(Path(download_dir).rglob(f"*{suffix}")))
    if not paths:
        raise FileNotFoundError(
            f"download reported success but no *{suffix} files under {download_dir}"
        )
    return list(paths)
