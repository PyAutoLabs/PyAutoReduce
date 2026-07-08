"""
ACS/WFC — adapter #1 (design doc phase 1).

Values follow the design doc's deviation table: CTE-corrected ``_flc``
exposures, cps output units, IVM weights, north-up final grid.
"""

from .adapter import InstrumentAdapter, register

ACS_WFC = register(
    InstrumentAdapter(
        key="acs_wfc",
        mast_instrument_name="ACS/WFC",
        native_scale=0.05,
        calibrated_suffix="FLC",
        reference_env_key="jref",
        crds_reference_subpath="references/hst/acs",
        supports_cte_correction=True,
        default_drizzle_kwargs={
            "skymethod": "globalmin+match",
            "final_wht_type": "IVM",
            "final_units": "cps",
            "final_rot": 0.0,
        },
        saturation_dn=80_000.0,
    )
)
