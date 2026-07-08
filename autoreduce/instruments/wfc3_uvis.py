"""
WFC3/UVIS — adapter #2 (roadmap phase 2).

The ACS-like path: CTE-corrected ``_flc`` exposures, ``iref`` references,
cps/IVM/north-up outputs. Native plate scale 0.0396″/pix — the published
lensing anchor is the Bayer et al. (arXiv:1803.05952) F390W reduction of
SDSS J0252+0039 at exactly this output scale with pixfrac 1.0.
"""

from .adapter import InstrumentAdapter, register

WFC3_UVIS = register(
    InstrumentAdapter(
        key="wfc3_uvis",
        mast_instrument_name="WFC3/UVIS",
        native_scale=0.0396,
        calibrated_suffix="FLC",
        reference_env_key="iref",
        crds_reference_subpath="references/hst/wfc3",
        supports_cte_correction=True,
        default_drizzle_kwargs={
            "skymethod": "globalmin+match",
            "final_wht_type": "IVM",
            "final_units": "cps",
            "final_rot": 0.0,
        },
        saturation_dn=63_000.0,
        recommended_final_scale=0.0396,
    )
)
