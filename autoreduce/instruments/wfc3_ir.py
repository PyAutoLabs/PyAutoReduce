"""
WFC3/IR — adapter #3 (roadmap phase 2), the genuinely different path.

No CTE correction exists for the IR channel, so the calibrated product is
``_flt`` (already in e-/s from up-the-ramp fitting, which also rejects most
cosmic rays per read — AstroDrizzle's driz_cr then handles the residue when
multiple exposures exist). Native scale ~0.128″/pix under-samples the PSF, so
dithered programs conventionally drizzle to a finer grid; 0.065″/pix is the
adapter's recommendation (half-native, within the 0.06–0.08 range common in
deep-field practice) — `TargetSpec.final_scale` remains the user dial and
star-poor or poorly-dithered data may prefer coarser values.
"""

from .adapter import InstrumentAdapter, register

WFC3_IR = register(
    InstrumentAdapter(
        key="wfc3_ir",
        mast_instrument_name="WFC3/IR",
        native_scale=0.128,
        calibrated_suffix="FLT",
        reference_env_key="iref",
        crds_reference_subpath="references/hst/wfc3",
        supports_cte_correction=False,
        default_drizzle_kwargs={
            "skymethod": "globalmin+match",
            "final_wht_type": "IVM",
            "final_units": "cps",
            "final_rot": 0.0,
        },
        saturation_dn=78_000.0,
        recommended_final_scale=0.065,
    )
)
