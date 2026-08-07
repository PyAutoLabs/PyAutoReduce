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
        # MDRIZTAB wfc3/3562021pi_mdz.fits (issue #65). 528 = 512 + 16 — the
        # blob bit plus hot pixels. Blobs are detector-fixed IR channel
        # features that calwf3 flags but does not remove; rejecting them on
        # snapshot data with tiny dithers punches structured zero-coverage
        # holes in the mosaic (PJ011646, 5 exposures, a 123-px hole at
        # r = 5.3"). The IR rows are the reason this is a table and not a
        # pair: the two bits columns DIFFER at N = 2-3, where the separate
        # (median-building) drizzle still keeps every bit while the final
        # drizzle already drops to 528.
        dq_bits_rows=((1, 65535, 65535), (2, 65535, 528), (4, 528, 528)),
    )
)
