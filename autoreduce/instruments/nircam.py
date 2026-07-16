"""
JWST/NIRCam — adapters #4 and #5 (roadmap phase 3), the first non-HST path.

Both channels reduce from level-2 ``_cal`` products (calwebb_image2 output,
MJy/sr) and combine through the ``jwst`` pipeline's calwebb_image3 — the
drizzle analogue (tweakreg / skymatch / outlier_detection / resample). CRDS
routes through the JWST server; references sync under ``references/jwst``.

Output-scale recommendations follow the COSMOS-Web mosaic convention the
parity anchor uses: SW at 0.03″/pix, LW at 0.06″/pix. ``saturation_dn`` is
the detector full well in electrons, as elsewhere; NIRCam ~ 105 ke- (2RG
arrays), conservative here.
"""

from .adapter import InstrumentAdapter, register

_COMMON = dict(
    calibrated_suffix="CAL",
    reference_env_key="CRDS_PATH",  # jwst pipeline reads CRDS_PATH directly
    crds_reference_subpath="references/jwst",
    supports_cte_correction=False,
    observatory="jwst",
    crds_server_url="https://jwst-crds.stsci.edu",
    combine_backend="jwst_image3",
    mast_obs_collection="JWST",
    default_drizzle_kwargs={
        # calwebb_image3 resample step keywords (dial-mapped in jwst_combine)
        "pixfrac": 1.0,
        "kernel": "square",
        "weight_type": "ivm",
    },
    saturation_dn=100_000.0,
    # Injection (simulate.md phase 2a): input in Jy/pixel; nominal NIRCam
    # gain sizes the Poisson draw only (disclosed in provenance).
    inject_units="Jy",
    e_per_dn=2.0,
)

NIRCAM_SW = register(
    InstrumentAdapter(
        key="nircam_sw",
        mast_instrument_name="NIRCAM/IMAGE",
        native_scale=0.031,
        recommended_final_scale=0.03,
        **_COMMON,
    )
)

NIRCAM_LW = register(
    InstrumentAdapter(
        key="nircam_lw",
        mast_instrument_name="NIRCAM/IMAGE",
        native_scale=0.063,
        recommended_final_scale=0.06,
        **_COMMON,
    )
)

# NIRCam filter -> channel routing (wavelength < 2.4 micron = SW).
SW_FILTERS = {"F070W", "F090W", "F115W", "F140M", "F150W", "F162M", "F164N",
              "F150W2", "F182M", "F187N", "F200W", "F210M", "F212N"}
LW_FILTERS = {"F250M", "F277W", "F300M", "F322W2", "F323N", "F335M", "F356W",
              "F360M", "F405N", "F410M", "F430M", "F444W", "F460M", "F466N",
              "F470N", "F480M"}


def nircam_adapter_for_filter(filter_name: str) -> InstrumentAdapter:
    """Route a NIRCam filter to its channel adapter; loud on unknown filters."""
    name = filter_name.upper()
    if name in SW_FILTERS:
        return NIRCAM_SW
    if name in LW_FILTERS:
        return NIRCAM_LW
    raise KeyError(f"unknown NIRCam filter {filter_name!r} — not in SW/LW tables")
