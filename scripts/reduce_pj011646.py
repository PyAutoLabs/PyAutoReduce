"""
Validation (issue #25): reduce PJ011646 (PASSAGES J011646.77, program 14653,
WFC3/IR F160W, ~2596s, confirmed via MAST at (19.194875, -24.617194)) through
the *production* pipeline at Aris's exact grid — 0.0642"/pix, 201x201 — then
compare the products against his modeling dataset with sub-pixel registration.

WFC3-IR leg of the reduction-validation series (slacs1430+4105 / issue #17
was ACS). IR deltas per docs/design/wfc3.md: _flt inputs, half-native output
scale, and the few-dither fine-grid rule — pixfrac 1.0 (the J0252 F160W
zero-weight-speckle finding), with the larger correlated-noise factor R
reported per run as always.

Run:  ~/venv/PyAuto/bin/python scripts/reduce_pj011646.py
Network + drizzlepac required; unit tests never import this.
"""

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from autoreduce import TargetSpec, reduce_target  # noqa: E402

# --- WORKAROUND (issue #25; escalated as bug/pyautoreduce prompt) ----------
# The pipeline has no DQ-bits dial, so AstroDrizzle's default treats every
# flagged pixel as bad. PJ011646's five snapshot exposures all carry DQ 512
# (WFC3-IR *blob*) at the same detector pixels with only 2-6 px dithers, so
# a structured zero-coverage hole lands in the cutout (outside the 3.8"
# science mask) and the packaging guard rightly refuses to ship it.
# Standard IR practice (and Aris's reduction, which has no hole) passes 512
# as usable. Inject final_bits/driz_sep_bits=512 here until the adapter
# grows the dial.
from autoreduce.drizzle import combine as _combine_mod  # noqa: E402

_orig_drizzle_kwargs_for = _combine_mod.drizzle_kwargs_for


def _ir_bits_kwargs(spec, adapter, n_exposures):
    kwargs = _orig_drizzle_kwargs_for(spec, adapter, n_exposures)
    kwargs.update(final_bits=512, driz_sep_bits=512)
    return kwargs


_combine_mod.drizzle_kwargs_for = _ir_bits_kwargs
# ---------------------------------------------------------------------------

ARIS_DIR = Path("/mnt/c/Users/Jammy/Science/aris_PJ011646/dataset/aris/PJ011646")
CACHE_ROOT = REPO / "scripts" / "cache"
OUTPUT_ROOT = REPO / "scripts" / "output"

SPEC = TargetSpec(
    name="pj011646",
    ra=19.194875,
    dec=-24.617194,
    instrument="wfc3_ir",
    filter_name="F160W",
    proposal_ids=("14653",),
    cutout_shape=(201, 201),
    final_scale=0.0642,
    final_pixfrac=1.0,
)


def main():
    record = reduce_target(SPEC, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)
    print(json.dumps(record["drizzle"]["weight_uniformity"], indent=2))

    from astropy.io import fits

    from autoreduce.validation import registered_ratios

    out_dir = OUTPUT_ROOT / SPEC.name
    new_data = fits.getdata(out_dir / "data.fits").astype(float)
    new_noise = fits.getdata(out_dir / "noise_map.fits").astype(float)
    aris_data = fits.getdata(ARIS_DIR / "data.fits").astype(float)
    aris_noise = fits.getdata(ARIS_DIR / "noise_map.fits").astype(float)

    summary = {
        "n_exposures": record["acquire"]["n_exposures"],
        **registered_ratios(new_data, new_noise, aris_data, aris_noise),
        "correlated_noise_factor_applied": record["noise"]["correlated_noise_factor"],
        "psf_diagnostics": record["psf"],
    }
    print("[parity] ---- production parity (translation-only; orientation")
    print("[parity]      handled in the phase-2 comparison script) ----")
    print(json.dumps(summary, indent=2))
    (out_dir / "parity_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
