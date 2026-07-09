"""
Keck-AO spike (issue #11): B1938+666 through the nirc2_native path — the
phase-4 analogue of the slacs0008 spike.

Phased and degradable: each phase prints + writes JSON under
prototypes/output/b1938_keck/, so a failure in a later phase (archive
quirks, column names) still leaves the earlier evidence on disk.

  phase 1  discover: cone-query KOA for the SHARP B1938+666 NIRC2 frames
           (narrow camera, K'); report programs / nights / frame counts.
  phase 2  psf stars: list same-night object frames pointing away from the
           target; group into epochs — the tier-A PSF-star candidates.
  phase 3  reduce: pinned-id TargetSpec -> reduce_target.
  phase 4  checks: blank-sky noise closure, PSF FWHM vs SHARP's ~65-70 mas,
           weight uniformity (acceptance checks 1-2 of keck_ao.md).

Run:  ~/venv/PyAuto/bin/python prototypes/b1938_keck_spike.py [--max-frames N]
Network + pykoa + drizzle required; unit tests never import this.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from autoreduce import instruments  # noqa: E402
from autoreduce.acquire import koa  # noqa: E402
from autoreduce.target import TargetSpec  # noqa: E402
from autoreduce.pipeline import reduce_target  # noqa: E402

# B1938+666 (SHARP I; Lagattuta et al. 2012): observed UT 2010 June 29-30,
# NIRC2 narrow camera, K' (14,760 s) — pre-2015, i.e. Yelda-solution epoch.
RA, DEC = 294.60496, 66.81450
FILTER = "Kp"

OUT = REPO / "prototypes" / "output" / "b1938_keck"
CACHE = REPO / "prototypes" / "cache" / "b1938_keck"


def _dump(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, default=str))
    print(f"[spike] wrote {name}")


def phase1_discover(adapter):
    table = koa.query_science_frames(
        RA, DEC, adapter, FILTER, OUT, proposal_ids=None, koa_ids=None
    )
    cols = table.colnames
    rows = [
        {c: str(r[c]) for c in cols if c in
         ("koaid", "progid", "date_obs", "itime", "coadds", "filter", "camname")}
        for r in table
    ]
    _dump("phase1_science_frames.json", {"n": len(table), "columns": cols, "rows": rows})
    return table


def phase2_split_pointings(science_table):
    """
    The cone result carries two pointing clusters, both targname'd by the
    tip-tilt star (SHARP convention): the lens dithers, and the PSF/tt star
    itself ~20" away, interleaved in time (Lagattuta et al. 2012's PSF-star
    strategy). Split them by pointing separation from the lens.
    """
    import numpy as np

    sep = (
        np.hypot(
            (np.asarray(science_table["ra"], float) - RA)
            * np.cos(np.radians(DEC)),
            np.asarray(science_table["dec"], float) - DEC,
        )
        * 3600.0
    )
    lens = science_table[sep < 12.0]
    star = science_table[sep >= 12.0]
    summary = {
        "n_lens_pointings": len(lens),
        "n_star_pointings": len(star),
        "star_cluster_offset_arcsec": [
            round(float((np.median(np.asarray(star["ra"], float)) - RA)
                        * np.cos(np.radians(DEC)) * 3600.0), 1),
            round(float((np.median(np.asarray(star["dec"], float)) - DEC)
                        * 3600.0), 1),
        ] if len(star) else None,
        "star_itimes": sorted({float(i) for i in star["itime"]}) if len(star) else [],
    }
    _dump("phase2_pointing_split.json", summary)
    return lens, star


def phase3_reduce(lens_table, star_table, max_frames):
    from collections import Counter

    import numpy as np

    # Pin the modal science setup (the pipeline refuses mixed ITIME/COADDS;
    # short acquisition frames drop out here) — for SHARP B1938: 180s x 1.
    setups = Counter((float(r["itime"]), int(r["coadds"])) for r in lens_table)
    (modal_itime, modal_coadds), _ = setups.most_common(1)[0]
    print(f"[spike] lens setups {dict(setups)}; using {modal_itime}s x {modal_coadds}")
    science_ids = [
        str(r["koaid"])
        for r in lens_table
        if float(r["itime"]) == modal_itime and int(r["coadds"]) == modal_coadds
    ][:max_frames]

    # PSF star: prefer the shortest-ITIME frames (a K~13-14 tip-tilt star
    # saturates the narrow camera in 180 s; the short frames exist for the
    # unsaturated core). Fall back to everything if no short frames exist.
    star_itimes = np.asarray(star_table["itime"], float)
    short = star_itimes <= 60.0
    chosen = star_table[short] if short.any() else star_table
    star_ids = [str(k) for k in chosen["koaid"]][:12]
    spec = TargetSpec(
        name="b1938+666",
        ra=RA,
        dec=DEC,
        instrument="nirc2_narrow",
        filter_name=FILTER,
        final_scale=0.010,  # SHARP convention
        final_pixfrac=1.0,
        cutout_shape=(281, 281),
        koa_science_ids=tuple(science_ids),
        koa_psf_star_ids=tuple(star_ids) or None,
    )
    record = reduce_target(spec, cache_root=CACHE, output_root=OUT)
    _dump("phase3_reduction_record.json", record)
    return spec, record


def phase4_checks(spec, record):
    import numpy as np
    from astropy.io import fits

    from autoreduce.noise.rms import empirical_background_rms

    out_dir = OUT / spec.name
    data = fits.getdata(out_dir / "data.fits").astype(float)
    noise = fits.getdata(out_dir / "noise_map.fits").astype(float)

    # Check 1 — blank-sky closure. The noise map is the decorrelated-
    # equivalent (x R, chi^2-correct) while the measurable mosaic RMS is
    # correlation-suppressed (~ /R by the same Casertano argument), so the
    # apples-to-apples statistic is empirical x R^2 vs the map's background
    # floor: ~1 when the per-frame budget (sky + dark + MCDS read noise) is
    # right, x6-x40 off on unit errors.
    r_factor = float(record["noise"]["correlated_noise_factor"])
    empirical = empirical_background_rms(data)
    predicted_floor = float(np.nanmedian(noise[noise < np.nanpercentile(noise, 50)]))
    closure = empirical * r_factor**2 / predicted_floor

    # Check 2 — PSF core FWHM vs SHARP's ~65-70 mas. A tier-B fallback
    # (no candidates key) is reported explicitly, never as an empty pass.
    psf_diag = record["psf"]
    if "candidates" not in psf_diag:
        raise RuntimeError(
            f"reduction fell back to a non-tier-A PSF ({psf_diag.get('method')}) "
            f"— the B1938 acceptance run requires PSF-star candidates"
        )
    checks = {
        "blank_sky_closure_empirical_R2_over_predicted": round(closure, 3),
        "closure_pass_0.6_1.6": bool(0.6 < closure < 1.6),
        "psf_candidates": psf_diag.get("candidates"),
        "fwhm_in_sharp_range_45_120mas": [
            bool(0.045 < c["fwhm_arcsec"] < 0.120)
            for c in psf_diag.get("candidates", [])
        ],
        "weight_uniformity_cutout": record["drizzle"].get(
            "weight_uniformity_cutout"
        ),
        "n_exposures": record["acquire"]["n_exposures"],
        "total_exptime_s": record["drizzle"].get("total_exptime"),
    }
    _dump("phase4_acceptance_checks.json", checks)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--stop-after", type=int, default=4)
    args = parser.parse_args()

    adapter = instruments.get("nirc2_narrow")
    science = phase1_discover(adapter)
    if args.stop_after < 2:
        return
    lens, stars = phase2_split_pointings(science)
    if args.stop_after < 3:
        return
    spec, record = phase3_reduce(lens, stars, args.max_frames)
    if args.stop_after < 4:
        return
    checks = phase4_checks(spec, record)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
