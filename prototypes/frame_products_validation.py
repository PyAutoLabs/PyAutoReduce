"""
Real-data validation of the per-exposure frame-products mode (issue #16):
reduce slacs0008-0004 twice from a warm cache — flag off (baseline), then
`frame_products=True` — byte-compare the mosaic products between the runs,
then check the frames/ tree is modeling-ready.

Run:  ~/venv/PyAuto/bin/python prototypes/frame_products_validation.py \
          [--cache-root ...] [--output-root ...]
Network only on a cold cache; drizzlepac + deepCR required; unit tests never
import this.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from autoreduce import TargetSpec, reduce_target  # noqa: E402

MOSAIC_PRODUCTS = ("data.fits", "noise_map.fits", "psf.fits", "psf_full.fits")


def _spec(frame_products: bool) -> TargetSpec:
    # The issue-#2 acceptance target: proposal-filtered production set.
    return TargetSpec(
        name="slacs0008-0004",
        ra=2.012333,
        dec=-0.068944,
        proposal_ids=("10886",),
        frame_products=frame_products,
    )


def _check(condition: bool, label: str, failures: list) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        failures.append(label)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=REPO / "scripts" / "cache")
    parser.add_argument(
        "--output-root", type=Path, default=REPO / "prototypes" / "output" / "frame_products_validation"
    )
    args = parser.parse_args()

    failures: list = []
    records = {}
    for label, flag in (("baseline", False), ("frames_on", True)):
        out_root = args.output_root / label
        print(f"=== reduce_target [{label}] -> {out_root}", flush=True)
        records[label] = reduce_target(
            _spec(flag), cache_root=args.cache_root, output_root=out_root
        )

    base_dir = args.output_root / "baseline" / "slacs0008-0004"
    frames_run_dir = args.output_root / "frames_on" / "slacs0008-0004"

    # 1. Mosaic path untouched: products byte-identical across the two runs.
    for product in MOSAIC_PRODUCTS:
        same = (base_dir / product).read_bytes() == (
            frames_run_dir / product
        ).read_bytes()
        _check(same, f"mosaic {product} byte-identical", failures)

    # 2. Provenance: fragment present iff the flag is on.
    _check("frames" not in records["baseline"], "baseline record has no frames key", failures)
    fragment = records["frames_on"].get("frames", {})
    _check(bool(fragment), "frames_on record carries the frames fragment", failures)
    print(json.dumps(fragment, indent=2))

    # 3. frames/ tree vs manifest.
    frames_dir = frames_run_dir / "frames"
    manifest = json.loads((frames_dir / "manifest.json").read_text())
    entries = manifest["frames"]
    chip_dirs = sorted(d for d in frames_dir.iterdir() if d.is_dir())
    _check(len(chip_dirs) == len(entries), f"chip dirs ({len(chip_dirs)}) == manifest entries ({len(entries)})", failures)
    _check(
        len(entries) + len(manifest["skipped_chips"])
        == 2 * records["frames_on"]["drizzle"]["n_exposures"],
        "entries + skipped == 2 chips per ACS exposure",
        failures,
    )
    for d in chip_dirs:
        for product in ("data.fits", "noise_map.fits", "dq.fits", "cr_mask.fits"):
            _check((d / product).exists(), f"{d.name}/{product} exists", failures)

    # 4. One frame is modeling-ready: finite data, positive noise, masked-by-
    #    noise applied, CR pixels flagged, target anchor consistent with the
    #    written WCS (SIP-only header vs full-distortion anchor: <0.5 px).
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    total_cr = sum(e["n_cr_pixels"] for e in entries)
    _check(total_cr > 0, f"deepCR flagged pixels across frames (total {total_cr})", failures)
    shape = tuple(manifest["frame_cutout_shape"])
    for entry in entries[:1] + entries[-1:]:
        d = frames_dir / entry["dir"]
        data = fits.getdata(d / "data.fits").astype(float)
        noise = fits.getdata(d / "noise_map.fits").astype(float)
        cr = fits.getdata(d / "cr_mask.fits")
        _check(data.shape == shape and noise.shape == shape, f"{entry['dir']} shapes == {shape}", failures)
        _check(bool(np.isfinite(data).all()), f"{entry['dir']} data finite", failures)
        _check(bool((noise > 0).all()), f"{entry['dir']} noise positive", failures)
        if entry["n_masked_pixels"]:
            _check(
                float(noise.max()) == 1.0e8, f"{entry['dir']} masked-by-noise applied", failures
            )
            _check(bool((data[noise >= 1.0e8] == 0.0).all()), f"{entry['dir']} masked data zeroed", failures)
        if entry["n_cr_pixels"]:
            _check(
                bool((noise[cr.astype(bool)] >= 1.0e8).all()),
                f"{entry['dir']} CR pixels masked in noise",
                failures,
            )
        with fits.open(d / "data.fits") as hdul:
            x, y = WCS(hdul[0].header).world_to_pixel_values(
                manifest["target"]["ra"], manifest["target"]["dec"]
            )
        dx = abs(float(x) - entry["target_pixel"][0])
        dy = abs(float(y) - entry["target_pixel"][1])
        _check(
            dx < 0.5 and dy < 0.5,
            f"{entry['dir']} target_pixel vs header WCS ({dx:.3f}, {dy:.3f}) px",
            failures,
        )

    print("=" * 60)
    if failures:
        print(f"VALIDATION FAILED — {len(failures)} check(s):")
        for f in failures:
            print(" -", f)
        return 1
    print(f"VALIDATION PASSED — {len(entries)} chips, {total_cr} deepCR pixels, mosaic byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
