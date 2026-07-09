"""
ALMA spike (issue #14): 2016.1.00282.S / G09v1.40 through the visibility
branch — the phase-5 validation anchor (docs/design/alma.md).

Phased and degradable: each phase prints + writes JSON under
prototypes/output/alma_g09v140/, so a failure in a later phase still leaves
the earlier evidence on disk.

  phase 1  acquire: resolve calibrated MS (--ms-dir; the ARC-delivered /
           scriptForPI-restored layout) or download the project tarballs
           from the archive and report what they contain.
  phase 2  reduce: TargetSpec -> reduce_target through split/extract/
           assemble/package (modular casatools + casatasks, headless).
  phase 3  reference comparison: numeric match of visibilities and
           uv_wavelengths against the modeler's exported files
           (--reference-dir, his uv_wavelengths_*/visibilities_* fits) —
           per-(uid, spw) blocks, before Stokes-I averaging is compared
           the reference does not average polarizations.
  phase 4  autolens round-trip: al.Interferometer.from_fits on the packaged
           products + a dirty-image sanity check (source visible).

Run:  ~/venv/PyAuto/bin/python prototypes/alma_g09v140_spike.py \
          --ms-dir /path/to/calibrated_ms [--reference-dir /path/to/aris]
casatools + casatasks required (pip); phase 4 needs autolens; unit tests
never import this.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from autoreduce.acquire import alma as alma_acquire  # noqa: E402
from autoreduce.pipeline import reduce_target  # noqa: E402
from autoreduce.target import TargetSpec  # noqa: E402
from autoreduce.visibilities import extract as extract_mod  # noqa: E402
from autoreduce.visibilities import split as split_mod  # noqa: E402

PROJECT = "2016.1.00282.S"
FIELD = "G09v1.40"
UIDS = ("A002_Xb9b1b9_X3046", "A002_Xb99cbd_X2456")
SPWS = ("1", "2")
WIDTH = 240
# G09v1.40 (H-ATLAS J085358.9+015537).
RA, DEC = 133.49542, 1.59367

OUT = REPO / "prototypes" / "output" / "alma_g09v140"
CACHE = REPO / "prototypes" / "cache" / "alma_g09v140"


def _write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, default=str))
    print(f"  wrote {OUT / name}")


def phase_1_acquire(ms_dir: Path) -> list:
    print(f"phase 1: acquire ({'local ' + str(ms_dir) if ms_dir else 'archive'})")
    if ms_dir:
        paths = alma_acquire.resolve_calibrated_ms(ms_dir, UIDS)
    else:
        tarballs = alma_acquire.download_product_tarballs(
            PROJECT, CACHE / "tarballs"
        )
        extracted = alma_acquire.extract_calibrated_ms_from_tarballs(
            tarballs, CACHE / "ms"
        )
        if not extracted:
            raise SystemExit(
                alma_acquire.restore_guidance(PROJECT, CACHE / "tarballs")
            )
        paths = alma_acquire.resolve_calibrated_ms(CACHE / "ms", UIDS)
    facts = []
    for path in paths:
        num_chan = extract_mod.num_channels_per_spw(path)
        facts.append({"ms": path.name, "num_chan_per_spw": num_chan.tolist()})
        print(f"  {path.name}: NUM_CHAN per spw = {num_chan.tolist()}")
    _write("phase1_acquire.json", {"measurement_sets": facts})
    return paths


def phase_2_reduce(ms_dir: Path) -> dict:
    print("phase 2: reduce_target through the visibility branch")
    spec = TargetSpec(
        name="alma_g09v140",
        ra=RA,
        dec=DEC,
        instrument="alma",
        alma_uids=UIDS,
        alma_field=FIELD,
        alma_spws=SPWS,
        alma_width=WIDTH,
        alma_ms_dir=str(ms_dir) if ms_dir else None,
        alma_project_code=PROJECT,
    )
    record = reduce_target(spec, cache_root=CACHE, output_root=OUT)
    print(
        f"  packaged {record['package']['n_visibilities']} visibilities "
        f"({record['assemble']['blocks'].keys()})"
    )
    _write("phase2_reduce.json", record)
    return record


def phase_3_reference_compare(ms_dir: Path, reference_dir: Path) -> None:
    """
    Compare raw per-(uid, spw) extraction against the modeler's exports:
    his visibilities are per polarization ([2, Nvis, 2]) and his
    uv_wavelengths per channel — so the comparison runs on the split MS
    columns before Stokes-I averaging.
    """
    from astropy.io import fits

    print(f"phase 3: reference comparison against {reference_dir}")
    report = {}
    work_dir = OUT / "alma_g09v140" / "work"
    for uid in UIDS:
        for spw in SPWS:
            tag = f"{uid}_{FIELD}_spw_{spw}_width_{WIDTH}"
            ref_vis_path = reference_dir / f"visibilities_{tag}.fits"
            ref_uv_path = reference_dir / f"uv_wavelengths_{tag}.fits"
            if not (ref_vis_path.exists() and ref_uv_path.exists()):
                report[tag] = "reference files missing — skipped"
                print(f"  {tag}: reference missing, skipped")
                continue
            spw_ms = split_mod.spw_ms_path(work_dir, uid, FIELD, spw, WIDTH)
            columns = extract_mod.columns_from(spw_ms)
            ours_vis = np.stack(
                (columns.data.real, columns.data.imag), axis=-1
            ).squeeze()
            ref_vis = np.squeeze(fits.getdata(ref_vis_path))
            ours_uv = np.squeeze(
                np.stack(
                    (
                        columns.uvw[0][None, :]
                        * columns.chan_freq[:, None]
                        / 299792458.0,
                        columns.uvw[1][None, :]
                        * columns.chan_freq[:, None]
                        / 299792458.0,
                    ),
                    axis=-1,
                )
            )
            ref_uv = np.squeeze(fits.getdata(ref_uv_path))
            vis_match = bool(
                ours_vis.shape == ref_vis.shape
                and np.allclose(ours_vis, ref_vis, rtol=1e-6, atol=0.0)
            )
            uv_match = bool(
                ours_uv.shape == ref_uv.shape
                and np.allclose(ours_uv, ref_uv, rtol=1e-6, atol=0.0)
            )
            report[tag] = {
                "visibilities_shape_ours": list(ours_vis.shape),
                "visibilities_shape_ref": list(ref_vis.shape),
                "visibilities_match": vis_match,
                "uv_wavelengths_match": uv_match,
            }
            print(f"  {tag}: vis match={vis_match}, uv match={uv_match}")
    _write("phase3_reference.json", report)


def phase_4_autolens_round_trip() -> None:
    print("phase 4: al.Interferometer round-trip + dirty image")
    import autolens as al

    product_dir = OUT / "alma_g09v140"
    real_space_mask = al.Mask2D.circular(
        shape_native=(256, 256), pixel_scales=0.05, radius=4.0
    )
    dataset = al.Interferometer.from_fits(
        data_path=product_dir / "data.fits",
        noise_map_path=product_dir / "noise_map.fits",
        uv_wavelengths_path=product_dir / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerNUFFT,
    )
    dirty = dataset.dirty_image
    peak_snr = float(np.max(np.abs(dirty)) / np.std(dirty))
    print(
        f"  {dataset.data.shape[0]} visibilities; dirty-image peak/rms = "
        f"{peak_snr:.1f} (expect >> 1 for a detected ring)"
    )
    _write(
        "phase4_autolens.json",
        {"n_visibilities": int(dataset.data.shape[0]), "dirty_peak_over_rms": peak_snr},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms-dir", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--skip-autolens", action="store_true")
    args = parser.parse_args()

    phase_1_acquire(args.ms_dir)
    phase_2_reduce(args.ms_dir)
    if args.reference_dir:
        phase_3_reference_compare(args.ms_dir, args.reference_dir)
    if not args.skip_autolens:
        phase_4_autolens_round_trip()
