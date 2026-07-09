"""
The visibility-branch orchestrator (docs/design/alma.md):

    acquire -> split -> extract -> assemble -> package

Dispatched from `autoreduce.pipeline.reduce_target` when the instrument
adapter's domain is "visibility"; shares the TargetSpec / ExposureCache /
provenance machinery with the imaging pipeline and none of its stages.
"""

from pathlib import Path
from typing import Dict, List

from ..acquire import alma as alma_acquire
from ..acquire import cache as cache_mod
from ..package import interferometer as interferometer_mod
from ..package import provenance as provenance_mod
from ..target import TargetSpec
from . import assemble as assemble_mod
from . import extract as extract_mod
from . import split as split_mod


def _require(spec: TargetSpec) -> None:
    missing = [
        name
        for name in ("alma_uids", "alma_field", "alma_spws")
        if not getattr(spec, name)
    ]
    if missing:
        raise ValueError(
            f"visibility reduction of {spec.name!r} requires TargetSpec "
            f"fields {missing} (docs/design/alma.md, spec stage)"
        )


def _acquire(spec: TargetSpec, cache: cache_mod.ExposureCache) -> List[Path]:
    """
    Calibrated per-uid measurement sets: from ``alma_ms_dir`` when given
    (ARC delivery / prior restore — today's common case), else from the
    archive download path, loud with restore guidance when the tarballs
    carry no calibrated MS (design doc, "Calibrated-MS acquisition").
    """
    if spec.alma_ms_dir:
        return alma_acquire.resolve_calibrated_ms(
            Path(spec.alma_ms_dir), spec.alma_uids
        )
    if not spec.alma_project_code:
        raise ValueError(
            f"{spec.name!r}: set alma_ms_dir (calibrated MS directory) or "
            f"alma_project_code (archive download) — neither is present"
        )
    target_dir = cache.target_dir(spec.name)
    ms_dir = target_dir / "ms"
    try:
        return alma_acquire.resolve_calibrated_ms(ms_dir, spec.alma_uids)
    except FileNotFoundError:
        pass  # not yet downloaded/extracted — fall through to the archive
    tarball_dir = target_dir / "tarballs"
    tarballs = sorted(tarball_dir.glob("*.tar")) or (
        alma_acquire.download_product_tarballs(
            spec.alma_project_code, tarball_dir
        )
    )
    cache.record_download(
        spec.name, [str(p) for p in tarballs], source="alma"
    )
    extracted = alma_acquire.extract_calibrated_ms_from_tarballs(
        tarballs, ms_dir
    )
    if not extracted:
        raise FileNotFoundError(
            alma_acquire.restore_guidance(spec.alma_project_code, tarball_dir)
        )
    return alma_acquire.resolve_calibrated_ms(ms_dir, spec.alma_uids)


def reduce_visibility_target(
    spec: TargetSpec,
    adapter,
    cache: cache_mod.ExposureCache,
    out_dir: Path,
    work_dir: Path,
) -> Dict:
    """Run the visibility branch for one target; returns the provenance record."""
    _require(spec)
    record: Dict = {"target": spec.as_dict(), "instrument": adapter.key}

    ms_paths = _acquire(spec, cache)
    record["acquire"] = {
        "measurement_sets": [p.name for p in ms_paths],
        "source": "local" if spec.alma_ms_dir else "alma-archive",
    }

    sets, labels, split_record = [], [], []
    for uid, ms in zip(spec.alma_uids, ms_paths):
        field_ms = split_mod.split_field(ms, uid, spec.alma_field, work_dir)
        num_chan = extract_mod.num_channels_per_spw(ms)
        for spw in spec.alma_spws:
            width = split_mod.resolve_width(spec.alma_width, spw, num_chan)
            spw_ms = split_mod.split_spw(
                field_ms, uid, spec.alma_field, spw, width, work_dir
            )
            columns = extract_mod.columns_from(spw_ms)
            sets.append(assemble_mod.assemble_ms_products(columns))
            labels.append(f"{uid}/spw{spw}")
            split_record.append(
                {"uid": uid, "spw": str(spw), "width": int(width)}
            )
    record["split"] = {"blocks": split_record, "field": spec.alma_field}

    combined = assemble_mod.concatenate(sets, labels)
    record["assemble"] = combined.provenance

    products = interferometer_mod.write_products(
        out_dir,
        combined.visibilities,
        combined.uv_wavelengths,
        combined.noise_map,
    )
    record["package"] = {
        "products": products,
        "n_visibilities": int(combined.visibilities.shape[0]),
        "contract": "al.Interferometer.from_fits",
    }
    provenance_mod.write_reduction_json(out_dir, record)
    return record
