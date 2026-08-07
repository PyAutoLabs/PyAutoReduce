# dataset/

**No FITS is ever committed to this repository** — the `.gitignore` enforces it. Reduced
products (`data.fits`, `noise_map.fits`, `psf.fits`, `psf_full.fits`, `reduction.json`) land in
`output/<target>/` at runtime, and downloaded exposures live in `cache/`; both directories are
gitignored and created by the scripts.

This folder holds only **per-target `TargetSpec` YAML files** that you may add for your own
targets. A spec file is the declarative input of a reduction — load it with
`TargetSpec.from_yaml(path)` and pass it to `reduce_target` (see
`scripts/guides/target_spec.py` for every dial). An example:

```yaml
# dataset/slacs0008-0004.yaml
name: slacs0008-0004
ra: 2.012333
dec: -0.068944
proposal_ids: [10886]
cutout_shape: [281, 281]
final_pixfrac: 0.6
```

```python
from autoreduce import TargetSpec, reduce_target

spec = TargetSpec.from_yaml("dataset/slacs0008-0004.yaml")
record = reduce_target(spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)
```

Keeping one YAML per target makes a sample reproducible: the pipeline is a pure function of the
spec plus the archive, so committing the spec (never the FITS) is enough to reproduce the
dataset.
