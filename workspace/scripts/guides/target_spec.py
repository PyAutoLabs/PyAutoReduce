"""
Guides: The Target Spec
=======================

A **PyAutoReduce** reduction is declared, not scripted: everything the pipeline will do to your
target is stated up front on one frozen `TargetSpec`, and `reduce_target` is a pure function of
that spec plus the archive. This guide is the reference for the declaration itself — every dial
annotated, the YAML round-trip that makes specs committable, the `dataclasses.replace` idiom for
variants, and the validation guard rails that reject a bad declaration before a single byte is
downloaded.

Constructing specs touches no network and no heavy dependency, so this entire guide runs offline
in seconds — it is the one workspace script you can always run.

__Contents__

- **The Declaration Philosophy:** Why the spec is the reduction, and what that buys you.
- **Imports:** `TargetSpec`, the instrument registry, and the standard library.
- **The Full Dial Set:** One construction naming every dial, with what each one does.
- **Instrument Adapters:** The registry behind the `instrument` dial — native scales and recommendations.
- **YAML Round Trip:** Write a spec to YAML, load it back, prove they are equal.
- **Variants With replace:** The frozen-dataclass idiom for dial studies.
- **Validation Guard Rails:** The `ValueError`s a bad declaration earns, demonstrated.
- **Pre-Flight Guards:** The cross-dial combinations `reduce_target` rejects before any download.
- **Wrap Up:** Summary and good places to checkout next.

__The Declaration Philosophy__

Reduction scripts rot: a chain of tool calls with tweaked parameters, run once, half-remembered.
A declaration does not. The `TargetSpec` is a frozen dataclass carrying the target's identity,
the instrument, and every literature-contested dial — pixel scale, pixfrac, kernel, cosmic-ray
method, PSF options — so that the spec *is* the reduction: same spec plus same archive equals
same dataset, modulo upstream reference-file updates, which the provenance records.

This buys three things. Reproducibility: commit one small YAML per target (see
`dataset/README.md`) and anyone regenerates your sample. Auditability: the full spec is embedded
in every `reduction.json`, so a dataset always carries its own definition. And honesty: dials the
literature disagrees on (published HST lens reductions span no-drizzle to pixfrac 0.6 to 1.0)
are surfaced as user-facing configuration, never buried defaults.

__Imports__

Everything here is public API and light: the spec class, the instrument registry, and the
standard library for the YAML round-trip and variants.
"""

import dataclasses
import tempfile
from pathlib import Path

from autoreduce import TargetSpec, instruments
from autoreduce.instruments import nircam_adapter_for_filter

"""
__The Full Dial Set__

One construction, every dial, each with its job. The values below are the defaults (so this spec
is equivalent to `TargetSpec(name=..., ra=..., dec=...)`) except where a trailing comment says
otherwise — the point is the annotated tour, not the target.
"""
spec = TargetSpec(
    # Identity:
    name="slacs0008-0004",          # Names the output folder: output/<name>/.
    ra=2.012333,                    # Target right ascension, degrees (J2000).
    dec=-0.068944,                  # Target declination, degrees (J2000).
    instrument="acs_wfc",           # Adapter key — see the registry section below for all 11.
    filter_name="F814W",            # The band to reduce.
    proposal_ids=("10886",),        # Pin acquisition to these programmes; None = everything at the coords.
    # Acquisition:
    sync_references=True,           # Re-sync CRDS reference files each HST run; False = offline opt-out
                                    # for a previously-warmed cache (raises without one).
    # Packaging:
    cutout_shape=(281, 281),        # Science/noise cutout, output pixels (~14" at 0.05"/pix). Strict
                                    # coverage: a cutout off the mosaic raises — size to the data.
    # Drizzle (imaging combine):
    final_scale=0.05,               # Output pixel scale, arcsec/pix (the SLACS convention for ACS).
    final_pixfrac=0.8,              # Drizzle drop size, (0, 1]; smaller = less correlated noise but
                                    # needs richer dithers (see guides/noise_maps.py).
    final_kernel="square",          # Drizzle kernel; published practice varies, so it is a dial.
    cr_method="driz_cr",            # Cosmic rays: "driz_cr" (STScI stack rejection, default) or
                                    # "deepcr" (per-frame CNN masks; HST ACS/UVIS only, opt-in).
    # PSF:
    psf_shape=(21, 21),             # Compact convolution kernel; must be odd.
    psf_full_shape=(61, 61),        # Extended wings product; must be odd.
    psf_backend="epsf",             # Mosaic-path back-end: "epsf" (photutils) or "starred"
                                    # (super-sampled ePSF; the [starred] extra).
    psf_star_pass="auto",           # Which drizzle pass feeds star finding: "auto" (no extra cost),
                                    # "science" (pin to shipped mosaic), "no_cr" (dedicated
                                    # CR-flag-ignoring second drizzle; HST opt-in).
    psf_from_frames=False,          # Build the mosaic PSF by combining per-frame ePSFs through the
                                    # drizzle geometry instead of from mosaic stars (HST/JWST).
    # Frame products:
    frame_products=False,           # Also package every calibrated exposure as a native-scale
                                    # per-frame dataset under output/<name>/frames/ (HST/JWST/Keck).
    # Alignment:
    alignment_tolerance_pix=0.1,    # Recorded intent for TweakReg refinement; currently unread —
                                    # the pipeline trusts the archive's Gaia-tied WCS and records
                                    # the cross-correlation evidence instead (honest caveat).
    # Synthetic-source injection (see the simulator.py scripts):
    inject_image=None,              # Path to a plain FITS image to inject into the real frames
                                    # (e-/s per pixel for HST/Keck, Jy per pixel for JWST).
    inject_pixel_scale=None,        # arcsec/pix of that image; REQUIRED with inject_image.
    inject_position=None,           # (ra, dec) degrees to centre it on; None = the target.
    inject_psf=None,                # PSF FITS to convolve with; None = each frame's own ePSF.
    inject_seed=0,                  # Seed for the injected source's Poisson realisations.
    # ALMA / visibility branch (ignored by imaging instruments):
    alma_uids=None,                 # Execution-block uids pinning the measurement sets.
    alma_field=None,                # Science field name inside the MS, e.g. "G09v1.40".
    alma_spws=None,                 # Spectral windows to extract; leave line-bearing spws out.
    alma_width=0,                   # Channel-averaging width; 0 = collapse each spw (continuum).
    alma_ms_dir=None,               # Local calibrated-MS directory; None = archive acquisition.
    alma_project_code=None,         # ALMA project code for archive acquisition.
    alma_sim_antennalist="alma.cycle8.3.cfg",  # simobserve dials (active only when injecting
    alma_sim_totaltime_s=1800.0,               # on a visibility-domain instrument): array config,
    alma_sim_integration_s=10.0,               # total on-source time, integration time,
    alma_sim_freq_ghz=230.0,                   # observing frequency,
    alma_sim_pwv_mm=0.5,                       # and precipitable water vapour (0 = noiseless).
    # Keck / ground-based branch (ignored by space-based instruments):
    koa_science_ids=None,           # Explicit KOA frame ids pinning the science set (the raw
                                    # archive has no association tables).
    koa_psf_star_ids=None,          # PSF-star frames reduced pipeline-identically (tier A).
    sky_window=9,                   # Running-sky window: temporally adjacent frames medianed
                                    # per frame (K' sky varies on minutes timescales).
)

print(f"Declared: {spec.name} ({spec.instrument}/{spec.filter_name}), "
      f"{spec.final_scale}\"/pix, pixfrac {spec.final_pixfrac}")

"""
__Instrument Adapters__

The `instrument` dial selects an adapter from a registry — everything instrument-specific
(native pixel scale, calibrated product type, combine backend, archive, recommended output
scale) lives behind it, which is why the rest of the spec is instrument-agnostic.
"""
print(f"Registered instruments: {sorted(instruments.registered_keys())}")

for key in ("acs_wfc", "wfc3_uvis", "wfc3_ir", "nircam_sw", "nircam_lw", "nirc2_narrow"):
    adapter = instruments.get(key)
    print(
        f"  {key:13s} native {adapter.native_scale:.4f}\"/pix, "
        f"recommended final_scale {adapter.recommended_final_scale}\"/pix"
    )

"""
The adapter's `recommended_final_scale` documents sensible sampling for the detector —
`final_scale` remains your dial, but deviating from the recommendation is a choice the
instrument scripts discuss (WFC3/IR's 0.065" recommendation against its 0.128" native pixels is
the sharpest example).

NIRCam has a routing helper you should always use rather than picking `nircam_sw`/`nircam_lw` by
hand — the filter name determines the channel:
"""
for filter_name in ("F115W", "F277W"):
    adapter = nircam_adapter_for_filter(filter_name)
    print(f"  {filter_name} -> {adapter.key} ({adapter.native_scale:.3f}\"/pix native)")

"""
__YAML Round Trip__

Specs are committable: `TargetSpec.from_yaml` loads a per-target YAML file (coercing lists to
the tuples the dataclass wants), which is how a reduced sample stays reproducible — one YAML per
target in `dataset/`, never the FITS. We write one to a temporary path and prove the round trip
is exact.
"""
yaml_text = """\
name: slacs0008-0004
ra: 2.012333
dec: -0.068944
proposal_ids: [10886]
cutout_shape: [281, 281]
final_pixfrac: 0.8
"""

with tempfile.TemporaryDirectory() as tmp:
    yaml_path = Path(tmp) / "slacs0008-0004.yaml"
    yaml_path.write_text(yaml_text)
    spec_from_yaml = TargetSpec.from_yaml(yaml_path)

print(f"Round trip exact: {spec_from_yaml == spec}")
print(f"as_dict() keys: {len(spec.as_dict())} dials serialised into every reduction.json")

"""
__Variants With replace__

The spec is a frozen dataclass — you never mutate one, you derive variants with
`dataclasses.replace`. This is the idiom for dial studies (compare pixfracs, try the deepCR CR
route, request frame products) while everything unstated stays identical, and it is exactly how
`scripts/hst_acs/dials.py` builds its trade study.
"""
spec_pixfrac06 = dataclasses.replace(spec, name="slacs0008-0004_p06", final_pixfrac=0.6)
spec_deepcr = dataclasses.replace(spec, name="slacs0008-0004_deepcr", cr_method="deepcr")
spec_frames = dataclasses.replace(spec, name="slacs0008-0004_frames", frame_products=True)

for variant in (spec_pixfrac06, spec_deepcr, spec_frames):
    print(f"  {variant.name}: pixfrac {variant.final_pixfrac}, cr {variant.cr_method}, "
          f"frames {variant.frame_products}")

"""
Give each variant its own `name` — the name is the output folder, and two specs sharing one
would overwrite each other's products.

__Validation Guard Rails__

A declaration is only trustworthy if a bad one cannot exist, so `TargetSpec` validates at
construction and raises `ValueError` immediately — long before `reduce_target`, the network, or
an hour of drizzling could discover the problem for you. The guard rails, demonstrated:
"""
attempts = [
    ("RA outside +/-360 degrees",
     dict(name="bad", ra=400.0, dec=0.0)),
    ("pixfrac outside (0, 1]",
     dict(name="bad", ra=0.0, dec=0.0, final_pixfrac=0.0)),
    ("even PSF shape (no centre pixel)",
     dict(name="bad", ra=0.0, dec=0.0, psf_shape=(20, 20))),
    ("unknown cr_method",
     dict(name="bad", ra=0.0, dec=0.0, cr_method="lacosmic")),
    ("injection dials without inject_image",
     dict(name="bad", ra=0.0, dec=0.0, inject_pixel_scale=0.05)),
    ("inject_image without its pixel scale",
     dict(name="bad", ra=0.0, dec=0.0, inject_image="arc.fits")),
]

for label, kwargs in attempts:
    try:
        TargetSpec(**kwargs)
    except ValueError as e:
        print(f"  [rejected] {label}: {e}")
    else:
        print(f"  [UNEXPECTED] {label} was accepted — this should not happen")

"""
The full rule set: RA within +/-360 and Dec within +/-90 degrees; `final_pixfrac` in (0, 1];
`cr_method` one of "driz_cr"/"deepcr" and `psf_star_pass` one of "auto"/"science"/"no_cr"; the
cutout and both PSF shapes two positive integers with the PSF shapes odd; the `alma_sim_*` times
and frequency positive; and injection all-or-nothing — `inject_image` requires
`inject_pixel_scale`, and no injection dial may be set without `inject_image`.

__Pre-Flight Guards__

A second layer of guards lives at the top of `reduce_target` and checks *combinations* of dials
against the chosen instrument — still before any download, so an unsupported combination costs
you an exception in milliseconds, not a broken dataset in an hour:

| Declaration | Supported on | Why it is bounded |
|---|---|---|
| `frame_products=True` | HST, JWST, Keck | Per-frame packaging exists for these paths only. |
| `psf_from_frames=True` | HST, JWST | The Keck AO mosaic PSF is the tier-A epoch design instead. |
| `cr_method="deepcr"` | HST AstroDrizzle path, ACS/WFC + WFC3/UVIS | Needs a registered deepCR model; WFC3/IR cosmic rays are already ramp-flagged. |
| `psf_star_pass="no_cr"` | HST AstroDrizzle path, without `psf_from_frames` | It is a second full AstroDrizzle pass for the stars. |
| `inject_image=...` | HST, JWST and Keck imaging paths, or visibility simulation | Injection is built per combine backend (simobserve on the ALMA branch). |

These raise as loud design boundaries rather than silently skipped options — the instrument
scripts point out which apply as they use each feature.

__Wrap Up__

The `TargetSpec` is the whole interface: identity, instrument, and every contested dial in one
frozen, validated, committable declaration. Everything downstream — the provenance, the
reproducibility story, the dial studies — follows from that one design decision.

The following locations of the workspace are good places to checkout next:

- `dataset/README.md`: the one-YAML-per-target convention for committing your sample's specs.
- `scripts/start_here.py`: the spec above driving a real end-to-end reduction.
- `scripts/hst_acs/dials.py`: `dataclasses.replace` variants powering a real drizzle trade study.
- `scripts/guides/output_contract.py`: where the spec reappears verbatim inside `reduction.json`.

This guide runs fully offline — spec construction touches neither the network nor the heavy
instrument stacks.
"""
