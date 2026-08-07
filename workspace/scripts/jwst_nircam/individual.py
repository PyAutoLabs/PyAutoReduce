"""
JWST NIRCam: Individual Frames
==============================

Everything so far has modeled a *mosaic*: one resampled image combining every exposure.
This script produces the alternative — per-exposure **frame products**: a native-pixel
cutout of the target from every calibrated exposure, each with its own noise map, DQ
map and PSF, plus a manifest that records how they relate. You then model the frames
*jointly*, instead of (or as well as) the mosaic.

Why would you? Two properties the mosaic can never have:

- **Uncorrelated noise.** Nothing has been resampled, so no Casertano R, no correlated
  neighbours — every frame's per-pixel noise is honest as-is, and the joint likelihood
  over frames is exact where the mosaic's is an approximation.
- **The sub-pixel information survives.** On the undersampled SW channel, the dither
  pattern's sub-pixel phases carry information a resampled mosaic partially destroys
  (aliasing + interpolation). Forward-modeling the frames uses those phases directly.

The precision-measurement literature is moving the same way: the Roman HLIS shear study
of Yamamoto et al. (2022, https://arxiv.org/abs/2203.08845) benchmarked joint
multi-epoch measurement against coadds and found multi-epoch performed better —
avoiding exactly the coadd-PSF discontinuities and correlated noise the mosaic route
accepts. Honest counterweight: published JWST *extended-source* practice is mosaic-based
today — per-frame lens modeling with NIRCam is ahead of the field, not following it.

__Contents__

- **Imports:** Import **PyAutoReduce** and the supporting libraries.
- **Paths:** Anchor the cache and output folders to the workspace root.
- **The Frame Spec:** `frame_products=True` on the COSMOS-Web ring.
- **The _crf Frames:** Why frames come from outlier_detection's flagged products, not raw `_cal` files.
- **The Run:** Reduce with frame products enabled.
- **The DQ Policy — DO_NOT_USE Only:** Why JWST frame masking differs fundamentally from HST's.
- **Manifest Walk — Schema v2:** Units, sky subtraction, source family, per-frame entries.
- **Registration Residuals and the Reliability Flag:** The edge-of-detector story and the honest null.
- **Per-Frame PSFs and the STPSF Fallback:** Tier-1 ePSF per frame, Tier 2b when stars run out.
- **Frame-Level Caveats:** The artifacts that arrive unmitigated.
- **Loading a Frame in PyAutoLens:** Each frame pair is a native-scale `al.Imaging`.
- **Wrap Up:** Where to go next.

__Imports__
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

from autoreduce import TargetSpec, reduce_target
from autoreduce.instruments import nircam_adapter_for_filter

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the
folder containing `scripts/`). **PyAutoReduce** requires absolute paths: its combine step
changes the working directory internally, so relative paths would break.

We give the frame reduction its own output tree (`output/frames/`) while keeping the
spec `name` identical to `start_here.py` — the exposure cache is keyed by name, so the
downloaded `_cal` files are shared and only the pipeline stages re-run.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"  # downloaded exposures + CRDS references (shared with start_here.py)
OUTPUT_ROOT = WORKSPACE / "output"  # reduced datasets, one folder per target
FRAMES_OUTPUT = OUTPUT_ROOT / "frames"  # this script's outputs, separate from the mosaic run

"""
__The Frame Spec__

One dial turns frames on: `frame_products=True`. The mosaic is still produced — frames
are packaged *as well*, into a `frames/` sub-folder beside the four mosaic products.
Each frame cutout is at the **native** detector scale (0.063"/pixel for LW here;
0.031"/pixel on SW), because the entire point is that nothing gets resampled.

We use the F277W ring data already cached by `start_here.py`. The SW bands are where
the sampling argument bites hardest, but the LW run demonstrates every mechanism at a
quarter of the pixel count — swap the band to F115W/F150W and the same code runs at
0.031"/pixel.
"""
band = "F277W"
adapter = nircam_adapter_for_filter(band)

spec = TargetSpec(
    name=f"cosmos_web_ring_{band.lower()}",  # same name as start_here.py -> shares its exposure cache
    ra=150.10048,  # the COSMOS-Web ring (Mercier et al. 2024)
    dec=1.89301,
    instrument=adapter.key,  # "nircam_lw"
    filter_name=band,
    proposal_ids=("1727",),  # COSMOS-Web only
    final_scale=adapter.recommended_final_scale,  # mosaic dials still apply — the mosaic is produced too
    final_pixfrac=1.0,
    cutout_shape=(209, 209),  # mosaic cutout; frame cutouts are sized at native scale automatically
    frame_products=True,  # THE dial: package a native-frame cutout per exposure alongside the mosaic
)

"""
__The _crf Frames__

Which files should the frames be cut from? Not the raw `_cal` products — something
better exists. When calwebb_image3 runs its `outlier_detection` step (comparing each
exposure against the median of all resampled dithers), it can write back the
per-exposure results: the `_crf` files — tweakreg-updated, outlier-flagged calibrated
frames. **PyAutoReduce** runs image3 with `save_results` enabled on that step precisely
so the `_crf` products exist, and packages frames from them. They are the exact JWST
analogue of HST's driz_cr-flagged `_flc` files: single exposures that carry the *stack's*
verdict on their deviant pixels (residual CRs, snowball residue, hot pixels,
persistence) in their DQ arrays.

If image3 did not run (a single-exposure reduction has no stack to compare against),
packaging falls back to the `_cal` files and the manifest records that absence honestly
— the `source` field below tells you which family you got.

__The Run__
"""
print(
    """
    Reducing the COSMOS-Web ring (F277W) with frame_products=True.

    With the cache warm from start_here.py this skips the MAST download but re-runs
    calwebb_image3 plus the frame packaging — expect several minutes. Each exposure
    yields one native-scale frame cutout under frames/.
    """
)

record = reduce_target(spec, cache_root=CACHE_ROOT, output_root=FRAMES_OUTPUT)

out_dir = FRAMES_OUTPUT / spec.name
frames_dir = out_dir / "frames"

print("Frames block:", json.dumps(record["frames"], indent=2))

"""
The `frames` provenance block summarises the packaging: how many exposures went in, how
many chips were written (one NIRCam `_cal`/`_crf` file carries one detector, so
exposures and chips map one-to-one here), how many were skipped (no overlap with the
target), how many carry a per-frame PSF, and the headline registration residual.

__The DQ Policy — DO_NOT_USE Only__

Every frame ships a `dq.fits` with the full JWST data-quality bitmask — but the
masked-by-noise policy (noise set to 1e8, data zeroed) applies **only to pixels whose
DQ has the DO_NOT_USE bit set**, plus off-chip and non-finite-error pixels. This is a
deliberate divergence from the HST frame policy, where *any* nonzero DQ bit masks the
pixel, and the reason is the ramp architecture from `step_by_step.py`:

JWST cosmic rays are *removed during ramp fitting* — a jump corrupts groups, the slope
is refit from the clean segments, and the affected pixel emerges with a valid rate and
an informational `JUMP_DET` flag. `JUMP_DET` therefore rides *good* pixels. Masking any
nonzero DQ would throw away swathes of perfectly usable data in every frame. Only
`DO_NOT_USE` — calwebb's considered bad-pixel verdict, which image3's outlier_detection
also sets in the `_crf` products — means bad. The manifest's `dq_semantics` block spells
this out per reduction, including what bits 1 (DO_NOT_USE) and 4 (JUMP_DET) mean, so
the policy travels with the data.

__Manifest Walk — Schema v2__

The frames manifest (`frames/manifest.json`) is the contract for everything in the
folder. The header fields worth knowing:

- `version: 2` — the schema generation (v2 generalised the HST-era manifest to JWST).
- `data_units` — **"MJy/sr"**: frames keep their native surface-brightness units,
  matching the mosaic (defaults-first; the packaging is loud if exposures arrive with
  heterogeneous units).
- `sky_subtracted` / `sky_keyword` (per frame) — image3's skymatch *records* each
  exposure's background level (`BKGLEVEL`) rather than subtracting it; frame packaging
  subtracts that recorded level so every frame shares the mosaic's zero point, and
  writes down the number and the keyword it came from.
- `source` — which input family the frames were cut from: the `_crf` outlier-flagged
  products (normal case) or the `_cal` fallback with its recorded absence.
- `frame_cutout_shape` / `native_scale` — the per-frame geometry: native pixels, no
  resampling.
- `cr_method` — for JWST: ramp-level jump detection plus image3 outlier_detection via
  the `_crf` DQ; there is no deepCR model for JWST and none is needed.
"""
manifest = json.loads((frames_dir / "manifest.json").read_text())

print("Manifest version:", manifest["version"])
print("Data units:", manifest["data_units"])
print("Source family:", manifest["source"])
print("Frame cutout shape:", manifest["frame_cutout_shape"])
print("Native scale:", manifest["native_scale"], '"/pixel')
print("CR method:", manifest["cr_method"])
print("DQ policy:", manifest["dq_semantics"]["policy"])

first = manifest["frames"][0]
print("First frame entry keys:", sorted(first.keys()))
print(
    f"  {first['dir']}: exptime {first['exptime']}s, sky_subtracted "
    f"{first['sky_subtracted']} ({first['sky_keyword']}), "
    f"{first['n_masked_pixels']} masked px, psf method {first['psf'].get('method')}"
)

"""
__Registration Residuals and the Reliability Flag__

Joint frame modeling needs to know how well the frames' WCS solutions agree — if frame
astrometry disagrees at a significant fraction of a pixel, the model must fit per-frame
offsets. So the packaging *measures* the relative registration: it phase-correlates
each frame's cutout against a reference frame through the shipped WCS and records the
residual, per frame, in the manifest's `registration` blocks (alongside the header's
absolute-catalog metadata — two different things the `registration_note` carefully
distinguishes).

The JWST validation on this very field taught the measurement some humility, and the
result is **schema v2's reliability flag**. COSMOS-Web's dither pattern routinely puts
the ring near a detector *edge*, so some frame cutouts are mostly off-chip mask. Phase
correlation between two mostly-masked cutouts locks onto the mask geometry, not the
sky — producing spectacular ~200-pixel fake "residuals" that would terrify anyone who
read them as astrometry. The fix, recorded per frame:

- the reference is the best-covered frame;
- any pair where more than 20% of pixels are masked is flagged
  `residual_reliable: false`;
- the headline `max_registration_residual_px` is computed over *reliable* pairs only —
  and when no clean pair exists it is an honest **null**, meaning "unmeasured", never a
  mask artifact dressed up as a shift.

The measurement floor is ~0.1–0.3 px where masked pixels bite the source, so sub-0.1 px
values are consistent with zero. Practical stance for modeling: treat the shifts as
known when residuals are far below your modeling scale; otherwise free per-frame
(dy, dx) offsets with priors of the recorded width.
"""
print("Headline max registration residual (native px):", manifest["max_registration_residual_px"])
for entry in manifest["frames"]:
    reg = entry["registration"]
    print(
        f"  {entry['dir']}: reliable={reg.get('residual_reliable')}, "
        f"residual=({reg.get('residual_dy_px')}, {reg.get('residual_dx_px')})"
    )

"""
__Per-Frame PSFs and the STPSF Fallback__

Each frame gets its own PSF attempt: the Tier-1 ePSF machinery runs on the frame's own
stars (at native sampling, DQ-patching only DO_NOT_USE pixels — the same policy as the
data). A lens cutout is a small field, though, and a single frame often lacks the
stars a mosaic accumulates; a star-poor frame is a *recorded outcome*, not a fatal
error.

This is where JWST has a card HST never did: **Tier 2b**. When the frame's star field
cannot support an ePSF, `stpsf` (the STScI optical model, formerly WebbPSF) is
evaluated at that frame's detector and target position, and the detector-sampled,
geometric-distortion-included kernel (`DET_DIST`) ships instead — keeping every frame
modelable. The literature caveat travels in the diagnostics verbatim (empirical PSFs
are consistently preferred over models for decomposition work — see `psf.py`), so a
model-PSF frame is always identifiable. If `stpsf` is not installed, that too is a
recorded outcome, and the frame ships without a PSF.

Note one subtlety for SW work: a single frame's ePSF is *itself undersampled* — the
sampling recovery happens across frames (the sub-pixel dither phases), which is exactly
the information joint frame modeling exploits.
"""
for entry in manifest["frames"]:
    psf_info = entry["psf"]
    print(f"  {entry['dir']}: psf method = {psf_info.get('method')}")
    if "caveat" in psf_info:
        print(f"      caveat: {psf_info['caveat'][:80]}...")

"""
__Frame-Level Caveats__

Frames trade the mosaic's approximations for the mosaic's *protections*. Artifacts that
mosaic-level processing mitigates arrive at the frame level unmitigated: 1/f banding
(per-amplifier stripes along the slow-read axis), wisps and any snowball residue that
escaped the ramp flags land in your frames raw, because the corrections that remove
them (where teams apply them at all) act on or across the stack. The manifest carries
this honesty; your modeling error budget should too. For routine extended-source work
the mosaic remains the default — frames are the precision option, most compelling for
undersampled SW data and substructure/shear-grade measurements.

__Loading a Frame in PyAutoLens__

Each frame folder is itself a valid `al.Imaging` dataset at the native pixel scale —
`data.fits` + `noise_map.fits` (+ `psf.fits` where viable). Joint fitting then means
one `Analysis` per frame, summed — with the mass model shared and, if the registration
residuals warrant, per-frame offsets free.
"""
try:
    import autolens as al
except ImportError:
    al = None
    print(
        "PyAutoLens is not installed (pip install autolens), so the frame-loading demo "
        "is skipped — the frame products above are complete."
    )

if al is not None:
    loadable = [e for e in manifest["frames"] if e["psf"].get("method") not in (None, "none")]
    if loadable:
        entry = loadable[0]
        frame_dir = frames_dir / entry["dir"]
        frame_dataset = al.Imaging.from_fits(
            data_path=frame_dir / "data.fits",
            noise_map_path=frame_dir / "noise_map.fits",
            psf_path=frame_dir / "psf.fits",
            pixel_scales=manifest["native_scale"],  # native pixels — no resampling anywhere
        )
        print(
            f"Loaded frame {entry['dir']} as al.Imaging: shape "
            f"{frame_dataset.data.shape_native} at {manifest['native_scale']}\"/pixel."
        )

"""
Finally, a look at the frames themselves — the same ring, once per exposure, each at
native sampling with its own noise and DQ.
"""
plot_dir = out_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

entries = manifest["frames"][:4]
fig, axes = plt.subplots(1, max(len(entries), 1), figsize=(4.5 * max(len(entries), 1), 4.5))
axes = np.atleast_1d(axes)

for ax, entry in zip(axes, entries):
    frame_data = fits.getdata(frames_dir / entry["dir"] / "data.fits").astype(float)
    scale = np.nanpercentile(frame_data, 99.5)
    ax.imshow(np.arcsinh(frame_data / (0.05 * max(scale, 1e-12))), origin="lower", cmap="magma")
    ax.set_title(entry["dir"], fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plot_path = plot_dir / "individual_frames.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)

print(f"Frame gallery saved to: {plot_path.resolve()}")

"""
__Wrap Up__

You now have the ring as the mosaic *and* as its constituent native frames: `_crf`
cutouts carrying the stack's outlier verdicts, masked by the DO_NOT_USE-only policy,
with per-frame skies, PSFs (empirical where possible, flagged STPSF where not),
measured relative registration with an honest reliability flag — and uncorrelated
noise with no Casertano R anywhere, because nothing was resampled.

The following locations of the workspace are good places to checkout next:

- `scripts/jwst_nircam/psf.py`: the PSF tiers behind the per-frame kernels.
- `scripts/hst_acs/individual.py`: the HST frame-products sibling — any-bit DQ policy, deepCR, MDRIZSKY.
- `scripts/jwst_nircam/step_by_step.py`: where the `_crf` files come from in the image3 chain.
- `scripts/guides/noise_maps.py`: why uncorrelated frame noise makes the joint likelihood exact.

__Env__ (Developer Only)

ENV: network
"""
