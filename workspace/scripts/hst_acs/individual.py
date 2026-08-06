"""
HST ACS: Individual Frame Products
==================================

Every script so far has produced one drizzled mosaic per target. Drizzling is a resampling, and
resampling has a price a lens modeler pays forever after: **correlated noise**. Each input pixel
is shared among neighbouring output pixels (Fruchter & Hook 2002, PASP 114, 144), so adjacent
mosaic pixels do not carry independent noise — while the chi^2 of every pixel-based lens
likelihood assumes exactly that independence. **PyAutoReduce** compensates with the scalar
Casertano correction R, but a scalar is an approximation to a full covariance, and precision
applications (pixelized source reconstructions, Bayesian evidence comparisons, substructure
searches) feel the difference.

There is a clean way out: don't resample. Model the individual calibrated exposures
*simultaneously*, each on its own native pixel grid, with its own PSF and its own genuinely
uncorrelated noise-map. The SL2S survey took a version of this stance years ago — reducing
WFPC2 data while explicitly preserving native pixels to avoid correlated noise (Gavazzi et al.
2012, https://arxiv.org/abs/1202.3852) — and **PyAutoReduce**'s per-exposure frame products are
that idea as a first-class packaging mode: set `frame_products=True` and every calibrated `_flc`
chip that covers the target ships as its own modeling-ready `al.Imaging` dataset, alongside (not
instead of) the mosaic.

This script runs that mode on the SLACS anchor, then walks everything it produces: the `frames/`
tree, the manifest, the per-frame noise (ERR-based, no R — the whole point), the deepCR cosmic
ray masks, the per-frame PSFs and their drop-convolution combination, and the registration
residuals that tell you how well the frames agree about where the sky is.

**Dependency note:** per-frame cosmic-ray masking uses deepCR, an optional extra — install with
`pip install "autoreduce[frames]"` before running. The run itself reuses the exposure cache from
`start_here.py`.

__Contents__

- **Imports:** Import **PyAutoReduce** and the other libraries we need.
- **Paths:** Anchor the cache and output locations to the workspace root.
- **Target Spec:** The same anchor with `frame_products=True` and `psf_from_frames=True`.
- **Run:** The reduction — mosaic products plus the frames/ tree.
- **The Frames Tree:** What lands on disk, per exposure and chip.
- **Manifest Walk:** The frames/manifest.json schema, entry by entry.
- **Per-Frame Noise — No R:** ERR-based noise-maps with genuinely uncorrelated pixels.
- **Sky Pedestal:** Why MDRIZSKY must be subtracted from frames but not the mosaic.
- **deepCR Masks:** Per-frame cosmic-ray masks — mask-only, never inpainted.
- **Per-Frame PSFs:** Native ePSFs per chip, and the honest star-poor outcome.
- **The Frame-Combined Mosaic PSF:** `psf_from_frames` and the drop-convolution combine.
- **Registration Residuals:** Measured frame-to-frame alignment, and its reliability flag.
- **Load A Frame In PyAutoLens:** Each frame pair loads at native scale.
- **Wrap Up:** Where to go next in the workspace.

__Imports__
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from autoreduce import TargetSpec, reduce_target

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the folder
containing `scripts/`). **PyAutoReduce** requires absolute paths: its drizzle step changes the
working directory internally, so relative paths would break.

The output goes to its own subfolder so the `start_here.py` products stay untouched; the
exposure cache is shared, so no re-download happens if you ran that script first.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"                     # Shared with every other script in this folder.
OUTPUT_ROOT = WORKSPACE / "output" / "individual"   # Own output root: mosaic + frames/ land here.

"""
__Target Spec__

Two dials change from the default reference reduction:

- `frame_products=True` — package every calibrated chip covering the target as a
  native-pixel dataset under `frames/`, in addition to the mosaic. (HST, JWST and Keck only;
  the pipeline fails fast on any other instrument.)

- `psf_from_frames=True` — build the *mosaic's* PSF by combining the per-frame ePSFs through
  the drizzle geometry, instead of measuring stars on the resampled mosaic. More on this below.

The mosaic path itself is untouched by `frame_products` — the pipeline's validation
byte-compares the mosaic products between a flag-off and flag-on run.
"""
SPEC = TargetSpec(
    name="slacs0008-0004",  # Same target and cache entry as start_here.py.
    ra=2.012333,  # Right ascension in degrees.
    dec=-0.068944,  # Declination in degrees.
    proposal_ids=("10886",),  # The SLACS ACS program.
    frame_products=True,  # Package per-exposure native-pixel datasets under frames/.
    psf_from_frames=True,  # Mosaic PSF from combined per-frame ePSFs (drop-convolution route).
)

"""
__Run__

The frame packaging runs after the ordinary pipeline (so the driz_cr DQ flags exist in the
exposures) and before any cache eviction could delete the frames it reads.
"""
print(
    "\n"
    "Running the reduction with frame products enabled.\n"
    "Warm cache (after start_here.py): a few minutes, plus deepCR inference per chip.\n"
    "Cold cache: add the ~0.5 GB MAST download (~10-30 minutes).\n"
)

record = reduce_target(SPEC, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)

out_dir = OUTPUT_ROOT / SPEC.name
frames_dir = out_dir / "frames"

print("Reduction complete.")
print("\n--- frames fragment (reduction.json) ---")
print(json.dumps(record["frames"], indent=2))

"""
__The Frames Tree__

On disk, next to the usual mosaic products:

    output/individual/slacs0008-0004/
      data.fits, noise_map.fits, psf.fits, psf_full.fits, reduction.json   <- the mosaic, as ever
      frames/
        manifest.json
        <rootname>_chip1/ {data.fits, noise_map.fits, dq.fits, cr_mask.fits [, psf.fits, psf_full.fits]}
        <rootname>_chip2/ {...}

One directory per (exposure, SCI chip) that covers the target. ACS/WFC has two CCD chips per
exposure; chips whose footprint misses the target are skipped and recorded — so do not expect
2 x n_exposures directories.

Each chip directory is a self-contained dataset: `data.fits` + `noise_map.fits` load directly as
an imaging dataset, `dq.fits` (int32) keeps the full calibration DQ bit information, and
`cr_mask.fits` (uint8) the cosmic-ray mask, for any consumer wanting a different masking policy
than the shipped one.
"""
chip_dirs = sorted(d for d in frames_dir.iterdir() if d.is_dir())
print("\n--- frames tree ---")
for d in chip_dirs:
    contents = sorted(p.name for p in d.iterdir())
    print(f"  {d.name}: {contents}")

"""
__Manifest Walk__

`frames/manifest.json` (schema version 2) is the frame products' contract. Top level:

- `frame_cutout_shape` — the native-pixel stamp shape, derived from the mosaic dials
  (`cutout_shape * final_scale / native_scale`, odd-forced): the same sky footprint as the
  mosaic cutout, no new user dial.
- `native_scale` — 0.05 arcsec/pix for ACS/WFC; the pixel scale every frame loads at.
- `data_units` — e-/s: each SCI/ERR chip is converted from electrons so frames and mosaic share
  the cps flux scale.
- `cr_method` — the per-frame CR machinery (deepCR on ACS; see below).
- `dq_semantics` — the masking policy the shipped noise-maps encode.
- `max_registration_residual_px` — the at-a-glance registration verdict (final section).
- `frames` — one entry per chip directory; `skipped_chips` the off-target ones.
"""
manifest = json.loads((frames_dir / "manifest.json").read_text())

print("\n--- manifest (top level) ---")
print(f"version                     : {manifest['version']}")
print(f"frame_cutout_shape          : {manifest['frame_cutout_shape']}")
print(f"native_scale                : {manifest['native_scale']}")
print(f"data_units                  : {manifest['data_units']}")
print(f"cr_method                   : {manifest['cr_method']}")
print(f"driz_cr_run                 : {manifest['driz_cr_run']}")
print(f"max_registration_residual_px: {manifest['max_registration_residual_px']}")
print(f"frames                      : {len(manifest['frames'])}")
print(f"skipped_chips               : {len(manifest['skipped_chips'])}")

entry = manifest["frames"][0]
print("\n--- first frame entry ---")
print(json.dumps(entry, indent=2))

"""
__Per-Frame Noise — No R__

Here is the selling point, in the numbers. The per-frame noise-map is the `calacs`-propagated
ERR extension — native-pixel Poisson + read noise + dark, computed by the calibration pipeline
itself (ACS Data Handbook, https://hst-docs.stsci.edu/acsdhb) — unit-converted alongside the
science array. **No Casertano factor is applied, because nothing has been resampled.** Every
pixel's noise is genuinely independent, so a chi^2 over these frames is exactly what its
statistics claim, with no scalar-correction approximation anywhere.

Compare the provenance: the mosaic's noise block records `correlated_noise_factor` = R > 1; the
frame products record none, structurally. The masking policy is "masked-by-noise": any nonzero
DQ bit, deepCR cosmic-ray pixel, off-chip pixel or non-finite ERR pixel gets noise = 1e8 with the
data zeroed, so masked pixels drop out of any likelihood without a separate mask file.
"""
frame_data = fits.getdata(frames_dir / entry["dir"] / "data.fits").astype(float)
frame_noise = fits.getdata(frames_dir / entry["dir"] / "noise_map.fits").astype(float)

good = frame_noise < 1.0e7
print("\n--- per-frame noise (first frame) ---")
print(f"mosaic R (for contrast)  : {record['drizzle']['correlated_noise_factor']:.3f}")
print("frame R                  : none — nothing resampled")
print(f"median frame noise (e-/s): {np.median(frame_noise[good]):.4f}")
print(f"masked pixels            : {entry['n_masked_pixels']} "
      f"({entry['n_masked_pixels'] / frame_noise.size:.2%} of the stamp)")

"""
__Sky Pedestal__

A subtlety the manifest records per frame: AstroDrizzle's sky subtraction is *virtual* — the
`globalmin+match` sky is stored in each chip's `MDRIZSKY` header keyword and subtracted only
during the drizzle, leaving the `_flc` files untouched. A naively-cut frame would therefore carry
a sky pedestal the mosaic lacks, and joint frame+mosaic modeling would disagree about the
background by exactly that amount.

The frame packaging subtracts each chip's `MDRIZSKY` explicitly, and records the value and
keyword (`sky_subtracted`, `sky_keyword`) in the manifest entry — auditable, like everything
else.
"""
print("\n--- sky pedestal (first frame) ---")
print(f"sky_subtracted : {entry['sky_subtracted']} (keyword {entry['sky_keyword']})")

"""
__deepCR Masks__

Frames need a cosmic-ray answer the mosaic path cannot give them. `driz_cr` flags CRs against a
median *stack*, so its DQ flags exist only where several exposures overlap — and per-frame
modeling needs a mask for every frame *on its own*, single-exposure visits included.

The per-frame machinery is deepCR (Zhang & Bloom 2020, ApJ 889, 24,
https://ui.adsabs.harvard.edu/abs/2020ApJ...889...24Z): a CNN that flags cosmic-ray pixels in
individual exposures, trained largely on exactly this ACS/F814W regime, with higher completeness
than the classical Laplacian-edge method (LACosmic, van Dokkum 2001, PASP 113, 1420) at fixed
false-positive rate. On ACS/WFC the published `ACS-WFC` model is used, and the manifest records
the exact model and threshold so datasets remain re-maskable later.

**Mask-only by contract**: deepCR can also *inpaint* the flagged pixels, and **PyAutoReduce**
never uses that — bad pixels are masked (noise = 1e8), never fabricated. `cr_mask.fits` keeps
the raw mask.
"""
total_cr = sum(e["n_cr_pixels"] for e in manifest["frames"])
print("\n--- deepCR ---")
print(f"CR pixels flagged across frames: {total_cr}")
for e in manifest["frames"]:
    print(f"  {e['dir']}: {e['n_cr_pixels']} CR pixels")

"""
Let's look at one frame: data, noise-map and CR mask side by side. Cosmic rays are obvious in
the native frame — sharp tracks that the mosaic (where the stack rejection removed them) never
shows you.
"""
cr_mask = fits.getdata(frames_dir / entry["dir"] / "cr_mask.fits")

plot_dir = out_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

sky_sigma = np.median(frame_noise[good])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(np.arcsinh(frame_data / (3.0 * sky_sigma)), origin="lower", cmap="magma")
axes[0].set_title(f"{entry['dir']} data (native pixels)")
axes[1].imshow(np.clip(frame_noise, 0.0, 5.0 * sky_sigma), origin="lower", cmap="viridis")
axes[1].set_title("noise_map (ERR-based, no R)")
axes[2].imshow(cr_mask, origin="lower", cmap="gray")
axes[2].set_title("cr_mask (deepCR)")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plot_path = plot_dir / "individual_frame.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)

print(f"\nPlot saved to: {plot_path}")

"""
__Per-Frame PSFs__

Each chip directory that can support it also ships `psf.fits` / `psf_full.fits`: an effective PSF
built from the frame's *own* full chip, on native (undrizzled, distorted) pixels — the correct
PSF for modeling that frame. DQ-flagged pixels are screened out of the star finding, the target
itself is excluded, and the saturation cap is formed in the frame's native units.

The honest outcome is recorded, not hidden: **insufficient stars is a recorded result, not a
hard stop**. A single ~500 s frame legitimately may lack the minimum usable stars; its data
products remain useful, the manifest `psf` block says `method: "none"` with the reason, and the
reduction prints a loud notice that such a frame is not modelable until a model-PSF tier exists
(it currently does not — see `psf.py` for that honesty story on the mosaic side too).
"""
print("\n--- per-frame PSFs ---")
for e in manifest["frames"]:
    psf_block = e["psf"]
    method = psf_block["method"]
    extra = "" if method == "none" else f", n_stars={psf_block.get('n_stars_used')}"
    print(f"  {e['dir']}: method={method}{extra}")

"""
__The Frame-Combined Mosaic PSF__

`psf_from_frames=True` changed how the *mosaic's* PSF was built, too. Instead of measuring stars
on the resampled mosaic, each frame's native ePSF is:

1. convolved with the drizzle drop (a `final_pixfrac`-wide box, applied as an exact
   fractional-width Fourier convolution),
2. resampled onto the mosaic grid through the local frame-to-mosaic WCS Jacobian at the target
   position, and
3. exposure-time-weighted averaged.

This honours the drizzled-PSF invariant *by construction* — it applies the drizzle geometry to
the PSF itself — while sidestepping mosaic resampling artifacts and star scarcity (every frame's
full star field contributes, rather than only the stars that survive on one resampled image).
The recorded approximation: local-affine geometry plus drop convolution; sub-pixel
output-sampling phases are not modelled. It is loud when no frame yields an ePSF — it never
silently falls back to mosaic stars.
"""
print("\n--- mosaic PSF (frame-combined) ---")
print(json.dumps(record["psf"], indent=2))

"""
__Registration Residuals__

Joint multi-frame modeling needs to know how well the frames agree about where the sky is. Each
manifest entry carries a `registration` block with two very different kinds of number — the
manifest's own `registration_note` spells out the distinction, because the header keywords invite
misreading:

- `wcsname` / `wcstype` / `rms_ra_mas` / `rms_dec_mas` / `nmatches` — the astrometric solution
  behind the frame's WCS, stating the group's **absolute** alignment to an external catalog
  (for this target, ~44 mas to GSC 2.4.2). Absolute accuracy is *not* what frame-joint modeling
  consumes.

- `residual_dy_px` / `residual_dx_px` — the **measured relative** registration error against the
  reference frame: resample through both shipped WCS, phase-correlate, read the shift. This is
  the number that matters, and on this target it sits at or below ~0.1 native pixels.

- `residual_reliable` — the measurement's own honesty flag: where CR-masked pixels bite the
  source, the estimator degrades to ~0.1-0.3 px, and heavily-masked pairs are flagged
  unreliable rather than reported as precise.

The default modeling stance: treat the shipped WCS as known (the residuals sit below the scales
standard modeling constrains); for precision work, free per-frame (dy, dx) nuisance parameters
with Gaussian priors of the recorded residual width.
"""
print("\n--- registration residuals ---")
for e in manifest["frames"]:
    reg = e["registration"]
    print(
        f"  {e['dir']}: dy={reg['residual_dy_px']:+.3f} px, dx={reg['residual_dx_px']:+.3f} px, "
        f"reliable={reg['residual_reliable']}, reference={reg['reference']}, "
        f"absolute={reg['wcsname']}"
    )
print(f"headline: max residual {manifest['max_registration_residual_px']} px")

"""
__Load A Frame In PyAutoLens__

Each frame pair loads as an imaging dataset at the native pixel scale from the manifest — and,
because of the masked-by-noise convention, needs no mask file for its DQ/CR/off-chip pixels.
"""
try:
    import autolens as al
except ImportError:
    al = None
    print(
        "\nPyAutoLens is not installed, so the loading demonstration is skipped.\n"
        "Install it with `pip install autolens`. The frame products themselves are complete."
    )

if al is not None:
    frame_dir = frames_dir / entry["dir"]

    dataset = al.Imaging.from_fits(
        data_path=frame_dir / "data.fits",
        noise_map_path=frame_dir / "noise_map.fits",
        psf_path=frame_dir / "psf.fits" if (frame_dir / "psf.fits").exists() else None,
        pixel_scales=manifest["native_scale"],  # Native ACS pixels — 0.05"/pix, from the manifest.
    )
    print(f"\nLoaded frame {entry['dir']} into PyAutoLens: shape {dataset.shape_native} "
          f"at {dataset.pixel_scales} arcsec/pix.")
    print(
        "Fit all frames simultaneously by creating one analysis per frame and summing the\n"
        "likelihoods — see the autolens_workspace multi-dataset examples for the pattern."
    )

"""
__Wrap Up__

You now have both representations of the same photons: one drizzled mosaic (convenient, one
PSF, one dataset — but correlated noise, scalar-corrected) and N native frames (uncorrelated
noise, exact per-frame PSFs, registration shipped as information — but N datasets to model
jointly). Which to use is a modeling decision; the reduction ships both honestly.

The following locations of the workspace are good places to checkout next:

- `scripts/hst_acs/psf.py`: the PSF story, including the per-frame tiers seen here.
- `scripts/hst_acs/dials.py`: the correlated-noise numbers (R vs pixfrac) that motivate frame
  products in the first place.
- `scripts/hst_acs/simulator.py`: injection testing — the frames are where injection happens.
- `scripts/guides/noise_maps.py`: noise recipes and why chi^2 cares about correlation.

__Env__ (Developer Only)

ENV: network
"""
