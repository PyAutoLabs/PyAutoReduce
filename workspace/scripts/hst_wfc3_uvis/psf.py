"""
PSF: HST WFC3/UVIS
==================

This script builds the WFC3/UVIS point spread function on a genuinely star-rich field — a
pointing in the globular cluster Omega Centauri — and compares **PyAutoReduce**'s two
empirical PSF backends head to head: the default photutils effective-PSF builder and the
STARRED joint-reconstruction backend.

Why a star cluster, in a workspace about lensing? Because PSF validation needs *stars*, and
lens fields rarely provide them. In extragalactic fields most "point sources" that survive a
star-finder's cuts are compact galaxies, which silently broaden any PSF built from them. On a
real stellar field the empirical, sub-pixel-registered stack of the stars *is* the PSF, so
concentration and radial-profile comparisons against it are meaningful truth references. This
is the field on which the UVIS backend comparison was validated, and this script reproduces
that setup.

For the full PSF architecture — the tier system, `psf_star_pass`, per-frame PSFs, the
drizzled-PSF invariant — see `hst_acs/psf.py`; this script covers what UVIS adds and what a
crowded field changes.

__Contents__

- **Why PSF Accuracy Matters:** PSF errors masquerade as the lensing signal.
- **The Focus Problem:** HST's PSF breathes, so empirical beats optical modeling.
- **Imports:** Import the required Python libraries.
- **Paths:** Anchor every path to the workspace root, absolutely.
- **Target Spec:** Omega Cen F606W — a clean 6x60s dithered visit.
- **Star Selection in a Crowded Field:** What the selection cuts do when stars are everywhere.
- **The Default Backend:** Reduce with the photutils ePSF (tier 1).
- **The STARRED Backend:** Reduce again with `psf_backend="starred"` (tier 1b).
- **The Head To Head:** The validated comparison numbers and what they mean.
- **Diagnostics and Plots:** Read the psf provenance block and visualize both PSFs.
- **Wrap Up:** Summary of the script and next steps.

__Why PSF Accuracy Matters__

A lens model fits the data *convolved with the PSF*. If the PSF is wrong, the model cannot
reproduce the arcs no matter how good the mass model is, and the misfit appears as structured
residuals along exactly the high-S/N lensed features. Those residuals are indistinguishable
from real astrophysical perturbations — dark substructure, source complexity — so a PSF error
does not just degrade the fit, it *biases the science* by faking or masking the faintest
signals lens modeling looks for. This is why **PyAutoReduce** treats the PSF as a first-class
product with recorded diagnostics, not an afterthought.

__The Focus Problem__

HST's focus changes continuously as the telescope's structure expands and contracts through
each orbit ("breathing"), so the PSF varies from exposure to exposure. The modern response is
*empirical*: measure the PSF from stars in the science data itself, in the tradition of the
effective PSF (ePSF) of Anderson & King (2000, PASP 112, 1360;
https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A) — the instrumental PSF convolved with
the pixel response, built from dithered star images. STScI maintains focus-diverse ePSF
libraries for WFC3 — Anderson (2016, WFC3 ISR 2016-12) for the IR channel and WFC3 ISR
2018-14 for focus-diverse UVIS models, collected at
https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/psf — which contextualize what
"the" UVIS PSF even means: a family, indexed by focus. **PyAutoReduce** builds its PSF from
the field's own stars, which bakes the observation's actual focus in automatically.

The two backends this script compares implement that empirical philosophy differently:

- **photutils ePSFBuilder** (tier 1, the default;
  https://photutils.readthedocs.io/en/stable/user_guide/epsf.html) — the open-source
  Anderson & King iterative ePSF construction.
- **STARRED** (tier 1b, opt-in) — JAX-based joint PSF reconstruction from several stars with
  starlet-regularized narrow-PSF fitting, from the COSMOGRAIL lensing community
  (Michalewicz et al. 2023, JOSS 8(85), 5340; Millon et al. 2024, AJ,
  https://arxiv.org/abs/2402.08725).
"""

"""
__Imports__

`dataclasses.replace` is the idiom for spec variants: `TargetSpec` is frozen, so a second
reduction with one dial changed is a `replace(spec, ...)`, never a mutation.
"""
import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoreduce import TargetSpec, reduce_target

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the folder
containing `scripts/`). **PyAutoReduce** requires absolute paths: its drizzle step changes the
working directory internally, so relative paths would break.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"      # downloaded exposures + CRDS references (re-used across runs)
OUTPUT_ROOT = WORKSPACE / "output"    # reduced datasets, one folder per target

"""
__Target Spec__

The field: Omega Centauri (NGC 5139) through WFC3/UVIS F606W, proposal 15733 — six dithered
60s exposures at a single pointing about 10" off the cluster core. Short exposures on a
globular cluster give exactly what PSF work wants: hundreds of bright-but-unsaturated,
well-sampled, genuinely stellar point sources in one frame.

The drizzle dials are the UVIS-standard native scale with pixfrac 1.0 (see
`hst_wfc3_uvis/start_here.py` for why), and the cutout is enlarged to 401x401 pixels (~16")
to enclose a rich isolated-star sample.
"""
RA, DEC = 201.69283, -47.47906  # Mean pointing of the proposal-15733 F606W exposures.

spec = TargetSpec(
    name="omegacen_f606w",     # Output folder name under output/.
    ra=RA,                     # Field centre right ascension in degrees.
    dec=DEC,                   # Field centre declination in degrees.
    instrument="wfc3_uvis",    # UVIS adapter: _flc, iref, native 0.0396"/pix, sat 63 ke-.
    filter_name="F606W",       # Broad V-band: high stellar S/N in 60s.
    proposal_ids=("15733",),   # Pin to one clean 6x60s dithered visit.
    final_scale=0.0396,        # Native-scale output (UVIS adapter recommendation).
    final_pixfrac=1.0,         # 6-dither visit -> full drop for guaranteed coverage.
    cutout_shape=(401, 401),   # ~16": a rich isolated-star sample for the PSF build.
)

"""
__Star Selection in a Crowded Field__

Before either backend runs, **PyAutoReduce** selects PSF stars from the mosaic with a fixed
set of cuts (they are deliberately *not* `TargetSpec` dials — the selection is part of the
pipeline's contract, so its behavior is uniform across datasets): a 10-sigma detection
threshold, point-like sharpness and roundness cuts, a minimum separation of 25 pixels between
candidates, an edge margin, an exclusion radius around the target position, and a saturation
ceiling — candidates whose peak exceeds 70% of full well are rejected.

On this field those cuts earn their keep in an unusual way. In a globular cluster the
challenge is not finding stars but *rejecting* most of them: crowding violates the minimum
separation for the majority of detections, and the brightest cluster stars saturate — with
the UVIS full well at ~63 ke- and 60s exposures, any star peaking above roughly
0.7 x 63000 / 60 e-/s is excluded. What survives is a subset of isolated, unsaturated,
well-exposed stars — fewer than you might expect from a cluster field, and exactly the ones
you want.

__The Default Backend__

First, the tier-1 reduction with the default `psf_backend="epsf"` (photutils). One recorded
finding from validating this exact field is worth knowing: the photutils builder can fail on
the extended 61x61 `psf_full` product here — background over-subtraction leaves the faint
ePSF wings slightly negative, and the builder rejects the non-positive flux — while the
compact 21x21 core builds cleanly. STARRED, with its regularized wings, delivered both. A
backend difference you only discover on real data, which is why the validation anchors are
real fields.
"""
print(
    "Running the Omega Cen F606W reduction (photutils ePSF backend). First "
    "run downloads 6 exposures from MAST + CRDS references (expect tens of "
    "minutes); re-runs use the cache."
)

record_epsf = reduce_target(spec, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT)

out_epsf = OUTPUT_ROOT / spec.name

print(f"photutils psf block: {json.dumps(record_epsf['psf'], indent=2)}")

"""
__The STARRED Backend__

Now the same field through tier 1b. STARRED ships as the optional `[starred]` extra
(`pip install "autoreduce[starred]"`) because it pulls in JAX and is GPL-licensed; it is
imported lazily, and if it is missing the pipeline raises rather than silently falling back —
you asked for STARRED, you get STARRED or a loud error. The `try/except` below turns that
into a friendly message so this script degrades gracefully on a minimal install.

The spec is the same field with two changes: a new `name` (so the two reductions live side by
side under `output/`) and the backend dial.
"""
spec_starred = dataclasses.replace(
    spec,
    name="omegacen_f606w_starred",  # Separate output folder for the side-by-side comparison.
    psf_backend="starred",          # Tier 1b: STARRED joint PSF reconstruction.
)

from autoreduce.psf.starred_epsf import StarredUnavailableError

print("Running the same reduction with psf_backend='starred' (cache warm — fast)...")

try:
    record_starred = reduce_target(
        spec_starred, cache_root=CACHE_ROOT, output_root=OUTPUT_ROOT
    )
except StarredUnavailableError:
    record_starred = None
    print(
        "STARRED is not installed, so the tier-1b reduction is skipped. "
        'Run `pip install "autoreduce[starred]"` to enable it — the '
        "photutils reduction above is complete and this script continues "
        "with it alone."
    )

out_starred = OUTPUT_ROOT / spec_starred.name

if record_starred is not None:
    print(f"STARRED psf block: {json.dumps(record_starred['psf'], indent=2)}")

"""
__The Head To Head__

How do you score two PSFs against each other? On a stellar field there is a truth reference:
register every selected star to sub-pixel accuracy, stack them, and normalize — the empirical
stack. Then compare each backend's PSF to it on simple, robust statistics: the central 3x3
concentration (how much flux the core holds) and the RMS deviation of the radial profile.

The validated comparison on this exact field found:

- empirical star stack: concentration **0.58** (the truth reference);
- STARRED tier 1b:      concentration **0.54** — close to the stack, and the winner on
  radial-profile RMS;
- photutils tier 1:     concentration **0.39** — a visibly softer core.

The regime rule **PyAutoReduce** draws from this (and from the matching JWST comparison):
STARRED wins on *well-sampled* data — and UVIS F606W at native scale is well-sampled — while
photutils remains the safe default and wins on undersampled data (e.g. NIRCam SW), where
STARRED's own diagnostics flag the regime (it warns below a sampling FWHM of ~1.6 pixels).
Neither backend is "better"; the sampling regime decides, and the diagnostics recorded in
`reduction.json` tell you which regime you are in.

__Diagnostics and Plots__

The `psf` provenance block records the method, the number of stars used and the star-selection
pass for either backend — compare `n_stars_used` between the two runs above. Now plot the
delivered PSFs side by side on a log stretch, where core sharpness and wing behavior are
visible at once.
"""
from astropy.io import fits

psf_epsf = fits.getdata(out_epsf / "psf.fits").astype(float)

panels = [("photutils tier 1", psf_epsf)]

if record_starred is not None:
    psf_starred = fits.getdata(out_starred / "psf.fits").astype(float)
    panels.append(("STARRED tier 1b", psf_starred))

plot_dir = out_epsf / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5), squeeze=False)

for ax, (title, psf) in zip(axes[0], panels):
    im = ax.imshow(
        np.log10(np.clip(psf / psf.max(), 1e-6, None)), origin="lower", cmap="magma"
    )
    ax.set_title(f"{title} (log10, peak-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle("omegacen_f606w: UVIS PSF backends")
fig.tight_layout()

plot_path = plot_dir / "psf_backends.png"
fig.savefig(plot_path, dpi=120)
plt.close(fig)

print(f"Plot saved to: {plot_path}")

"""
Both reductions also ship `psf_full.fits` — the 61x61 extended-wings PSF for modeling
workflows that need the far profile (`guides/output_contract.py` explains the two-PSF
contract). And both PSFs obey the drizzled-PSF invariant: they are built from stars in the
*drizzled* mosaic, so they carry the same scale, pixfrac and kernel as the data they will be
convolved with.

__Wrap Up__

You have built the UVIS PSF two ways on a field where the truth is knowable, seen the
validated numbers that make STARRED the recommended backend for well-sampled UVIS data, and
read the diagnostics that let you audit the choice per dataset. When you reduce a lens field
(where the "stars" are scarcer and more suspect), the same machinery applies — with the
star-selection cuts and the `psf_star_pass` dial doing proportionally more of the work.

The following locations of the workspace are good places to checkout next:

- `scripts/hst_acs/psf.py`: the full PSF architecture — tiers, `psf_star_pass`, per-frame
  PSFs, the drizzled-PSF invariant.
- `scripts/hst_wfc3_uvis/start_here.py`: the UVIS anchor reduction these dials come from.
- `scripts/jwst_nircam/psf.py`: the same STARRED-vs-photutils story in the JWST sampling
  regimes.
- `scripts/guides/output_contract.py`: psf.fits vs psf_full.fits and the delivery contract.

__Env__ (Developer Only)

ENV: network
"""
