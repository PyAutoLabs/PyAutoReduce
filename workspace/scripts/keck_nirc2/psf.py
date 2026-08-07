"""
PSF: Keck NIRC2 AO
==================

The PSF is where AO lens modeling is won or lost. A lensed arc is a ~0.1"-wide feature
convolved with a ~0.07" PSF: get the PSF wrong and the smooth-model residuals it leaves are
exactly the kind of localized flux a substructure search would claim as a dark subhalo. SHARP
III (Chen et al. 2016, MNRAS 462, 3457, arXiv:1601.01321) devotes itself to this problem —
the AO PSF is *the* dominant systematic of AO lens modeling.

This script tells the AO PSF story end to end as **PyAutoReduce** implements it: why no
reduction-time AO PSF can be final, the tier-A PSF-star-epoch design and its cosmic-ray
vetting gates, the `psf_provisional` contract, and the fallback tiers. It reads the products
of `start_here.py`, so run that first.

__Contents__

- **The AO PSF Problem:** Time-variable, field-variable, and no stable model library exists.
- **The SHARP Practice:** Dedicated PSF stars, and final PSF selection during modeling — not reduction.
- **Imports:** Import the required Python libraries.
- **Paths:** Anchor every path to the workspace root.
- **Tier A — PSF-Star Epochs:** Each epoch reduced pipeline-identically, every epoch shipped as a candidate.
- **Grouping Into Epochs:** MJD-gap grouping of the PSF-star frames, demonstrated on the cache.
- **The Vetting Gates:** Coherence and diffraction-sharpness — how a cosmic ray fails to impersonate a star.
- **The Candidates:** Load and plot every `psf_candidate_<i>.fits`, with the selection diagnostics.
- **Provisional By Contract:** `psf_provisional: true`, always — final selection belongs to modeling.
- **SHARP Parity:** The candidate FWHMs land in the ~65-70 mas range SHARP reports.
- **The FWHM Definition Wart:** Tier A and tier B measure FWHM differently — never cross-compare.
- **Tier B And Tier C:** The in-field ePSF fallback, and why target-based reconstruction is out of scope.
- **Wrap Up:** Summary and where to go next.

__The AO PSF Problem__

An AO PSF is not a property of the instrument; it is a property of the *moment*. The
correction quality tracks the seeing and the AO loop performance, so the PSF changes from
visit to visit and frame to frame. It also changes *across the field* — anisoplanatism
elongates the PSF away from the guide star (the AIROPA project models exactly this
spatial variability for NIRC2; Witzel et al., JATIS 8, 038007, arXiv:2210.10940 and
arXiv:2207.00548). On typical lens fields the delivered Strehl is ~10-30% and the FWHM
~60-90 mas (van Dam et al. 2006, PASP 118, 310, measure 30-40% Strehl at K only for bright,
on-axis stars).

Contrast this with HST or JWST, where optical models (TinyTim, STPSF) and stable ePSF
libraries exist because the optics barely change. For AO there is no stable library to look
up — and the narrow camera's ~10" field of view rarely contains a usable star, so you cannot
simply measure the PSF in-field either. (Telescope-telemetry PSF reconstruction — Ragland et
al., SPIE 9909 — is an active alternative, but not a reduction-pipeline commodity.) The PSF
must come from dedicated observations or from the science data itself.

__The SHARP Practice__

SHARP's answer, inherited here, has two halves:

1. **Observe a dedicated PSF star**, interleaved in time with the science dithers — for
   B1938+666 the tip-tilt star itself, ~20" away, visited repeatedly through the sequence
   (Lagattuta et al. 2012, MNRAS 424, 2800). Each visit samples the PSF *at that moment*.

2. **Defer the final choice to modeling.** No reduction-time statistic can know which epoch
   best matches the PSF that was in force during the science frames — but the lens model
   can: fit the same data with each candidate PSF and let the Bayesian evidence decide
   (the SHARP I practice). Reduction's job is to deliver *all* the candidates, reduced
   honestly, and to be explicit that none of them is final.

The logical extreme of this philosophy is to infer the PSF from the lensed images
themselves, iteratively, during modeling — SHARP III (Chen et al. 2016) showed this reaches
precision comparable to or better than HST, and Chen et al. 2021 (MNRAS 508, 755,
arXiv:2106.11060) validated AO PSF reconstruction against the astrometric requirements of
time-delay cosmography. That is a *modeling-stage* technique (tier C below) — but it only
works because the reduction hands modeling clean, pipeline-consistent starting candidates.

__Imports__
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from autoreduce.sky import group_by_time_gaps

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the
folder containing `scripts/`). We read the products and cache of `start_here.py`.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"
OUTPUT_ROOT = WORKSPACE / "output"

out_dir = OUTPUT_ROOT / "b1938+666"
record_path = out_dir / "reduction.json"

if not record_path.exists():
    raise FileNotFoundError(
        f"No reduction record at {record_path} — run scripts/keck_nirc2/start_here.py "
        f"first; this script reads its products."
    )

record = json.loads(record_path.read_text())
psf_diag = record["psf"]

"""
__Tier A — PSF-Star Epochs__

The default Keck PSF path (`tier A`) turns the interleaved PSF-star visits into candidates:

- The PSF-star frames are grouped into **epochs** — contiguous visits, split wherever the
  MJD gap between frames exceeds 600 s.
- Each epoch is reduced **pipeline-identically**: the same calibration masters, the same
  scaled running sky, the same one-pixmap `nirc2_native` combine at the same `final_scale`,
  `pixfrac` and kernel as the science mosaic. This is the drizzled-PSF invariant — the
  delivered PSF has been through *exactly* the resampling the science data have, so its
  pixel-level correlations and sub-pixel structure match what convolution with the model
  requires.
- **Every surviving epoch ships** as `psf_candidate_<i>.fits` (full `psf_full_shape`
  stamps), and `psf.fits` / `psf_full.fits` are cut from the *sharpest* candidate — highest
  peak fraction, a Strehl proxy, since at fixed total flux a higher peak means a better
  correction.

__Grouping Into Epochs__

The epoch grouping is the same `group_by_time_gaps` helper the sky stage uses, with the
600 s PSF-visit gap. We can reproduce it from the cached PSF-star frame headers directly.
"""
star_paths = sorted(
    p for p in (CACHE_ROOT / "b1938+666" / "psf").rglob("*.fits*") if "download" not in p.name
)

if star_paths:
    star_mjds = sorted(float(fits.getheader(p)["MJD-OBS"]) for p in star_paths)
    epochs = group_by_time_gaps(star_mjds, gap_s=600.0)
    print(f"{len(star_paths)} PSF-star frames -> {len(epochs)} epochs; sizes {[len(e) for e in epochs]}.")
else:
    print("No cached PSF-star frames found; skipping the epoch-grouping demonstration.")

"""
__The Vetting Gates__

A single-frame epoch has no outlier protection: with one frame there is nothing to
median-reject against, so a cosmic-ray hit can sit right where the star should be, and the
combine will faithfully resample it. Two physical gates catch impostors before they become
candidates, with every rejection recorded (reason and all) in provenance:

- **Coherence**: a real AO PSF is spatially coherent — a bright 3x3 core well above the
  local background scatter, with positive total flux. A hot pixel on empty sky is neither.
  Starless epochs (telescope offsets, failed acquisitions) fail here with a recorded reason;
  only an *all*-epochs-starless result is fatal.

- **Diffraction sharpness**: a real PSF cannot be narrower than the telescope's diffraction
  core (~45-50 mas at K' on Keck). Any "PSF" with FWHM below max(2 output pixels, 25 mas)
  is rejected as a cosmic ray. This gate earned its keep on the B1938 validation run: the
  CR-contaminated epochs measured 11-16 mas — far below anything optics can produce — while
  the real star measured ~72 mas.
"""
for rej in psf_diag.get("rejected_epochs", []):
    print(f"Rejected epoch {rej['epoch']} ({rej['n_frames']} frames): {rej['reason']}")
if not psf_diag.get("rejected_epochs"):
    print("No epochs were rejected in this run.")

"""
__The Candidates__

Now load what tier A shipped. The candidate list in provenance carries, per epoch: the
frame count, start MJD, peak fraction (the Strehl proxy used for selection) and the FWHM.
"""
print(f"Method:            {psf_diag['method']}")
print(f"Candidates:        {psf_diag['n_candidates']}, selected epoch {psf_diag['selected_epoch']}")
print(f"Selection rule:    {psf_diag['selection']}")

candidate_paths = sorted(
    out_dir.glob("psf_candidate_*.fits"),
    key=lambda p: int(p.stem.rsplit("_", 1)[1]),
)
candidates = [fits.getdata(p).astype(float) for p in candidate_paths]
stats = psf_diag["candidates"]

selected_index = int(np.argmax([s["peak_fraction"] for s in stats]))

fig, axes = plt.subplots(1, max(len(candidates), 1), figsize=(4 * max(len(candidates), 1), 4.5))
axes = np.atleast_1d(axes)
for i, (ax, cand, stat) in enumerate(zip(axes, candidates, stats)):
    ax.imshow(np.arcsinh(cand / cand.max() * 100.0), origin="lower", cmap="magma")
    marker = "  <- psf.fits" if i == selected_index else ""
    ax.set_title(
        f"candidate {i} (epoch {stat['epoch']})\n"
        f"FWHM {stat['fwhm_arcsec'] * 1000:.0f} mas, peak {stat['peak_fraction']:.3f}{marker}",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plot_dir = out_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)
cand_plot = plot_dir / "psf_candidates.png"
plt.savefig(cand_plot, dpi=150)
plt.close()
print(f"Saved candidate gallery to: {cand_plot.resolve()}")

"""
Look at the gallery: even between epochs of the same night the core width and the halo
structure differ — that visible epoch-to-epoch variation *is* the AO PSF problem, rendered.

__Provisional By Contract__

The provenance marks the Keck PSF `psf_provisional: true` — always, with no code path that
sets it false. This is deliberate contract, not a temporary limitation:

- The peak-fraction selection is a reasonable *reduction-time* heuristic, but the question
  "which candidate matches the PSF during the science frames?" is only answerable *against
  the science data* — by fitting the lens model once per candidate and comparing Bayesian
  evidence (the SHARP I practice). The information needed for the decision simply does not
  exist at reduction time.
- Marking the PSF provisional keeps a semi-trustworthy product from masquerading as a final
  one — the same honesty principle as the `injected:` block in simulated reductions. Your
  modeling workflow should treat `psf.fits` as the *default starting candidate*, and the
  `psf_candidate_<i>.fits` family as the decision to make.
"""
print(f"psf_provisional: {psf_diag['psf_provisional']}  (always true on the Keck path)")

"""
__SHARP Parity__

The published SHARP reference for this dataset puts the PSF-candidate core FWHM at
~65-70 mas. The acceptance validation for this pipeline checks each candidate lands in a
generous 45-120 mas window (diffraction floor to poor-correction ceiling) and expects the
selected candidate near the SHARP value.
"""
for stat in stats:
    fwhm_mas = stat["fwhm_arcsec"] * 1000
    in_range = 45.0 < fwhm_mas < 120.0
    print(f"Candidate epoch {stat['epoch']}: FWHM {fwhm_mas:.0f} mas — {'within' if in_range else 'OUTSIDE'} 45-120 mas.")

"""
__The FWHM Definition Wart__

One honest wart: the FWHM printed above is tier A's *equivalent-area* definition — the
diameter of a circle with the same area as the above-half-maximum region. The tier-B/tier-1
ePSF machinery measures FWHM from a radial profile instead. The two definitions agree only
for a perfectly circular PSF, which an AO PSF is not. Never compare a tier-A FWHM against a
tier-B one, or against a differently-defined literature value, at face precision — this is a
recorded open item, not a hidden inconsistency.

__Tier B And Tier C__

- **Tier B — in-field ePSF (the fallback).** If a spec pins no `koa_psf_star_ids`, the Keck
  path falls through to the same photutils ePSF machinery the HST path uses, building the
  PSF from whatever stars the ~10" field offers (usually few to none — which is exactly why
  tier A exists). A tier-B Keck PSF is *still* flagged provisional. The STARRED ePSF backend
  (Michalewicz et al. 2023, JOSS; Millon et al. 2024, AJ) is directly applicable to AO data
  in this same field-star role, though see the HST/JWST examples for where each backend wins.

- **Tier C — target-based reconstruction (out of scope).** For lensed AGN, the sharpest PSF
  source is the lensed quasar images themselves, fit iteratively during modeling (Chen et
  al. 2016; the PSFr and STARRED two-channel deconvolution tools work in this regime). That
  is a modeling-stage concern by construction — it requires the lens model — and is
  permanently outside reduction scope. The reduction's job ends at honest candidates.

__Wrap Up__

The AO PSF is time-variable, field-variable and unmodellable from first principles at
reduction time — so **PyAutoReduce** ships *every* pipeline-identically-reduced PSF-star
epoch as a candidate, vets each against the physics a cosmic ray cannot fake, selects a
default by Strehl proxy, and marks the result provisional so the final, evidence-based
choice happens where it belongs: against your science data, in the lens model.

The following locations of the workspace are good places to checkout next:

- `scripts/keck_nirc2/start_here.py`: the reduction that produced these candidates.
- `scripts/keck_nirc2/simulator.py`: injection with an epoch-specific candidate PSF as `inject_psf`.
- `scripts/hst_acs/psf.py`: the HST PSF story — star selection, ePSF tiers, `psf_star_pass`.
- `scripts/guides/output_contract.py`: `psf.fits` vs `psf_full.fits` and the drizzled-PSF invariant.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness. This script
reads products and cache written by `start_here.py`, so the smoke runner skips it.

ENV: network
"""
