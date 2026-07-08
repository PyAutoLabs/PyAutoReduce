# WFC3 (UVIS + IR) — per-stage deltas vs the ACS/WFC design

Phase 2. `hst_acs_pipeline.md` remains the stage-by-stage spec; this page
records only what WFC3 changes. Everything below lives in the two adapters
(`instruments/wfc3_uvis.py`, `instruments/wfc3_ir.py`) — no stage module
mentions a detector.

## WFC3/UVIS — the ACS-like path

| Stage | Delta vs ACS |
|-------|--------------|
| acquire | `_flc` (CTE-corrected) as ACS; references key **`iref`**, CRDS subpath `references/hst/wfc3` |
| align / drizzle / noise / psf / package | unchanged — same recipes, same dials |
| scale | native **0.0396″/pix**; adapter recommends output at native scale |
| saturation | ~63 ke- full well (star selection peak cut) |

**Validation anchor:** the published [Bayer et al.
(arXiv:1803.05952)](https://arxiv.org/abs/1803.05952) F390W reduction of
SDSS J0252+0039 — output 0.0396″/pix, **pixfrac 1.0**, noise recipe
σ = √(N/W + σ²_sky) with σ_sky ≈ 0.002 e-/s. Our integration script
(`scripts/reduce_j0252_uvis.py`) reduces the same data with those dials and
checks: output units e-/s, empirical σ_sky in that regime, noise-map
consistency, WHT uniformity. (Note their correlated-noise treatment is
blank-sky *realizations*, not the scalar R — with pixfrac 1.0 at native
scale, R = 1.5; comparisons account for whether R is applied.)

## WFC3/IR — the genuinely different path

| Stage | Delta vs ACS |
|-------|--------------|
| acquire | **`_flt`** — no CTE correction exists for the IR channel; `iref` references |
| drizzle | up-the-ramp fitting in `calwf3` already rejects most CRs per read; `driz_cr` still runs on multi-exposure stacks for the residue (defaults-first) — documented, revisit if IR integrations show over-flagging |
| scale | native **0.128″/pix** under-samples the PSF; adapter recommends **0.065″/pix** for dithered data (half-native, in the 0.06–0.08 deep-field range). The dial stays user-facing; the fine-grid Casertano branch (s < p) then applies, so R is materially larger — reported per run as always |
| units | `_flt` IR data are already e-/s (count rates); `final_units='cps'` unchanged |
| saturation | ~78 ke- effective full well |
| psf | same tiers; STScI focus-diverse ePSF grids exist for IR when tier 2 lands |

## Coverage audit vs `ajshajib/hst-lens` (the checklist, not the architecture)

Their three notebooks (Download / IR / UVIS) cover: archive download,
per-channel calibrated products, AstroDrizzle combination, and cutouts. Ours
adds what they lack: instrument adapters (theirs is notebook-per-channel),
provenance (`reduction.json`), an explicit noise-map recipe with correlated-
noise handling, tiered PSF construction with diagnostics, cache/eviction, and
loud-failure contracts. Nothing in their steps is absent from our stage graph.

## Open items

- IR integration target: discovered via MAST at run time; if no suitable
  IR lens dataset is reachable, the leg parks as a batched question.
- Tier-2 PSF for IR (focus-diverse ePSF grids) — with the roadmap's tier-2
  work, not phase 2.
