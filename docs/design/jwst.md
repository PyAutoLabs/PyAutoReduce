# JWST/NIRCam — per-stage deltas vs the HST design

Phase 3. The first non-HST observatory, and the phase that forced the
**backend dispatch**: stage 3 now routes through
`InstrumentAdapter.combine_backend` (`astrodrizzle` | `jwst_image3`) and the
CRDS server/env shape is adapter-owned (`observatory`, `crds_server_url`;
the jwst pipeline reads `CRDS_PATH` directly — no `jref`-style variable).

| Stage | Delta vs HST |
|-------|--------------|
| acquire | level-2 **`_cal`** products (calwebb_image2 output, **MJy/sr**), `obs_collection="JWST"`; no explicit bestrefs — the jwst pipeline syncs references lazily through `CRDS_PATH`/jwst-crds |
| align | tweakreg runs *inside* calwebb_image3 (defaults-first); the standalone stage only records WCS provenance |
| combine | `calwebb_image3` (tweakreg / skymatch / outlier_detection / **resample** — the drizzle analogue). The lensing dials map to the resample step: `pixel_scale`, `pixfrac`, `kernel`, `rotation=0`, `weight_type=ivm`. The multi-extension `_i2d` is normalized to standalone sci/wht/err files so downstream stages stay backend-agnostic |
| noise | **read, don't construct**: the resampled `ERR` array (Poisson + read noise + flat, propagated by the pipeline) × the same Casertano R (resample correlates pixels exactly as drizzle does). A blank-sky consistency check (`sky_over_err_floor`) is recorded; disagreement is investigated, never absorbed |
| units | native **MJy/sr** kept (defaults-first — no conversion unless parity demands one); `BUNIT` rides the cutout header |
| psf | tier-1 ePSF unchanged (NaN-masked star finding); **no full-well peak cut** — meaningless in surface-brightness units, and saturated cores arrive blanked from level 2. STPSF is the designated tier-2 back-end (open item) |
| scales | SW native 0.031″ → recommended 0.03″; LW native 0.063″ → recommended 0.06″ (the COSMOS-Web mosaic convention the parity anchor uses). Filter→channel routing via `nircam_adapter_for_filter` |

## Validation anchor — the COSMOS-Web ring, four bands

The autolens_assistant demo dataset
(`dataset/imaging/cosmos_web_ring/wavebands/{F115W,F150W,F277W,F444W}`)
carries modeling-ready products for the ring (RA 150.10048, +1.89301;
[Mercier et al. 2024](https://arxiv.org/abs/2309.15986)) in all four
COSMOS-Web bands — SW at 0.03″/pix (419²), LW at 0.06″/pix (209²), stripped
headers as usual. `scripts/reduce_cosmos_web_ring.py --band <F>` reduces each
band from MAST `_cal` exposures and reports sub-pixel-registered data/noise
ratios against the demo products (the SLACS-parity method).

## PSF options — what the JWST weak-lensing and AGN literature says (2026-07-08)

- **Weak lensing (COSMOS-Web's own practice):**
  [ShOpt.jl](https://arxiv.org/abs/2401.11625) (Berman & McCleary 2024) is
  COSMOS-Web's PSF characterization tool, benchmarked against **PSFEx** and
  **PIFF** on real + simulated COSMOS-Web NIRCam imaging; all model the PSF
  **empirically from field stars with low-order polynomial spatial variation
  in (X, Y)** across the resampled mosaic. NIRCam PSFs vary with time,
  bandpass and field position, so star-based per-mosaic models are the norm.
- **AGN decomposition:** [Zhuang & Shen
  2024](https://arxiv.org/abs/2304.13776) characterize NIRCam PSFs in 8
  filters: spatial FWHM variation shrinks strongly with wavelength (max/RMS
  ~20%/5% at F070W → **~3%/0.6% at F444W**); among SWarp / photutils / PSFEx
  they find **PSFEx best**; PSF mismatch biases host fluxes high. COSMOS-Web
  AGN work ([Zhuang et al. 2024](https://iopscience.iop.org/article/10.3847/1538-4357/ad1517))
  and the galight PSF-library approach (Ding et al.; SHELLQs-JWST) use
  curated star libraries / hybrid empirical PSFs; **pure STPSF (WebbPSF)
  models are consistently disfavoured vs empirical** for decomposition work.

**Adopted tiering for JWST (revision of the HST-era tier 2):**

| Tier | Method | When |
|------|--------|------|
| 1 | single ePSF from mosaic stars (current photutils implementation) | **LW bands** (F277W/F444W): spatial variation ≲1% RMS — a single ePSF at the lens position is adequate for lens-galaxy work |
| 2 | **spatially-varying empirical model evaluated at the lens position** — PSFEx-style polynomial (PSFEx or ShOpt back-end) | **SW bands** (F115W/F150W: ~5% RMS variation) and any weak-lensing-grade use; photutils ranks below PSFEx in the Zhuang & Shen benchmark, so this is the quality upgrade path |
| 2b | STPSF model PSF | fallback only when the field lacks stars — flagged in provenance, never silent (the literature's consistent verdict: empirical beats model for decomposition) |
| 3 | STARRED / PSFr iterative reconstruction | lensed quasars/AGN, unchanged from the HST design |

Phase 3 ships tier 1; tiers 2/2b are the follow-up (PSFEx/ShOpt are external
binaries/Julia — an integration decision for a dedicated prompt).

## Open items

- Tier-2 spatially-varying PSF back-end (PSFEx or ShOpt — see table above);
  STPSF as explicit 2b fallback; unit-aware saturation cut for star
  selection in MJy/sr mosaics.
- jwst pinned at **1.14.0** by the PyAuto env constraints (astropy 6.1.2);
  provenance records the version — revisit when the env's astropy moves.
- COSMOS-Web official reduction (Franco et al.) applies additional
  corrections (1/f striping, wisps, snowballs) beyond default calwebb;
  parity ratios will show whether they matter at lens-cutout scale.
