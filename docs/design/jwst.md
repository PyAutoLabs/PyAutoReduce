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

## Parity interpretation (maintainer guidance, 2026-07-09)

The demo dataset descends from the **bespoke COSMOS-Web team pipeline**
(custom 1/f destriping, wisp/snowball handling, their calibration vintage,
mosaics at 0.03″ SW / 0.06″ LW). The acceptance bar is therefore **"close +
internally consistent," not reproduction** — strong lensing needs its own
pipeline (this one), and order-unity data/noise ratios against the team
products are expected and acceptable. What must hold: our own internal
closures (sky vs ERR floor, WHT uniformity over the cutout, masked-pixel
policy) and cross-band consistency of any global scale offset.

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

Phase 3 ships tier 1; tier 2 (PSFEx/ShOpt) is the follow-up (external
binaries/Julia — an integration decision for a dedicated prompt). **Tier 2b
is live for frame products (issue #29):** `psf/stpsf_model.py` evaluates
STPSF at the frame's detector + target position and ships the `DET_DIST`
extension — detector-sampled *including geometric distortion*, the correct
kernel for native-frame products — whenever a frame's own star field cannot
support the tier-1 ePSF. The literature caveat rides the diagnostics, and a
missing stpsf install is a recorded outcome. Local gotcha: poppy
auto-detects cupy and JIT-compiles CUDA kernels that fail on this WSL2
toolchain — the wrapper pins `poppy.conf.use_cupy = False` (CPU FFTs are
ample at these fov sizes).

## Per-exposure frame products — feasibility (issue #24, 2026-07-10)

Whether the HST frames → registration → PSF chain (issues #16/#19/#21)
should and can extend to JWST. **Verdict: GO, phased** — technically
feasible with modest deltas; scientifically justified specifically for the
undersampled SW bands and precision applications, with mosaics remaining
the default for routine extended-source work.

**Implemented (issue #27, 2026-07-10)** per the deltas below, with three
facts implementation added: (a) the DQ policy divergence is load-bearing —
JWST masking is `dq & DO_NOT_USE` only (ramps remove CRs; informational
bits like JUMP_DET ride good pixels, and the ePSF estimator likewise
patches only DO_NOT_USE); (b) frame identity comes from the filename stem
minus the product suffix (JWST files carry no ROOTNAME; the HST-era
`split('_')[0]` collides across a visit); (c) the frames manifest is
**schema v2** — `data_units` derived (loud on heterogeneous inputs),
`sky_subtracted`/`sky_keyword` generalise MDRIZSKY, and `source` records
the input family (`_crf` vs `_cal` fallback); (d) **registration residuals
carry a reliability flag** — JWST dithers routinely put the target near
detector edges, and a correlation between mostly-masked cutouts locks onto
mask geometry (~200 px "residuals" on the COSMOS-Web validation), so the
reference is the best-covered frame, pairs with >20% masked pixels are
flagged unreliable, and the headline `max_registration_residual_px` is an
honest `null` when no clean pair exists.

### Should — what the literature and instrument design say

**For frame-level modeling:**

- **SW undersampling is the strongest argument.** NIRCam SW (native
  0.031″/px) undersamples its PSF (F115W FWHM ≈ 0.04″ ≈ 1.3 px); STScI's
  subpixel dither patterns exist precisely to recover that information, and
  a resampled mosaic partially destroys it (aliasing + interpolation).
  Forward-modeling the dithered frames uses the sub-pixel phases directly —
  the information-preserving approach.
- **Precision shear is moving frame-level.** The Roman HLIS metacalibration
  study (Yamamoto et al. 2022, arXiv:2203.08845) benchmarks joint
  multi-epoch (single-exposure) measurement against coadds — multi-epoch
  avoids coadd-PSF discontinuities and correlated noise and performed
  better (m = −0.76 ± 0.43% vs −1.13 ± 0.60%). The same argument applies
  to any shear-grade or substructure-grade lens measurement with NIRCam.
- **Frame-level is standard practice in crowded-field photometry and
  astrometry** (DOLPHOT's JWST module; the Anderson ePSF lineage) — the
  machinery culture exists, just not yet for extended-source fitting.
- **Resample correlates noise exactly as drizzle does** (this design's own
  noise stage applies the same Casertano R) — per-frame fitting removes the
  correlated-noise approximation entirely.

**Against / tempering:**

- Published JWST extended-source practice is mosaic-based today: COSMOS-Web
  weak lensing (ShOpt on mosaics), AGN decomposition (mosaic star
  libraries), deep-field galaxy-formation morphology. The scan found no
  published per-frame forward modeling of galaxy/lens sources with JWST —
  this would be ahead of the field, not following it.
- **Frame-level artifacts arrive unmitigated**: 1/f striping, wisps and
  snowballs are corrected (when they are) by mosaic-pipeline steps or team
  pipelines; per-frame modeling inherits them raw. The manifest must carry
  this caveat; the COSMOS-Web parity note above already flags the same gap
  for our mosaics.
- LW bands are well-sampled — little sampling gain there (the correlated
  noise and per-frame PSF arguments still apply).

### Can — deltas vs the HST frames mode

Anatomy is compatible: `_cal` files carry SCI/ERR/DQ (one detector per
file, so the existing per-(exposure, SCI-EXTVER) loop degenerates cleanly),
ramp-jump cosmic-ray flags are already in DQ from stage 1, and the
footprint/registration/PSF machinery is geometry-agnostic. The deltas:

1. **Input products** — package the `_crf` outputs of calwebb_image3
   (outlier-flagged, tweakreg-updated cal files; needs
   `steps={"outlier_detection": {"save_results": True}}` + capturing the
   paths in drizzle provenance) so frames carry the stack-based outlier
   flags, exactly as HST frames carry driz_cr flags. Fall back to `_cal`
   with a recorded absence when image3 didn't run.
2. **Units** — keep native MJy/sr (defaults-first, matches the mosaic):
   `_units_to_cps` gains a surface-brightness branch that records
   "none (native MJy/sr)" instead of raising.
3. **Sky** — no `MDRIZSKY`; skymatch's per-image levels live in the
   datamodel meta (`BKGLEVEL` on `_crf`). Subtract when present + record,
   0.0 recorded otherwise — mirroring the HST convention.
4. **CR provenance** — deepCR has no JWST model and isn't needed:
   `cr_method = "ramp-jump (calwebb stage 1) + image3 outlier_detection
   (crf)"`; `dq_semantics` switches to the JWST DQ flag table.
5. **WCS** — cal/crf carry gwcs (ASDF) plus the FITS-approx SIP the
   footprint filter already uses; the cutout ships the SIP approximation
   with its fidelity recorded, and the `target_pixel` anchor should project
   through gwcs where available. The measured relative-registration block
   carries over unchanged (it is empirical); the absolute-solution keywords
   (`RMS_RA`/`RMS_DEC`) have no cal-header equivalent — record the tweakreg
   fit metadata or "unknown".
6. **Per-frame ePSF** — machinery carries over with `peak_max=None` (the
   established convention for surface-brightness units); STPSF is the
   tier-2b fallback (per-detector, per-position — a stronger story than
   HST's TinyTim). Note: a single frame's ePSF is itself undersampled at
   SW; the `psf_from_frames` combination across subpixel dithers is where
   the sampling recovery actually happens.
7. **Guard** — relax the `frame_products`/`psf_from_frames` HST-only check
   to `observatory in ("hst", "jwst")` once the branch above lands.

### Recommendation

File the implementation as `feature/pyautoreduce/jwst_frame_products.md`
once accepted, scoped to the deltas above with the COSMOS-Web ring
(4 bands, SW + LW) as the validation anchor — it exercises undersampled SW
and well-sampled LW in one dataset. Frame-level artifacts (1/f, wisps)
ship as recorded caveats, not blockers.

## Open items

- Tier-2 spatially-varying PSF back-end (PSFEx or ShOpt — see table above);
  STPSF as explicit 2b fallback; unit-aware saturation cut for star
  selection in MJy/sr mosaics.
- jwst pinned at **1.14.0** by the PyAuto env constraints (astropy 6.1.2);
  provenance records the version — revisit when the env's astropy moves.
- COSMOS-Web official reduction (Franco et al.) applies additional
  corrections (1/f striping, wisps, snowballs) beyond default calwebb;
  parity ratios will show whether they matter at lens-cutout scale.
