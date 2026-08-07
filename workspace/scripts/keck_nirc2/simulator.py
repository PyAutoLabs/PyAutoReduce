"""
Simulator: Keck NIRC2 AO Injection
==================================

How do you know a reduction pipeline — or the lens model you run on its products — recovers
flux faithfully? You inject a source of *known* brightness into the real data, reduce again,
and check what comes out. This script injects a synthetic Einstein-ring into the real SHARP
B1938+666 frames, immediately after the ground stages have prepared them, and runs the
identical pipeline on the injected set.

Injection into real frames beats simulating an observation from scratch, because everything
hard about the data comes for free: the real sky and its variations, the real bad pixels,
the real dither geometry, the real AO PSF and its halo, the real correlated noise of the
combine. This is also the field's precedent for AO: Chen et al. 2021 (MNRAS 508, 755,
arXiv:2106.11060) validate AO PSF reconstruction on simulated lensed-quasar observations
built from realistic, *empirical* Keck AO PSFs, and the AIROPA project simulates NIRC2
scenes the same way (arXiv:2207.00548) — empirical-PSF injection plus a noise model, not
end-to-end atmospheric simulation.

Run `start_here.py` first: this script reuses its warm cache, and borrows a tier-A PSF
candidate from its output as the injection PSF.

__Contents__

- **Imports:** Import the required Python libraries.
- **Paths:** Anchor every path to the workspace root.
- **The Input Image:** A Sersic ring in pure numpy — the formula, the units, the contract.
- **Placement Without WCS:** Why injection never trusts the raw header WCS, and what it uses instead.
- **The Spec, From YAML:** `TargetSpec.from_yaml` plus `dataclasses.replace` — the injection idiom.
- **Clean Reduction:** The reference run, and the PSF candidate it donates.
- **Injected Reduction:** The same pipeline with the `inject_*` dials set.
- **No ERR Bookkeeping:** Why the Keck path updates no error extensions — noise is constructed downstream.
- **Recovery Check:** Difference the mosaics and compare recovered flux against what went in.
- **Registration Unchanged:** The measured offsets must agree between the clean and injected runs.
- **Plots:** Clean, injected, difference.
- **Wrap Up:** Summary and where to go next.

__Imports__
"""
import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.io import fits

from autoreduce import TargetSpec, reduce_target
from autoreduce import instruments
from autoreduce.acquire import koa

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the
folder containing `scripts/`). The clean and injected runs get separate output roots (each
run writes `<output_root>/<name>/`), while the cache — which injection never mutates — is
shared with `start_here.py`.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"
OUTPUT_ROOT = WORKSPACE / "output"

sim_root = OUTPUT_ROOT / "simulator_keck"
sim_root.mkdir(parents=True, exist_ok=True)

"""
__The Input Image__

The input to injection is deliberately dumb: a plain 2-D FITS image of the source *before
the telescope* — finite, non-negative, **not** PSF-convolved (the pipeline convolves with
the epoch PSF per frame), at its own pixel scale, in the Keck adapter's injection units of
**e-/s per pixel**. No lensing code is imported to make it; a ring is just geometry.

We build a Sersic ring: an exponential (n = 1) Sersic profile in the radial distance from a
circle of radius `r_ring`, mimicking a thin Einstein ring like B1938's:

    rho(y, x)  = sqrt(y^2 + x^2)                       radial distance from the ring centre
    I(y, x)    = A * exp( -b_1 * |rho - r_ring| / r_e )   with b_1 = 1.678 (n = 1)

(the general Sersic exponent is b_n ~ 2n - 1/3; n = 1 gives an exponential fall-off either
side of the ring crest). The image is normalised so its *total* flux is a known number of
e-/s — that number is what the recovery check hunts for. We render it at 5 mas/pixel, finer
than the native 9.942 mas, so the pipeline's flux-conserving resampling does the down-binning.
"""
INPUT_SCALE = 0.005  # arcsec/pixel of the input image (finer than native — deliberately).
RING_RADIUS = 0.30  # arcsec — the ring crest radius.
RING_R_E = 0.05  # arcsec — Sersic effective radius across the ring.
TOTAL_FLUX_EPS = 200.0  # e-/s — the known truth the recovery check compares against.

shape = (241, 241)
yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
rho = np.hypot(yy - cy, xx - cx) * INPUT_SCALE
ring = np.exp(-1.678 * np.abs(rho - RING_RADIUS) / RING_R_E)
ring = TOTAL_FLUX_EPS * ring / ring.sum()

input_path = sim_root / "input_sersic_ring_eps.fits"
fits.PrimaryHDU(ring.astype(np.float32)).writeto(input_path, overwrite=True)
print(f"Wrote input ring image ({TOTAL_FLUX_EPS:.0f} e-/s total) to: {input_path.resolve()}")

"""
__Placement Without WCS__

On HST or JWST, injection renders the input through each exposure's full-distortion WCS —
those WCS solutions are Gaia-tied and milliarcsecond-grade. A raw NIRC2 header WCS is
**arcsecond-grade**: at 9.942 mas/pixel, trusting it would misplace the source by *tens to
hundreds of native pixels*, and by a different amount in every frame.

So the Keck path never touches the header WCS. It reuses the same measured-offset arithmetic
the combine itself uses: a pre-pass measures `offsets_to_reference` on the prepared frames
(phase cross-correlation), reproduces the combine's deterministic mosaic geometry, places the
target at the mosaic centre (the Keck path's WCS convention), and inverts the frame<->mosaic
mapping to find each frame's injection centre. Because the injected content is placed
consistently with the measured offsets, it reinforces — never biases — the registration when
the combine re-measures it (checked empirically below).

One consequence to internalise: **`inject_position` is honoured as an offset from the
target, not as an absolute position.** You still write it as `(ra, dec)` degrees, but only
the *difference* from `(spec.ra, spec.dec)` is used, applied by offset arithmetic on the
mosaic grid (whose axes follow the detector frame, x ~ -RA / y ~ +Dec — our circular ring is
orientation-agnostic, so this does not matter here). We offset the ring 0.8" in Dec so the
difference image lands on clean sky, well inside the 2.81" cutout.

__The Spec, From YAML__

A reduction is declared, not scripted — and the cleanest injection workflow declares the
*base* spec once (here, written to YAML and loaded back with `TargetSpec.from_yaml`, the
file format you would keep per-target in a real project), then derives the injected variant
with `dataclasses.replace`. `TargetSpec` is frozen, so `replace` is the only way to vary it
— which is exactly what makes clean-vs-injected pairs trustworthy: everything not explicitly
replaced is *guaranteed* identical.

First, pin the frame set exactly as `start_here.py` did (KOA has no association tables; see
that script for the full discovery story).
"""
RA, DEC = 294.60496, 66.81450
FILTER = "Kp"

adapter = instruments.get("nirc2_narrow")
query_dir = sim_root / "koa_queries"
query_dir.mkdir(parents=True, exist_ok=True)

print("Querying KOA to pin the B1938+666 frame set (metadata only; downloads hit the warm cache)...")

science_table = koa.query_science_frames(
    RA, DEC, adapter, FILTER, query_dir, proposal_ids=None, koa_ids=None
)

from collections import Counter

separation_arcsec = (
    np.hypot(
        (np.asarray(science_table["ra"], float) - RA) * np.cos(np.radians(DEC)),
        np.asarray(science_table["dec"], float) - DEC,
    )
    * 3600.0
)
lens_table = science_table[separation_arcsec < 12.0]
star_table = science_table[separation_arcsec >= 12.0]

setups = Counter((float(r["itime"]), int(r["coadds"])) for r in lens_table)
(modal_itime, modal_coadds), _ = setups.most_common(1)[0]
science_ids = [
    str(r["koaid"])
    for r in lens_table
    if float(r["itime"]) == modal_itime and int(r["coadds"]) == modal_coadds
]
star_itimes = np.asarray(star_table["itime"], float)
short = star_itimes <= 60.0
star_rows = star_table[short] if short.any() else star_table
star_ids = [str(k) for k in star_rows["koaid"]][:12]

spec_yaml = {
    "name": "b1938+666",
    "ra": RA,
    "dec": DEC,
    "instrument": "nirc2_narrow",
    "filter_name": FILTER,
    "final_scale": 0.010,
    "final_pixfrac": 1.0,
    "cutout_shape": [281, 281],
    "koa_science_ids": science_ids,
    "koa_psf_star_ids": star_ids,
}
spec_path = sim_root / "b1938_spec.yaml"
spec_path.write_text(yaml.safe_dump(spec_yaml, sort_keys=False))
print(f"Wrote base spec YAML to: {spec_path.resolve()}")

base_spec = TargetSpec.from_yaml(spec_path)

"""
__Clean Reduction__

The reference run. Its cache is warm from `start_here.py`, so no downloads happen — but the
full calibrate/sky/combine/noise/psf chain runs again into its own output root.
"""
print(
    """
    Starting the CLEAN reduction (reference run). With a warm cache this is
    reduction-only — expect minutes, not the first-run download time.
    """
)

clean_record = reduce_target(base_spec, cache_root=CACHE_ROOT, output_root=sim_root / "clean")

clean_dir = sim_root / "clean" / base_spec.name

"""
The clean run also donates the injection PSF. **`inject_psf` is required on the Keck path**:
the AO PSF varies per epoch, no stable model library exists, and the tier-A candidates are
not built until after the combine — so no automatic per-frame PSF source exists at injection
time. You must hand injection an epoch-specific PSF explicitly, and the natural choice is a
`psf_candidate_<i>.fits` from a previous reduction of the same data. We take the selected
(sharpest) candidate.
"""
psf_stats = clean_record["psf"]["candidates"]
selected_index = int(np.argmax([s["peak_fraction"] for s in psf_stats]))
inject_psf_path = clean_dir / f"psf_candidate_{selected_index}.fits"
print(f"Injection PSF: {inject_psf_path.name} (epoch {psf_stats[selected_index]['epoch']}, "
      f"FWHM {psf_stats[selected_index]['fwhm_arcsec'] * 1000:.0f} mas).")

"""
__Injected Reduction__

Now the injected variant via `dataclasses.replace`. The injection runs *after* the ground
stages have prepared the frames (calibrated, sky-subtracted) and *before* the combine — into
work-directory copies, never the cache. Per frame, the pipeline renders the input through
the measured-offset placement, convolves with your `inject_psf`, converts e-/s to that
frame's total electrons (x ITIME x COADDS), and adds a Poisson draw of those counts — the
injected source carries its own shot noise, like a real one.
"""
injected_spec = dataclasses.replace(
    base_spec,
    inject_image=str(input_path),  # the plain-FITS ring, e-/s per pixel, un-convolved.
    inject_pixel_scale=INPUT_SCALE,  # arcsec/pixel of the input image — required with inject_image.
    inject_position=(RA, DEC + 0.8 / 3600.0),  # honoured as an OFFSET from the target: +0.8" in Dec.
    inject_psf=str(inject_psf_path),  # REQUIRED on Keck: the epoch-specific tier-A candidate.
    inject_seed=0,  # deterministic Poisson draws (per-frame streams derive from this + the filename).
)

print(
    """
    Starting the INJECTED reduction — identical pipeline, identical cache,
    plus the synthetic ring. Expect a similar runtime to the clean run.
    """
)

injected_record = reduce_target(injected_spec, cache_root=CACHE_ROOT, output_root=sim_root / "injected")

injected_dir = sim_root / "injected" / base_spec.name

"""
__No ERR Bookkeeping__

On HST and JWST, injection must update each exposure's ERR extension in quadrature — those
pipelines *read* propagated errors downstream. The Keck prepared frames carry **no ERR
extension**, and injection deliberately keeps it that way: the Keck noise map is
*constructed* downstream from the mosaic counts and the IVM weights
(`R x sqrt(sci/exptime + 1/wht)`), so the injected source's Poisson noise flows into the
shipped `noise_map.fits` through the mosaic counts themselves — naturally, with no
bookkeeping to get wrong. The provenance `inject` block records everything that went in:
"""
print(f"Injected total: {injected_record['inject']['total_injected_e']:.0f} e- "
      f"across {len(injected_record['inject']['frames'])} frames.")
print(f"Placement: {injected_record['inject']['placement']}")
print(f"PSF source: {injected_record['inject']['psf_source']}")

"""
__Recovery Check__

Difference the two mosaics — everything real cancels, leaving the injected ring plus noise —
then sum the difference in an aperture around the ring and compare with the known input
flux. The aperture centre comes from the flux-weighted centroid of the significant positive
difference pixels (a ring has no central peak, so a naive brightest-pixel centre would land
on the crest).
"""
data_clean = fits.getdata(clean_dir / "data.fits").astype(float)
data_inj = fits.getdata(injected_dir / "data.fits").astype(float)
noise_inj = fits.getdata(injected_dir / "noise_map.fits").astype(float)

diff = data_inj - data_clean
pixel_scale = injected_record["package"]["pixel_scale"]

significant = np.nan_to_num(diff, nan=0.0)
significant = np.where(significant > 3.0 * np.nanmedian(noise_inj), significant, 0.0)
yy, xx = np.mgrid[0 : diff.shape[0], 0 : diff.shape[1]]
centroid_y = float((significant * yy).sum() / significant.sum())
centroid_x = float((significant * xx).sum() / significant.sum())

aperture = np.hypot(yy - centroid_y, xx - centroid_x) * pixel_scale <= RING_RADIUS + 3.0 * RING_R_E
recovered = float(np.nansum(diff[aperture]))
aperture_noise = float(np.sqrt(np.nansum(noise_inj[aperture] ** 2)))

recovery_ratio = recovered / TOTAL_FLUX_EPS
print(f"Injected {TOTAL_FLUX_EPS:.0f} e-/s; recovered {recovered:.0f} e-/s in the aperture "
      f"(ratio {recovery_ratio:.3f}; aperture noise {aperture_noise:.1f} e-/s).")

"""
Expect the ratio within a few percent of 1.0 (the injected Poisson noise and the aperture
noise set the scatter). A systematic deficit would mean flux is being lost somewhere in the
chain — which is precisely what this test exists to catch.

__Registration Unchanged__

The design promise from the placement section, checked: because the injected content is
placed consistently with the measured offsets, the combine's re-measured registration must
agree between the clean and injected runs. Disagreement would mean the injection biased the
phase correlation — invalidating the whole "identical pipeline" premise.
"""
off_clean = np.asarray(clean_record["drizzle"]["registration_offsets_native_pix"])
off_inj = np.asarray(injected_record["drizzle"]["registration_offsets_native_pix"])
max_shift = float(np.max(np.abs(off_clean - off_inj)))
print(f"Max registration shift clean vs injected: {max_shift:.3f} native pixels (expect < 0.1).")

report = {
    "injected_flux_eps": TOTAL_FLUX_EPS,
    "recovered_flux_eps": recovered,
    "recovery_ratio": recovery_ratio,
    "aperture_noise_eps": aperture_noise,
    "registration_max_offset_shift_pix": max_shift,
    "total_injected_e": injected_record["inject"]["total_injected_e"],
}
report_path = sim_root / "recovery_report.json"
report_path.write_text(json.dumps(report, indent=2))
print(f"Wrote recovery report to: {report_path.resolve()}")

"""
__Plots__

Clean, injected, and the difference — the injected ring should sit 0.8" north of the real
lens, and the difference panel should show *only* the ring on a field of noise.
"""
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
scale = np.nanpercentile(data_clean, 90)
axes[0].imshow(np.arcsinh(data_clean / scale), origin="lower", cmap="magma")
axes[0].set_title("clean data.fits")
axes[1].imshow(np.arcsinh(data_inj / scale), origin="lower", cmap="magma")
axes[1].set_title("injected data.fits")
axes[2].imshow(np.arcsinh(diff / np.nanmedian(noise_inj)), origin="lower", cmap="magma")
axes[2].set_title("difference (injected ring)")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plot_path = sim_root / "injection_recovery.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Saved injection-recovery plot to: {plot_path.resolve()}")

"""
__Wrap Up__

You injected a synthetic Einstein-ring of known flux into the real SHARP B1938+666 frames —
placed by measured-offset arithmetic (never the arcsecond-grade header WCS), convolved with
an epoch-specific tier-A PSF candidate, carrying its own Poisson noise — reduced through the
byte-identical pipeline, and verified both flux recovery and that the injection left the
registration untouched. The provenance `inject` block keeps the semi-synthetic dataset from
ever masquerading as real.

The following locations of the workspace are good places to checkout next:

- `scripts/keck_nirc2/start_here.py`: the clean reduction this script builds on.
- `scripts/keck_nirc2/psf.py`: where the injection PSF candidates come from, and their contract.
- `scripts/hst_acs/simulator.py`: the HST injection path — WCS-based placement and ERR bookkeeping, for contrast.
- `scripts/alma/simulator.py`: the visibility-domain analogue, via CASA simobserve.

__Env__ (Developer Only)

Not user documentation: this section configures the automated test harness. This script
needs network access (KOA metadata queries) plus the cache and products of `start_here.py`,
so the smoke runner skips it.

ENV: network
"""
