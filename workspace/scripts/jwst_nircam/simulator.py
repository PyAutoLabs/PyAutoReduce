"""
JWST NIRCam: Simulator
======================

How do you know a reduction pipeline conserves flux? You feed it a source whose
brightness you know *exactly*, run the identical pipeline, and check what comes out.
This script does that on JWST data with **PyAutoReduce**'s injection mode: a synthetic
lensed-arc image, built in pure numpy with known total flux in janskys, is injected
into the real COSMOS-Web F150W `_cal` exposures — each frame receiving the source
through its own WCS, its own photometric calibration and its own PSF — and the full
calwebb_image3 reduction then runs twice, clean and injected. Differencing the two
mosaics recovers the source, and the recovered flux tests the whole chain end to end.

This is also the honest way to *simulate* JWST lens data. Injection into real frames
inherits everything a from-scratch simulator must model and always gets slightly wrong:
real cosmic-ray statistics, the real sky, real bad pixels, the real dither geometry,
real PSF wings and real correlated noise downstream. Injection into real *reduced*
imaging is the established pattern in lensing (the COWLS forecast-vs-data comparisons,
https://arxiv.org/abs/2503.08785, are built on synthetic lenses in real image
statistics); pushing the injection one level deeper — into the calibrated exposures,
*before* the combine stage — has no lens-specific published example we know of at the
uncal/ramp level, and pre-launch ramp simulators are not the practical route today.
Injecting at the `_cal` level, through the real pipeline, is the gap **PyAutoReduce**
fills: everything downstream of the injection point is real.

__Contents__

- **Imports:** Import **PyAutoReduce** and the supporting libraries.
- **Paths:** Anchor the cache and output folders to the workspace root.
- **The Input Image:** A pure-numpy lensed arc in Jy per pixel — the formula, the contract, the total flux.
- **The Unit Chain:** Jy per pixel through each frame's own PHOTMJSR and PIXAR_SR — flux-exact in the mean.
- **Noise Bookkeeping:** The source's Poisson noise, the e_per_dn disclosure, and the ERR update before image3.
- **Clean and Injected Reductions:** The same spec twice, sharing one exposure cache.
- **The Inject Provenance Block:** What the reduction record says was done to the data.
- **Recovery:** Difference the mosaics, integrate in an aperture, convert MJy/sr to Jy, compare to truth.
- **Plots:** Clean, injected and difference images.
- **Wrap Up:** Where to go next.

__Imports__
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS

from autoreduce import TargetSpec, reduce_target
from autoreduce.instruments import nircam_adapter_for_filter

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the
folder containing `scripts/`). **PyAutoReduce** requires absolute paths: its combine step
changes the working directory internally, so relative paths would break.

The clean and injected reductions get separate output trees but share one exposure
cache — the cache is never mutated by injection (frames are copied to a scratch area
before the source is added), so the same downloaded `_cal` files serve both runs.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"  # downloaded exposures + CRDS references (shared by both runs)
OUTPUT_ROOT = WORKSPACE / "output"  # reduced datasets, one folder per target
INJECT_ROOT = OUTPUT_ROOT / "inject"  # this script's clean/ and injected/ trees live here

"""
__The Input Image__

The injection contract is deliberately spartan: a plain 2-D FITS image — finite,
non-negative, **not** PSF-convolved — whose pixel values are flux per pixel at a stated
pixel scale, oriented north-up, centred at the position you inject at. For JWST the
adapter sets `inject_units="Jy"`: pixel values are janskys per pixel, so the **total
source flux is simply the array sum**. No lensing library is imported to build it —
a simulated *input* should be transparent, so we write the arc as three lines of numpy.

We build a partial Einstein arc: an exponential profile in the radial distance from a
ring of radius `r_ring`, tapered azimuthally around an angle `phi_0` — the classic
morphology of a lensed source near a fold:

    I(r, phi) ∝ exp( -|r - r_ring| / w_r ) x exp( -(phi - phi_0)^2 / (2 w_phi^2) )

normalised so the array sums to `flux_jy`. The input grid is finer than the detector
(0.015"/pixel) so the injector — which deposits flux through each frame's own
distortion WCS with a flux-conserving drizzle-style footprint — resolves the arc's
width properly before the frame PSF is applied.
"""
FLUX_JY = 2.0e-6  # total injected flux: 2 microJy (m_AB ~ 25.6) — bright enough to recover cleanly
INPUT_PIXEL_SCALE = 0.015  # "/pixel of the input image — finer than the SW detector
R_RING = 0.9  # arc radius in arcsec
W_R = 0.12  # radial exponential scale of the arc, arcsec
W_PHI = 0.9  # azimuthal Gaussian width, radians (~50 deg of arc)
PHI_0 = np.pi / 3.0  # arc's central position angle

shape = (161, 161)
yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
cy, cx = shape[0] // 2, shape[1] // 2

r = np.hypot(yy - cy, xx - cx) * INPUT_PIXEL_SCALE  # radius from the arc centre, arcsec
phi = np.arctan2(yy - cy, xx - cx)  # azimuth, radians

arc = np.exp(-np.abs(r - R_RING) / W_R) * np.exp(-((phi - PHI_0) ** 2) / (2.0 * W_PHI**2))
arc = FLUX_JY * arc / arc.sum()  # normalise: array sum IS the total flux in Jy

input_dir = INJECT_ROOT
input_dir.mkdir(parents=True, exist_ok=True)
input_path = input_dir / "input_arc_jy.fits"
fits.PrimaryHDU(arc.astype(np.float32)).writeto(input_path, overwrite=True)

print(f"Input arc written to: {input_path.resolve()}")
print(f"Total input flux: {arc.sum():.3e} Jy over {shape[0]}x{shape[1]} px at {INPUT_PIXEL_SCALE}\"/px")

"""
__The Unit Chain__

Here is where JWST injection earns its keep. The `_cal` frames are in MJy/sr, but their
headers carry the two numbers that connect surface brightness back to detector physics:
`PHOTMJSR` (the calibration factor MJy/sr per DN/s) and `PIXAR_SR` (the pixel solid
angle in steradians). The injector converts your Jy-per-pixel input **through each
frame's own keywords** — not a global constant — so a frame with a slightly different
calibration receives a correspondingly different DN pattern, exactly as a real source
on the sky would.

The conversion is constructed so that the injected *mean* is flux-exact: the chain
Jy -> electrons -> DN/s -> MJy/sr algebraically reduces to dividing by
(PIXAR_SR x 1e6), with the detector gain cancelling out of the mean entirely. The gain
enters only one place —

__Noise Bookkeeping__

— the *Poisson draw*. A real source arrives as photons, so the injector realises the
source in electrons and draws Poisson noise on it (seeded deterministically from
`inject_seed` and the frame filename, so reruns are bit-identical). Converting
electrons needs a gain, and NIRCam `_cal` headers do not carry a per-frame one, so the
adapter supplies a nominal `e_per_dn = 2.0`. Because the mean is gain-free, this
approximation **shapes only the width of the injected Poisson scatter**, not the
recovered flux — and the provenance discloses exactly that, recording the nominal gain
and the caveat alongside every injection.

Equally important: the injector updates each frame's **ERR array in quadrature** with
the injected source's variance *before* calwebb_image3 runs. The variance planes then
propagate through resample like every other noise source, so the final `noise_map.fits`
of the injected reduction correctly knows the arc is there. An exposure without an ERR
array refuses injection loudly — silent noise-bookkeeping gaps are how simulations lie.

__Clean and Injected Reductions__

Two runs of the identical spec — the only difference is the three `inject_*` dials on
the second. We inject 3" east of the ring so the recovered arc lands on clean sky in
the same cutout, and use F150W (SW, 0.03"/pixel, 419x419 — the same COSMOS-Web
conventions as `multi_band.py`, whose cache this shares).
"""
RA, DEC = 150.10048, 1.89301  # the COSMOS-Web ring field (Mercier et al. 2024)
band = "F150W"
adapter = nircam_adapter_for_filter(band)

OFFSET_ARCSEC = 3.0
inject_ra = RA + OFFSET_ARCSEC / 3600.0  # 3" offset in RA (dec ~ 1.9 deg, cos(dec) ~ 1)

common = dict(
    name=f"cosmos_web_ring_{band.lower()}",  # same name as multi_band.py -> shares its exposure cache
    ra=RA,  # cutout stays centred on the ring; the injection lands 3" away inside it
    dec=DEC,
    instrument=adapter.key,  # "nircam_sw"
    filter_name=band,
    proposal_ids=("1727",),  # COSMOS-Web only
    final_scale=adapter.recommended_final_scale,  # 0.03"/pixel SW convention
    final_pixfrac=1.0,  # full drizzle drop
    cutout_shape=(419, 419),  # ~12.5" at 0.03"/pixel
)

print(
    """
    Run 1/2: the CLEAN reduction (no injection). With a warm cache from multi_band.py
    this is a combine-only run; cold, it downloads the F150W _cal exposures first.
    Expect tens of minutes per run either way — calwebb_image3 runs twice in this
    script.
    """
)

clean_record = reduce_target(
    TargetSpec(**common), cache_root=CACHE_ROOT, output_root=INJECT_ROOT / "clean"
)

print("Run 2/2: the INJECTED reduction (same spec + inject_* dials).")

injected_record = reduce_target(
    TargetSpec(
        **common,
        inject_image=str(input_path),  # the Jy-per-pixel arc built above (absolute path)
        inject_pixel_scale=INPUT_PIXEL_SCALE,  # the input image's own pixel scale
        inject_position=(inject_ra, DEC),  # absolute (ra, dec) where the arc centre lands
        inject_seed=0,  # deterministic Poisson realisation (per-frame seeds derive from this + filename)
    ),
    cache_root=CACHE_ROOT,
    output_root=INJECT_ROOT / "injected",
)

clean_dir = INJECT_ROOT / "clean" / common["name"]
injected_dir = INJECT_ROOT / "injected" / common["name"]

"""
__The Inject Provenance Block__

A semi-synthetic dataset that could pass for real data is a scientific hazard, so the
injected reduction's provenance carries an `inject` block stating exactly what was
added: the input image and its units and pixel scale, the total input flux, the
position, the PSF source used for the convolution (each frame's own Tier-1 ePSF unless
you supplied `inject_psf`), the seed, the realised total electrons, and a per-frame
list. The injected FITS frames themselves also carry INJECTED/INJIMG/INJSEED header
keys — the semi-synthetic status is stamped on every layer of the output.
"""
inject_block = injected_record["inject"]

print("Inject block:")
print(json.dumps({k: v for k, v in inject_block.items() if k != "frames"}, indent=2))
print(f"Frames injected: {len(inject_block['frames'])}")

"""
__Recovery__

Now the test. The difference of the injected and clean mosaics isolates the arc (plus
the injected Poisson scatter); summing the difference in an aperture around the
injection position gives the recovered surface brightness, and the conversion back to
flux is the MJy/sr bookkeeping from `start_here.py`:

    flux [Jy] = sum(SB [MJy/sr]) x Omega_pixel [sr] x 1e6,  Omega_pixel = (scale / 206265)^2

The noise prediction for the same aperture comes from the injected reduction's own
noise map, summed in quadrature — so this one number tests flux conservation through
acquisition, injection, image3, the ERR propagation and the packaging all at once.
"""
data_clean = fits.getdata(clean_dir / "data.fits").astype(float)
data_injected = fits.getdata(injected_dir / "data.fits").astype(float)
noise_injected = fits.getdata(injected_dir / "noise_map.fits").astype(float)
header = fits.getheader(injected_dir / "data.fits")

diff = data_injected - data_clean  # MJy/sr — the recovered arc

pixel_scale = injected_record["package"]["pixel_scale"]
omega = (pixel_scale / 206265.0) ** 2  # pixel solid angle in steradians

x_inj, y_inj = WCS(header).world_to_pixel_values(inject_ra, DEC)
yy, xx = np.mgrid[0 : diff.shape[0], 0 : diff.shape[1]]
aperture = np.hypot(yy - y_inj, xx - x_inj) * pixel_scale <= 2.0  # 2" radius around the arc

good = aperture & (noise_injected < 1.0e7)  # exclude masked-by-noise pixels from the sums

recovered_jy = float(diff[good].sum()) * omega * 1e6
aperture_noise_jy = float(np.sqrt((noise_injected[good] ** 2).sum())) * omega * 1e6

report = {
    "injected_flux_jy": FLUX_JY,
    "recovered_flux_jy_2arcsec": recovered_jy,
    "recovery_ratio": recovered_jy / FLUX_JY,
    "aperture_noise_jy": aperture_noise_jy,
    "total_injected_e": inject_block["total_injected_e"],
    "n_frames_injected": len(inject_block["frames"]),
}

report_path = INJECT_ROOT / "recovery_report.json"
report_path.write_text(json.dumps(report, indent=2))

print(json.dumps(report, indent=2))
print(f"Recovery report saved to: {report_path.resolve()}")

ok = abs(report["recovery_ratio"] - 1.0) < max(0.05, 3.0 * aperture_noise_jy / FLUX_JY)
print("RECOVERY", "OK" if ok else "DISCREPANT — inspect the report and the inject block")

"""
A recovery ratio within a few percent of 1 (or within 3x the aperture noise for faint
injections) is the pass condition — the same acceptance the **PyAutoReduce** JWST
injection validation uses. A systematic shortfall would point at flux non-conservation
somewhere in the chain (footprint clipping of the input, aperture too small for the
convolved arc, resample kernel effects); the per-frame `inject` provenance is where
the diagnosis starts.

__Plots__

Clean, injected and difference — the difference panel should show the arc alone,
sitting on (correlated) noise, exactly 3" east of the ring.
"""
plot_dir = INJECT_ROOT / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

scale = np.nanpercentile(data_clean, 99.5)
for ax, (title, image) in zip(
    axes,
    (("clean", data_clean), ("injected", data_injected), ("difference", diff)),
):
    ax.imshow(np.arcsinh(image / (0.05 * scale)), origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plot_path = plot_dir / "injection_recovery.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)

print(f"Recovery plot saved to: {plot_path.resolve()}")

"""
__Wrap Up__

You built a lensed arc with known flux in pure numpy, injected it into real NIRCam
`_cal` exposures through each frame's own calibration keywords and PSF (with its
Poisson noise realised and its variance propagated into ERR), ran the identical
calwebb_image3 reduction twice, and recovered the flux to within the noise — a
closed-loop test of the entire pipeline, on real detector data, with every synthetic
ingredient disclosed in provenance.

From here, the natural next step is science-grade simulation: replace the analytic arc
with a ray-traced source image (built in **PyAutoLens** and saved to FITS — still just
a Jy-per-pixel array as far as the injector is concerned) and you have realistic mock
JWST lenses embedded in genuine COSMOS-Web noise, ready for recovery tests of the full
modeling chain.

The following locations of the workspace are good places to checkout next:

- `scripts/hst_acs/simulator.py`: the HST injection sibling — e-/s units, driz_cr interactions.
- `scripts/jwst_nircam/start_here.py`: the units and provenance groundwork this script builds on.
- `scripts/jwst_nircam/individual.py`: injected frames also flow into frame products — semi-synthetic frames stay stamped.
- `autolens_workspace/scripts/imaging/simulator.py`: building ray-traced source images to feed this injector.

__Env__ (Developer Only)

ENV: network
"""
