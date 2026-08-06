"""
ALMA Simulator: simobserve
==========================

Every reduction pipeline needs a ground truth it can be tested against, and for ALMA the
observatory itself provides the canonical route: CASA's `simobserve` task turns a model
sky image into a measurement set of simulated visibilities for a chosen antenna
configuration, observing time and atmosphere — including realistic thermal noise
(https://casadocs.readthedocs.io/en/stable/notebooks/simulation.html).

This script builds a lensed-source-like image in pure numpy (Jy per pixel — no lensing
library needed), hands it to the **PyAutoReduce** visibility branch's simobserve
acquire-alternative, and pushes the simulated measurement set through the *identical*
split / extract / assemble / package chain the real G09v1.40 data traverses in
`start_here.py`. Because you set the input flux, you can close the loop: the recovered
short-baseline flux must match what you injected. Budget ~15 minutes; the simulation
itself takes a couple of minutes of CASA time.

__Contents__

- **The Simulation Route:** Why simobserve is the canonical realistic-ALMA simulation.
- **Imports:** Import **PyAutoReduce** and check for modular CASA.
- **Paths:** Anchor every path to the workspace root.
- **The Source Image:** A ring + core in Jy/pixel, from a formula, in pure numpy.
- **Sky Model:** The 4-axis FITS sky model simobserve consumes.
- **Simulation Dials:** The `TargetSpec` with every `alma_sim_*` dial explained.
- **Run The Simulation:** simobserve, then the identical visibility chain.
- **The Inject Block:** The provenance that marks this dataset as synthetic.
- **Flux Recovery:** Close the loop — shortest-baseline flux vs the injected total.
- **UV Coverage:** The simulated array's sampling pattern.
- **Load In PyAutoLens:** The dirty image of your own simulated source.
- **Real-MS Injection:** What is deliberately deferred, and the literature route it maps to.
- **Wrap Up:** Summary and good places to check out next.

__The Simulation Route__

For the imaging instruments, **PyAutoReduce**'s simulation strategy is *injection into
real frames*: real exposures carry cosmic rays, sky, bad pixels and PSF wings for free,
and no maintained raw-frame simulator exists for HST/Keck imaging anyway. ALMA is the one
instrument where the calculus flips: `simobserve` is observatory-supported, actively
maintained, and models the things that matter — the array configuration's uv sampling,
the integration cadence, and atmospheric + system-temperature thermal noise via a pwv
(precipitable water vapour) parameter. So for the visibility branch, fully-synthetic
simulation *is* the canonical route, and `inject_image` on an ALMA `TargetSpec` switches
acquisition to simobserve entirely — no archive, no uids, no calibrated MS required.

The downstream stages neither know nor care: a simulated MS is split, extracted,
assembled and packaged by exactly the code paths as real data, which is what makes this a
test of the pipeline and not just of the simulator.

__Imports__

`skymodel_fits` and `simobserve_kwargs` are public helpers of the simulate stage — we
demonstrate both standalone below before letting `reduce_target` drive them for real.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoreduce import TargetSpec, reduce_target
from autoreduce.visibilities.simulate import simobserve_kwargs, skymodel_fits

try:
    import casatools  # noqa: F401
    import casatasks  # noqa: F401  (provides simobserve and split)
except ImportError:
    print(
        "Modular CASA is not installed. The simulator needs the pip-installable "
        "CASA packages:\n\n    pip install casatools casatasks\n\nExiting cleanly."
    )
    sys.exit(0)

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the
folder containing `scripts/`). **PyAutoReduce** requires absolute paths: simobserve
writes its project directory relative to the working directory, and the pipeline manages
that by changing directory into scratch internally — relative paths would break.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_ROOT = WORKSPACE / "cache"
OUTPUT_ROOT = WORKSPACE / "output"

NAME = "alma_sim_ring"
RA, DEC = 137.0, 2.1        # an arbitrary field visible from the ALMA site

out_dir = OUTPUT_ROOT / NAME
out_dir.mkdir(parents=True, exist_ok=True)

"""
__The Source Image__

The input is a plain 2-D numpy array in **Jy per pixel** — the same flux contract as JWST
injection, and deliberately free of any lensing-library dependency. To evoke what ALMA
actually sees toward a lensed DSFG — an Einstein ring with a compact bright region — we
compose two analytic pieces on a radial grid r (arcsec from centre):

- a thin ring:      I_ring(r) = exp( -(r - r_ring)^2 / (2 w^2) ), with r_ring = 0.45",
  w = 0.06" — the smeared image of a compact source near the caustic;
- a compact core:   I_core(r) = exp( -1.678 r / r_eff ), an exponential (Sersic n=1)
  profile with r_eff = 0.08", where 1.678 is the n=1 Sersic constant that makes r_eff the
  half-light radius.

The sum is normalised so the array total equals `TOTAL_FLUX_JY` exactly — the number the
recovery check at the end must reproduce. At 0.02"/pixel the 129x129 field spans 2.6",
comfortably containing the ring.
"""
TOTAL_FLUX_JY = 0.02        # total source flux (typical of a bright lensed DSFG continuum)
PIXEL_SCALE = 0.02          # arcsec / pixel of the input image
SHAPE = (129, 129)

yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
cy, cx = SHAPE[0] // 2, SHAPE[1] // 2
r = np.hypot(yy - cy, xx - cx) * PIXEL_SCALE

ring = np.exp(-((r - 0.45) ** 2) / (2.0 * 0.06**2))     # thin Einstein-ring-like annulus
core = np.exp(-1.678 * r / 0.08)                        # compact exponential core
image = ring + 0.5 * core
image = TOTAL_FLUX_JY * image / image.sum()             # normalise: sum(image) == TOTAL_FLUX_JY

from astropy.io import fits

input_path = out_dir / "input_ring_jy.fits"
fits.PrimaryHDU(image.astype(np.float32)).writeto(input_path, overwrite=True)
print(f"Wrote input source image ({image.sum():.4f} Jy total) to {input_path}")

plt.figure(figsize=(5, 5))
plt.imshow(np.arcsinh(image / image.max() * 100.0), origin="lower", cmap="magma")
plt.title(f"input source ({TOTAL_FLUX_JY} Jy, Jy/pixel)")
input_png = out_dir / "plots"
input_png.mkdir(exist_ok=True)
input_png = input_png / "input_source.png"
plt.savefig(input_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved input-source plot to {input_png.resolve()}")

"""
__Sky Model__

simobserve does not consume a bare 2-D image: it wants a FITS cube with four world axes —
RA---SIN, DEC--SIN, STOKES, FREQ — carrying the pixel scale, sky position and observing
frequency in its WCS, with BUNIT = Jy/pixel. The public `skymodel_fits` helper performs
exactly that wrapping (the pipeline calls it internally); we demonstrate it standalone so
you can inspect the header it builds. Likewise `simobserve_kwargs` shows you the exact
headless simobserve call the pipeline will make, as a pure, inspectable dict — no
surprises hidden behind the one-call API.
"""
demo_skymodel = skymodel_fits(
    input_image=image,                  # the Jy/pixel array above
    pixel_scale=PIXEL_SCALE,            # arcsec / pixel, written into the WCS
    ra=RA,                              # sky position of the model centre (deg)
    dec=DEC,
    freq_ghz=230.0,                     # observing frequency for the FREQ axis
    out_path=out_dir / "skymodel_demo.fits",
)
header = fits.getheader(demo_skymodel)
print(
    f"skymodel axes: {[header[f'CTYPE{i}'] for i in (1, 2, 3, 4)]}, "
    f"BUNIT={header['BUNIT']!r}"
)

"""
__Simulation Dials__

The `alma_sim_*` dials on `TargetSpec` map one-to-one onto simobserve's own parameters.
Setting `inject_image` (plus its mandatory `inject_pixel_scale`) on an ALMA spec is the
switch that replaces archive acquisition with simulation — `alma_uids` / `alma_field` /
`alma_spws` are not required in this mode, because the simulated MS has exactly one field
and one spectral window.

The `alma_sim_pwv_mm` dial deserves the highlight: it sets the precipitable water vapour
of the simulated atmosphere, which (with the system temperature model) sets the thermal
noise level. **`alma_sim_pwv_mm=0` disables noise entirely** — a noiseless simulation,
invaluable when you want to test the chain's arithmetic (flux recovery below becomes
exact to numerical precision) rather than its statistics.
"""
spec = TargetSpec(
    name=NAME,                              # products land at output/<name>/
    ra=RA,                                  # phase centre of the simulated observation (deg)
    dec=DEC,
    instrument="alma",                      # the visibility-domain adapter
    inject_image=str(input_path),           # the Jy/pixel source image -> switches acquire to simobserve
    inject_pixel_scale=PIXEL_SCALE,         # arcsec / pixel of that image (required with inject_image)
    alma_sim_antennalist="alma.cycle8.3.cfg",  # antenna configuration file (a mid-compact 12-m config)
    alma_sim_totaltime_s=600.0,             # total on-source observing time (s)
    alma_sim_integration_s=10.0,            # correlator integration (dump) time (s)
    alma_sim_freq_ghz=230.0,                # observing frequency (Band 6 continuum)
    alma_sim_pwv_mm=0.5,                    # atmosphere: 0.5 mm pwv thermal noise (0 = noiseless)
)

"""
__Run The Simulation__

One call. Under the hood: `skymodel_fits` wraps the image, simobserve synthesises the
observation into a `<name>_sim` project directory in the work area, and then — this is
the point — the *same* split, extract, assemble and package stages as `start_here.py`
turn the simulated MS into the product triplet. The simulated dataset gets uid "sim",
field "0", spw "0" in provenance and sidecar names.
"""
print(
    "\nRunning simobserve + the visibility chain (a couple of minutes of CASA "
    "time for the simulation; the chain itself takes seconds)..."
)

record = reduce_target(
    spec,
    cache_root=CACHE_ROOT,      # unused in simulation mode (nothing is downloaded)
    output_root=OUTPUT_ROOT,    # products land at output/alma_sim_ring/
)

print(f"\nSimulation + reduction complete. Products in {out_dir}")
print(f"packaged {record['package']['n_visibilities']} visibilities")

"""
__The Inject Block__

A synthetic dataset must never masquerade as real data. The provenance record therefore
carries an `inject` block stating the source ("simobserve"), the sky model, the injected
total flux, and every simulation dial — including whether thermal noise was on. Any
consumer of `reduction.json` can (and should) check for this block before treating a
dataset as an observation.
"""
print("\ninject block:")
print(json.dumps(record["inject"], indent=2))

"""
__Flux Recovery__

The closure test. For a compact source, the visibility amplitude at the shortest
baselines approaches the total flux: a baseline much shorter than 1/theta_source barely
resolves the source, so its visibility's real part is (nearly) the full flux density.
We therefore average the real part over the shortest 5% of baselines and compare it to
the injected total. With thermal noise at pwv 0.5 mm and 10 minutes of integration,
agreement to within a few percent is expected; re-run with `alma_sim_pwv_mm=0.0` and the
ratio snaps to 1 at numerical precision.
"""
visibilities = fits.getdata(out_dir / "data.fits")
uv_wavelengths = fits.getdata(out_dir / "uv_wavelengths.fits")
noise_map = fits.getdata(out_dir / "noise_map.fits")

uv_dist = np.hypot(uv_wavelengths[:, 0], uv_wavelengths[:, 1])
short = uv_dist < np.percentile(uv_dist, 5.0)
recovered_jy = float(np.mean(visibilities[short, 0]))

print(f"\ninjected total flux   : {TOTAL_FLUX_JY:.4f} Jy")
print(f"short-baseline <Re(V)>: {recovered_jy:.4f} Jy")
print(f"recovery ratio        : {recovered_jy / TOTAL_FLUX_JY:.3f}")

"""
__UV Coverage__

The uv coverage here is not the G09v1.40 coverage — it is whatever the chosen antenna
configuration and hour-angle range produced, which is precisely why simulation is useful:
change `alma_sim_antennalist` to a more extended configuration and watch the coverage
(and the resolution of the dirty image below) transform, before you ever propose for the
real thing.
"""
plot_dir = out_dir / "plots"
u_klambda, v_klambda = uv_wavelengths[:, 0] / 1e3, uv_wavelengths[:, 1] / 1e3
plt.figure(figsize=(6, 6))
plt.scatter(u_klambda, v_klambda, s=0.5, lw=0, alpha=0.5)
plt.scatter(-u_klambda, -v_klambda, s=0.5, lw=0, alpha=0.5)
plt.xlabel(r"u [k$\lambda$]")
plt.ylabel(r"v [k$\lambda$]")
plt.gca().set_aspect("equal")
plt.title(f"simulated uv coverage ({visibilities.shape[0]} visibilities)")
uv_png = plot_dir / "uv_coverage.png"
plt.savefig(uv_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved uv-coverage plot to {uv_png.resolve()}")

"""
__Load In PyAutoLens__

The consumer-side round trip, exactly as for real data — which is the point: a simulated
dataset that loads and images identically to a real one is a simulated dataset you can
use to rehearse an entire modeling workflow. The dirty image should show your ring.
"""
try:
    import autolens as al
except ImportError:
    al = None
    print(
        "\nPyAutoLens is not installed, so the dirty-image check is skipped "
        "(pip install autolens). The simulated dataset itself is complete."
    )

if al is not None:
    real_space_mask = al.Mask2D.circular(
        shape_native=(128, 128),    # matches the 2.6" input field at 0.02"/pixel
        pixel_scales=0.02,
        radius=1.2,                 # arcsec — contains the 0.45"-radius ring
    )
    dataset = al.Interferometer.from_fits(
        data_path=out_dir / "data.fits",
        noise_map_path=out_dir / "noise_map.fits",
        uv_wavelengths_path=out_dir / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerNUFFT,
    )
    dirty = np.asarray(dataset.dirty_image.native)
    peak_over_rms = float(np.max(np.abs(dirty)) / np.std(dirty))
    print(f"\ndirty-image peak/rms = {peak_over_rms:.1f}")

    plt.figure(figsize=(6, 6))
    plt.imshow(dirty, origin="lower", cmap="magma")
    plt.colorbar(label="dirty-image intensity")
    plt.title(f"simulated ring: dirty image (peak/rms = {peak_over_rms:.1f})")
    dirty_png = plot_dir / "dirty_image.png"
    plt.savefig(dirty_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved dirty-image plot to {dirty_png.resolve()}")

"""
__Real-MS Injection__

Honesty section. There is a second simulation mode in the literature: predicting model
visibilities at the (u, v) points of a **real** measurement set and adding them to (or
replacing) its data — the interferometric analogue of source injection into real imaging
frames, with the real observation's calibration systematics carried along for free. CASA
supports the ingredients (the guide "Fit an arbitrary sky model to an existing MS"
documents predicting a model into MODEL_DATA and manipulating the data columns), and
published weight checks are built on similar machinery.

**PyAutoReduce** defers this mode deliberately: it needs Fourier prediction at arbitrary
uv points and phase-centre bookkeeping that simobserve mode does not, and simobserve
already answers the pipeline-validation question this script exists for. Only the
simobserve route exists today — do not look for a real-MS injection dial, there isn't
one.

__Wrap Up__

You simulated an ALMA observation of a ring + core source you specified to the microjansky,
watched the identical visibility chain reduce it, verified the injected flux came back at
the shortest baselines, and imaged the result through **PyAutoLens**.

Good places to checkout next:

- `scripts/alma/start_here.py` — the same chain on real G09v1.40 data.
- `scripts/alma/step_by_step.py` — each stage by hand, including the smearing budget and
  the weight audit (both equally applicable to simulated data — try them on this MS).
- `autolens_workspace/scripts/interferometer/` — fit a lens model to the visibilities you
  just simulated.
"""

"""
__Env__ (Developer Only)

ENV: network
"""
