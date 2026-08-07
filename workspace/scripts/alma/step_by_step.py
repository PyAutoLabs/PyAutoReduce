"""
ALMA Step By Step: Measurement Set To Visibilities
==================================================

`start_here.py` ran the **PyAutoReduce** visibility branch as one `reduce_target` call.
This script runs the same chain by hand, stage by stage, on the same G09v1.40 dataset —
because for ALMA, unlike the imaging instruments, every stage is public API: the split,
extract and assemble modules are importable functions you can compose yourself.

Along the way you will learn what a measurement set actually is, why the pipeline runs on
headless "modular" CASA, how far you may average channels before smearing bites (with the
limit computed for this dataset), and — most importantly — what the WEIGHT column really
contains and how to verify it against the visibility scatter before you trust it as your
noise map. Budget ~30 minutes of reading; the compute re-uses `start_here.py`'s split
measurement sets where present.

__Contents__

- **Measurement Set Anatomy:** The tables and columns inside an MS directory.
- **Headless CASA:** Why the pipeline uses pip-installed casatools/casatasks, not the CASA shell.
- **Imports:** Import the visibility-branch modules directly.
- **Paths:** Anchor every path to the workspace root.
- **Calibrated Input:** Resolve the calibrated measurement sets (or exit with guidance).
- **Field Split:** `casatasks.split` pass one — isolate the science field.
- **Channel Averaging Split:** `casatasks.split` pass two — average channels by `width`.
- **Smearing Limits:** Compute the bandwidth-smearing cost of the chosen width (Bridle & Schwab).
- **Extract:** `casatools.table` reads into the `MsColumns` contract.
- **Weights:** What WEIGHT means, why archival weights need scrutiny, the statwt story.
- **Weight Verification:** Check sigma = 1/sqrt(WEIGHT) against the visibility scatter.
- **UV Wavelengths:** Baselines in metres to (u, v) in wavelengths, per channel.
- **Stokes I:** The weighted polarization combine, and zero-weight rows dropped loudly.
- **Concatenate And Package:** Assemble the blocks and write the product triplet.
- **Continuum Only:** What this branch deliberately does not do yet.
- **Wrap Up:** Summary and good places to check out next.

__Measurement Set Anatomy__

A measurement set (MS) is not a file — it is a directory of binary tables (the CASA table
system), and it is the container every ALMA delivery ultimately becomes. The main table
has one row per (baseline, integration timestamp), with the columns this pipeline reads:

- `DATA` — the complex visibilities, shaped (n_polarizations, n_channels, n_rows). A
  calibrated `.ms.split.cal` delivery carries the calibrated science data here. (A raw MS
  being calibrated grows a `CORRECTED_DATA` column beside it, and modeling workflows add
  `MODEL_DATA`; after the observatory's final split, `DATA` is the calibrated column.)
- `UVW` — the projected baseline vector in metres, (3, n_rows).
- `WEIGHT` — the noise bookkeeping, (n_polarizations, n_rows): nominally 1/sigma^2 per
  complex visibility. Its sibling `SIGMA` describes the *raw* data's per-datum noise; for
  calibrated data `WEIGHT` is the authoritative one (more below).
- `ANTENNA1`, `ANTENNA2`, `TIME`, `SCAN_NUMBER` — which baseline, when, in which scan.

Around the main table sit subtables: `SPECTRAL_WINDOW` (channel frequencies and widths
per spw — a typical ALMA observation carries four spws across the band), `ANTENNA`,
`FIELD` (the science target *and* the calibrators share one MS until you split), and
more. The full data model is documented at https://casadocs.readthedocs.io/ — see the
Measurement Set pages and the Data Weights notebook referenced below.

__Headless CASA__

Historically this extraction recipe ran inside the interactive `casa` shell, with `tb`
and `split` existing only as globals injected into that session — fine at the telescope,
hopeless for a scripted, testable pipeline. CASA 6 solved this by shipping **modular**
packages on PyPI: `casatools` (the `table` tool and friends) and `casatasks` (`split`,
`simobserve`, ...), plain-Python importable and proven headless
(https://casadocs.readthedocs.io/).

**PyAutoReduce** builds only on the modular route, with one careful boundary: extraction
of an *already-calibrated* MS has no CASA-version coupling — any recent modular CASA
works. The notorious version-matching constraint binds only the `scriptForPI.py`
calibration restore (see `start_here.py`, "The Archive Reality"), which is exactly the
step the pipeline leaves manual rather than pretending to automate.

__Imports__

Because the visibility branch's stages are public modules, we import them directly:
`split` (the casatasks orchestration), `extract` (the casatools table reads), `assemble`
(pure numpy) and the interferometer packaging. casatools/casatasks themselves are
imported lazily inside those modules; we check for them up front to fail friendly.
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autoreduce.acquire import alma as alma_acquire
from autoreduce.package.interferometer import write_products
from autoreduce.visibilities.assemble import (
    assemble_ms_products,
    concatenate,
    stokes_i_combine,
    uv_wavelengths_from_uvw,
)
from autoreduce.visibilities.extract import columns_from, getcol, num_channels_per_spw
from autoreduce.visibilities.split import resolve_width, split_field, split_spw

try:
    import casatools  # noqa: F401  (the `table` tool behind the extract stage)
    import casatasks  # noqa: F401  (the `split` task behind the split stage)
except ImportError:
    print(
        "Modular CASA is not installed (pip install casatools casatasks). "
        "Exiting cleanly."
    )
    sys.exit(0)

"""
__Paths__

Reductions read and write real data, so we anchor every path to the workspace root (the
folder containing `scripts/`). **PyAutoReduce** requires absolute paths: CASA tasks write
scratch and logs relative to the working directory, so relative paths would break.

This script works in its own output folder, but points its split work directory at the
one `start_here.py` uses — the splits are idempotent, so if you ran `start_here.py` first
the CASA passes below complete instantly by reusing its measurement sets.
"""
WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = WORKSPACE / "output"

NAME = "alma_g09v140"
PROJECT = "2016.1.00282.S"
FIELD = "G09v1.40"
UIDS = ("A002_Xb9b1b9_X3046", "A002_Xb99cbd_X2456")
SPWS = ("1", "2")
WIDTH = 240

WORK_DIR = OUTPUT_ROOT / NAME / "work"          # shared with start_here.py (idempotent splits)
STEPS_OUT = OUTPUT_ROOT / f"{NAME}_steps"       # this script's own packaged products
WORK_DIR.mkdir(parents=True, exist_ok=True)

"""
__Calibrated Input__

As in `start_here.py`: the canonical input is a directory of calibrated measurement sets
(`uid___<uid>.ms.split.cal`), obtained via an ARC delivery or a local scriptForPI restore
at the matching CASA version. If it is absent we print the pipeline's guidance and exit
cleanly. To keep the walk-through readable we work on the *first* execution block only;
the full pipeline simply loops what you see here over every (uid, spw).
"""
ALMA_MS_DIR = Path(
    os.environ.get(
        "AUTOREDUCE_ALMA_MS_DIR", WORKSPACE / "dataset" / "alma" / "g09v140_calibrated"
    )
)

if not ALMA_MS_DIR.is_dir():
    print(
        f"No calibrated measurement-set directory at {ALMA_MS_DIR} "
        f"(or set AUTOREDUCE_ALMA_MS_DIR).\n"
    )
    print(alma_acquire.restore_guidance(PROJECT, WORKSPACE / "cache" / NAME / "tarballs"))
    print("\nExiting cleanly — nothing was written.")
    sys.exit(0)

try:
    ms_paths = alma_acquire.resolve_calibrated_ms(ALMA_MS_DIR, UIDS)
except FileNotFoundError as error:
    print(f"{error}\n\nExiting cleanly.")
    sys.exit(0)

uid = UIDS[0]
parent_ms = ms_paths[0]
print(f"Working execution block: {parent_ms.name}")

"""
Before splitting anything, interrogate the parent MS's spectral layout. `NUM_CHAN` per
spectral window is what the `width=0` continuum default reads to collapse each spw fully;
here we print it so the `width=240` choice below is transparent — spws 1 and 2 each carry
240 channels, so one width value collapses each to a single continuum channel.
"""
num_chan = num_channels_per_spw(parent_ms)
print(f"NUM_CHAN per spectral window: {num_chan.tolist()}")

"""
__Field Split__

Pass one of `casatasks.split` isolates the science field. A delivered MS interleaves the
science target with its calibrators — bandpass (a bright quasar), the interleaved phase
calibrator, the flux standard — because that is how the observation was scheduled. The
calibration solved from those sources is already *applied* to the data (this is a
calibrated MS); their visibilities themselves are not lens-modeling data, so the first
split keeps only rows whose field is G09v1.40.

Two dials on every split in this pipeline deserve their comment:

- `datacolumn="data"` — calibrated `.ms.split.cal` deliveries carry the calibrated
  visibilities in `DATA` (the observatory's own final split already moved them there).
- `keepflags=False` — rows flagged bad by the calibration pipeline are *dropped*, not
  carried as zero-weight placeholders. Flagged placeholder rows are a classic source of
  silent dataset bloat and downstream weight confusion; the pipeline refuses to carry
  them.

The split is idempotent: if the output MS already exists (e.g. from `start_here.py`), it
is reused. Interrupted runs are safe too — the split writes to a `.partial` directory
renamed into place only on success.
"""
print(f"\nSplitting field {FIELD!r} out of {parent_ms.name} (idempotent)...")
field_ms = split_field(parent_ms, uid, FIELD, WORK_DIR)
print(f"    field MS: {field_ms.name}")

"""
__Channel Averaging Split__

Pass two splits out each spectral window and averages its channels by `width`. This is
the step that makes uv-plane lens modeling computationally tractable: a raw dataset holds
~10^7-10^9 visibilities (rows x channels), and every likelihood evaluation must Fourier
transform the model to all of them. Published practice has always averaged first —
Hezaveh et al. 2016 (https://arxiv.org/abs/1601.01388, section 2) averaged the SDP.81
long-baseline data in exactly this way before their subhalo search — although modern
NUFFT solvers can now handle the un-averaged problem when needed (Powell et al. 2021,
https://arxiv.org/abs/2005.03609).

`resolve_width` turns the user dial into a concrete per-spw width: a positive value
passes through, `0` means "collapse the whole spw" by reading its channel count.
"""
spw_ms_by_spw = {}
for spw in SPWS:
    width = resolve_width(WIDTH, spw, num_chan)
    print(f"Splitting spw {spw} at width {width} (idempotent)...")
    spw_ms_by_spw[spw] = split_spw(field_ms, uid, FIELD, spw, width, WORK_DIR)
    print(f"    spw MS: {spw_ms_by_spw[spw].name}")

"""
__Smearing Limits__

Averaging is not free. Averaging channels assigns one mean frequency to visibilities that
were measured across a band of frequencies — but (u, v) scales with frequency, so the
average smears the response to emission *away from the phase centre* radially outward.
This is **bandwidth smearing**; its time-domain sibling (averaging integrations while the
Earth rotates the baselines) smears azimuthally. The standard reference is Bridle &
Schwab 1999 (ASP Conf. Ser. 180, 371): to first order the radial smearing extent for a
source at offset theta from the phase centre is

    delta_theta ~ (delta_nu / nu) * theta

and it decorrelates (attenuates) long-baseline amplitudes when delta_theta becomes
comparable to the synthesized beam. Since lensed arcs sit ~1 arcsecond off-centre, this
hard-limits how far you may average on long-baseline data.

Let's compute the worked exercise for *this* dataset and *this* width choice: read the
averaged channel width from the split MS, take the observing frequency and the longest
baseline, and compare the smearing extent at a 1" offset against the synthesized-beam
scale ~ 1/(max uv distance).
"""
OFFSET_ARCSEC = 1.0  # a typical arc offset from the phase centre
ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0

for spw, spw_ms in spw_ms_by_spw.items():
    chan_width_hz = float(
        np.mean(np.abs(np.atleast_1d(np.squeeze(getcol(spw_ms, "SPECTRAL_WINDOW", "CHAN_WIDTH")))))
    )
    chan_freq_hz = np.atleast_1d(
        np.squeeze(getcol(spw_ms, "SPECTRAL_WINDOW", "CHAN_FREQ"))
    ).astype(float)
    nu0_hz = float(np.mean(chan_freq_hz))
    uvw = getcol(spw_ms, "", "UVW").astype(float)
    max_uv_wavelengths = float(np.max(np.hypot(uvw[0], uvw[1])) * nu0_hz / 299792458.0)

    beam_arcsec = (1.0 / max_uv_wavelengths) * ARCSEC_PER_RAD
    smear_arcsec = (chan_width_hz / nu0_hz) * OFFSET_ARCSEC
    print(
        f"\nspw {spw}: averaged channel width = {chan_width_hz / 1e6:.1f} MHz at "
        f"{nu0_hz / 1e9:.1f} GHz (fractional bandwidth {chan_width_hz / nu0_hz:.2e})"
    )
    print(
        f"    longest baseline = {max_uv_wavelengths / 1e3:.0f} klambda -> beam scale "
        f"~ {beam_arcsec:.3f} arcsec"
    )
    print(
        f"    radial smearing at {OFFSET_ARCSEC:.1f}\" offset ~ {smear_arcsec:.4f} arcsec "
        f"= {smear_arcsec / beam_arcsec:.2f} of the beam scale"
    )

"""
Read the last line for each spw: if the smearing extent is a small fraction of the beam
scale, the width choice is safe for emission at that offset; as it approaches unity you
are actively blurring your arcs and biasing the lens model. Re-run this cell with your
own `width` and offset before averaging your own data — this five-line estimate is the
difference between a deliberate averaging choice and a silent resolution loss.

(Time averaging — `timebin` in CASA's split/mstransform — trades against the azimuthal
version of the same limit. **PyAutoReduce** deliberately exposes no time-averaging dial
yet; channel collapse alone tames the anchor dataset, and a time dial without its
smearing guard would invite exactly the silent loss described above.)

__Extract__

With the per-spw measurement sets on disk, extraction is a set of `casatools.table`
column reads — no CASA tasks, no shell, just arrays. `columns_from` returns the frozen
`MsColumns` contract the assemble stage consumes, with shapes normalised so continuum
(one channel) and line (many channels) widths flow through one code path:

- `data`    complex (n_pol, n_chan, n_rows)
- `uvw`     metres  (3, n_rows)
- `weight`  1/sigma^2 (n_pol, n_rows)
- `chan_freq` Hz (n_chan,)
- `antenna1` / `antenna2` / `time` / `scan` (n_rows,)

The antenna/time/scan columns are the diagnostic paper trail: the pipeline packages them
as per-block sidecars (`antennas_*.fits`, `scans_*.fits`, `times_*.fits`,
`frequencies_*.fits`) so a bad scan or antenna can be traced back after the fact. A
worked example of exactly this kind of extraction is the visread documentation
(https://mpol-dev.github.io/visread/), whose output arrays are precisely the contract
**PyAutoLens** consumes.
"""
columns_by_spw = {}
for spw, spw_ms in spw_ms_by_spw.items():
    columns = columns_from(spw_ms)
    columns_by_spw[spw] = columns
    n_pol, n_chan, n_rows = columns.data.shape
    print(
        f"spw {spw}: DATA ({n_pol} pol x {n_chan} chan x {n_rows} rows), "
        f"{np.unique(np.stack((columns.antenna1, columns.antenna2))).size} antennas, "
        f"{np.unique(columns.scan).size} scans"
    )

"""
__Weights__

The WEIGHT column is where uv-plane lens modeling is won or lost, because it becomes the
noise map, and a mis-scaled noise map biases *every posterior width* your model reports.
The convention (casadocs, "Data Weights and Combination" notebook,
https://casadocs.readthedocs.io/) is:

- `WEIGHT` = 1/sigma^2 per complex visibility, describing the calibrated data;
- `SIGMA`  = the per-datum noise of the *raw* data (superseded once calibrated);
- weights are initialised proportional to (channel width x integration time) at import,
  then rescaled by the system temperature and gain solutions when calibration is applied
  with `calwt=True`; per-channel variants live in `WEIGHT_SPECTRUM`.
- channel averaging in `split` propagates the weights: averaging N channels sums their
  weights, which is why the collapsed continuum visibilities above carry usefully large
  weights.

Why the scrutiny? The ALMA Knowledgebase article "How does CASA calculate the visibility
weights?" documents that the bookkeeping *changed across CASA versions* — only data
calibrated with CASA >= 4.2.2 initialises per-channel weights properly — so for archival
data the weights are guaranteed to be *relative* at best, not absolute 1/sigma^2. The
standard remedy is `statwt`, which recomputes WEIGHT/SIGMA empirically from the scatter
of the (line-free) visibilities themselves. Published uv-plane lens models treat this as
a first-class step: Hezaveh et al. 2016 (section 2) rescaled their weights from the
variance of difference visibilities, and Dye et al. 2018
(https://arxiv.org/abs/1705.05413) recalibrated weights before modeling.

**PyAutoReduce** extracts the weights as delivered and converts sigma = 1/sqrt(WEIGHT) —
it does not silently rescale data it cannot verify. The verification below is how you
check whether *your* MS needs `statwt` before you trust the noise map.

__Weight Verification__

The cleanest empirical estimate of the true noise uses **difference visibilities**
(the Hezaveh et al. 2016 approach): subtract successive integrations on the *same
baseline* within the same scan. The sky (a constant continuum source) cancels in the
difference; what remains is pure noise with variance twice the per-visibility variance.
Comparing that scatter to the WEIGHT-predicted sigma gives a single ratio:

- ratio ~ 1  : the weights are absolute; sigma = 1/sqrt(WEIGHT) is your noise map.
- ratio != 1 : the weights are only relative — run `statwt` (or rescale) before modeling.
"""
for spw, columns in columns_by_spw.items():
    a1, a2, t, scan = columns.antenna1, columns.antenna2, columns.time, columns.scan
    order = np.lexsort((t, scan, a2, a1))  # sort by baseline, then scan, then time
    same_pair = (
        (a1[order][1:] == a1[order][:-1])
        & (a2[order][1:] == a2[order][:-1])
        & (scan[order][1:] == scan[order][:-1])
    )
    data_sorted = columns.data[:, 0, :][:, order]  # first channel, both polarizations
    diff = data_sorted[:, 1:][:, same_pair] - data_sorted[:, :-1][:, same_pair]
    scatter_sigma = float(
        np.std(np.concatenate((diff.real.ravel(), diff.imag.ravel()))) / np.sqrt(2.0)
    )
    w = columns.weight
    predicted_sigma = float(np.mean(1.0 / np.sqrt(w[np.isfinite(w) & (w > 0.0)])))
    print(
        f"spw {spw}: scatter-derived sigma = {scatter_sigma:.4f} Jy, "
        f"WEIGHT-predicted sigma = {predicted_sigma:.4f} Jy, "
        f"ratio = {scatter_sigma / predicted_sigma:.2f}"
    )

"""
(Caveats worth knowing: the difference estimator assumes the source is constant between
successive integrations — true for continuum — and the comparison of means glosses over
weight variation across baselines; treat a ratio within a few tens of percent of unity as
healthy, and anything factors from unity as a statwt flag.)

__UV Wavelengths__

The MS stores one metric baseline vector per row, but the Fourier plane is sampled in
*wavelengths*: the same physical baseline samples a different (u, v) at each channel
frequency. The conversion is simply

    u [wavelengths] = u [metres] * frequency / c

and `uv_wavelengths_from_uvw` applies it per channel, returning (n_chan, n_rows, 2) — at
continuum width (one channel) that is one (u, v) per row, but the same function serves a
future line cube unchanged. Note only u and v are kept: the w term encodes non-coplanar
effects negligible over ALMA's arcsecond-scale fields of view.
"""
columns = columns_by_spw[SPWS[0]]
uv = uv_wavelengths_from_uvw(columns.uvw, columns.chan_freq)
uv_dist_klambda = np.hypot(uv[..., 0], uv[..., 1]).ravel() / 1e3
print(
    f"spw {SPWS[0]}: uv wavelengths {uv.shape}; baselines span "
    f"{uv_dist_klambda.min():.0f}-{uv_dist_klambda.max():.0f} klambda"
)

"""
__Stokes I__

The two parallel hands (XX, YY) both measure total intensity for an unpolarized source,
so the assemble stage forms the inverse-variance-weighted Stokes-I average per
visibility — the same estimator CASA's own Stokes conversion uses:

    I       = (w_xx * XX + w_yy * YY) / (w_xx + w_yy)
    sigma_I = 1 / sqrt(w_xx + w_yy)

`stokes_i_combine` implements it with two loud edge-case rules: a hand contributes only
where its weight is positive *and* its datum finite (a non-finite visibility must not
leave its weight in the denominator, which would silently bias the average low), and
visibilities where *neither* hand contributes are flagged out via the returned `keep`
mask — dropped and counted, never zero-filled. There is no Casertano correlated-noise
factor here, and never will be: that correction exists for *resampled pixels* whose noise
drizzling correlates; visibilities are independent samples.
"""
stokes_i, sigma, keep = stokes_i_combine(columns.data, columns.weight)
print(
    f"spw {SPWS[0]}: Stokes I {stokes_i.shape}; kept {int(np.count_nonzero(keep))} / "
    f"{keep.size} visibilities ({int(keep.size - np.count_nonzero(keep))} dropped "
    f"zero-weight/invalid); median sigma_I = {float(np.median(sigma[keep])):.4f} Jy"
)

"""
__Concatenate And Package__

`assemble_ms_products` wraps the three operations you just saw (Stokes-I combine, uv
conversion, flatten) for one (uid, spw) block, and `concatenate` stacks the blocks into
the final dataset, keeping per-block provenance. `write_products` then writes the
`(Nvis, 2)` triplet — validating on the way out that every array is finite and the noise
map strictly positive (a zero or negative sigma is a corrupt product, and the packager
crashes loudly rather than shipping it) — plus whichever diagnostic sidecars we hand it.

The result of this hand-built chain lands in its own folder; compare it against
`start_here.py`'s `output/alma_g09v140/` — for the shared execution block the numbers are
identical, because you just ran the same public functions the pipeline runs.
"""
sets, labels, sidecars = [], [], {}
for spw, cols in columns_by_spw.items():
    sets.append(assemble_ms_products(cols))
    labels.append(f"{uid}/spw{spw}")
    tag = f"{uid}_spw_{spw}"
    sidecars[f"antennas_{tag}"] = np.stack((cols.antenna1, cols.antenna2))
    sidecars[f"scans_{tag}"] = cols.scan
    sidecars[f"times_{tag}"] = cols.time
    sidecars[f"frequencies_{tag}"] = cols.chan_freq

combined = concatenate(sets, labels)
print(f"\nConcatenated {len(sets)} blocks: {combined.provenance['n_visibilities']} visibilities")

products = write_products(
    STEPS_OUT,
    combined.visibilities,
    combined.uv_wavelengths,
    combined.noise_map,
    sidecars=sidecars,
)
print(f"Packaged products in {STEPS_OUT}: {products}")

plot_dir = STEPS_OUT / "plots"
plot_dir.mkdir(exist_ok=True)
plt.figure(figsize=(6, 4))
plt.hist(np.hypot(combined.uv_wavelengths[:, 0], combined.uv_wavelengths[:, 1]) / 1e3, bins=60)
plt.xlabel(r"uv distance [k$\lambda$]")
plt.ylabel("visibilities")
plt.title(f"{FIELD} ({uid}): baseline distribution")
hist_png = plot_dir / "uv_distance_hist.png"
plt.savefig(hist_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved baseline-distribution plot to {hist_png.resolve()}")

"""
__Continuum Only__

Honesty section. This branch reduces **continuum** data: line-free spectral windows,
collapsed in frequency. What it deliberately does not do yet:

- **Emission-line / cube extraction.** Lines (like G09v1.40's OH+ and CO(9-8), Butler et
  al. 2021, https://arxiv.org/abs/2104.10077) need finer `width` values and *per-channel*
  product sets — an `al.Interferometer` per channel. The `width` dial and the per-channel
  `uv_wavelengths_from_uvw` machinery above already support it; the per-channel packaging
  is deferred until a line-modeling dataset needs it.
- **Line-channel flagging inside an spw.** Choosing the line-free channels (inspect with
  CASA's `listobs`/plotms before deciding your spws) remains your judgment call at spec
  time — that is why `alma_spws` is a user dial and spws 0 and 3 of this project are not
  in it.
- **statwt / self-calibration.** Weight recomputation and self-cal (see the CASA Guide
  "First Look at Self Calibration" and Richards et al. 2022, ALMA Memo 620,
  https://arxiv.org/abs/2207.05591 — bright lensed DSFGs are frequently self-calibrated)
  belong to the calibration side of the fence, upstream of this pipeline's input
  contract. The weight verification above tells you whether that upstream work is needed.

__Wrap Up__

You ran the visibility branch by hand: two idempotent CASA splits (field isolation, then
channel averaging with a computed smearing budget), casatools extraction into plain
arrays, a weight audit against the visibility scatter, and the pure-numpy assembly —
uv conversion, weighted Stokes-I combine, loud dropping of dead rows — into the packaged
`(Nvis, 2)` triplet.

Good places to checkout next:

- `scripts/alma/simulator.py` — the same chain fed by CASA's simobserve instead of the
  archive: simulate an ALMA observation of a source you control and validate flux
  recovery.
- `scripts/alma/start_here.py` — the one-call version of everything above, plus the
  **PyAutoLens** loading round-trip.
- `autolens_workspace/scripts/interferometer/` — uv-plane lens modeling on these
  products.
"""

"""
__Env__ (Developer Only)

ENV: network
"""
