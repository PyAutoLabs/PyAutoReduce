"""
Validation (issue #25): lens-model parity fit for PJ011646 — fit Aris's
modeling dataset and the PyAutoReduce WFC3-IR production reduction with an
*identical* model, mask, positions penalty and search, and compare the
inferred lens parameters. WFC3-IR sibling of the slacs1430 fit (issue #17,
prototypes/slacs1430_parity_fit.py); methodology lessons from that study:
orientation comes from the phase-2 dihedral scan (never assumed), fits run
serially on a quiet machine (~5.5 GB each).

Model: MGE lens light (30 linear Gaussians) + Isothermal + ExternalShear +
MGE source (20 linear Gaussians); Nautilus. PASSAGES J011646.77, F160W,
z_lens 0.555, z_source 2.125, ring radius ~2.2-2.8". The positions pair was
created at inspection (Aris's bundle ships none): the N/S arc-knot pair at
matching radii, threshold 0.7".

Run:  ~/venv/PyAuto/bin/python prototypes/pj011646_parity_fit.py aris
      ~/venv/PyAuto/bin/python prototypes/pj011646_parity_fit.py autoreduce
PyAutoLens + Nautilus required; unit tests never import this.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

DATASET_DIRS = {
    "aris": Path("/mnt/c/Users/Jammy/Science/aris_PJ011646/dataset/aris/PJ011646"),
    "autoreduce": REPO / "scripts" / "output" / "pj011646",
}
ARIS_DIR = DATASET_DIRS["aris"]
OUT = REPO / "prototypes" / "output" / "pj011646_parity"
ORIENTATION_JSON = (
    REPO.parent
    / "autolens_assistant"
    / "scripts"
    / "scratch"
    / "pj011646_parity"
    / "plots"
    / "pixel_comparison.json"
)

PIXEL_SCALE = 0.0642
MASK_RADIUS = 3.8  # arcsec, Aris info.json
Z_LENS, Z_SOURCE = 0.555, 2.125

# Created at inspection (issue #25 phase 0): N/S arc-knot pair, matching radii.
POSITIONS_ARIS = [[2.05, 0.58], [-2.31, 0.19]]

# Dihedral ops that include a reflection (determines the conjugation in the
# anchor similarity below); rotations are orientation-preserving.
REFLECTING_OPS = {"fliplr", "flipud", "transpose", "anti-transpose"}


def _centroid(img, y0, x0, w=6):
    y0, x0 = int(round(y0)), int(round(x0))
    win = img[y0 - 10 : y0 + 11, x0 - 10 : x0 + 11]
    dy, dx = np.unravel_index(np.argmax(win), win.shape)
    y0, x0 = y0 - 10 + dy, x0 - 10 + dx
    win = np.clip(img[y0 - w : y0 + w + 1, x0 - w : x0 + w + 1] - np.median(img), 0, None)
    yy, xx = np.mgrid[0 : 2 * w + 1, 0 : 2 * w + 1]
    return np.array(
        [y0 - w + (win * yy).sum() / win.sum(), x0 - w + (win * xx).sum() / win.sum()]
    )


def transform_positions_to_native(positions_yx: list) -> list:
    """Map Aris-frame (y, x) arcsec positions into the autoreduce frame.

    Exact 2-point similarity anchored on the lens and the bright companion
    (r ~ 5.3") in both frames; conjugation is chosen from the phase-2
    orientation verdict (reflection vs rotation), which must exist first.
    """
    from astropy.io import fits

    orientation = json.loads(ORIENTATION_JSON.read_text())["orientation"]
    conj = orientation in REFLECTING_OPS

    leg = fits.getdata(ARIS_DIR / "data.fits").astype(float)
    new = fits.getdata(DATASET_DIRS["autoreduce"] / "data.fits").astype(float)
    n = leg.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    far = np.hypot(yy - c, xx - c) * PIXEL_SCALE > 3.8
    lens_l, lens_n = _centroid(leg, c, c), _centroid(new, c, c)
    star_l = _centroid(leg, *np.unravel_index(np.argmax(np.where(far, leg, 0)), leg.shape))
    d_l = np.hypot(*(star_l - lens_l))
    # In the new frame, anchor on the far local maximum whose lens distance
    # matches the Aris anchor's (the companion and the corner object have
    # near-equal peaks — a plain argmax can pick different objects per frame).
    from scipy.ndimage import maximum_filter

    cand = (maximum_filter(new, size=9) == new) & far & (new > 10 * np.median(new))
    ys, xs = np.where(cand)
    dists = np.hypot(ys - lens_n[0], xs - lens_n[1])
    k = int(np.argmin(np.abs(dists - d_l)))
    star_n = _centroid(new, ys[k], xs[k])
    if abs(np.hypot(*(star_n - lens_n)) - d_l) > 4.0:
        raise RuntimeError("no matching far anchor within 4 px of the Aris distance")

    def z(p, origin):
        return complex(p[1] - origin[1], p[0] - origin[0])  # x + iy in px

    zl = z(star_l, lens_l)
    s = z(star_n, lens_n) / (np.conj(zl) if conj else zl)
    if not (0.97 < abs(s) < 1.03):
        raise RuntimeError(f"anchor transform not rigid: |s|={abs(s):.4f}")
    out = []
    for (y, x) in positions_yx:
        zp = complex(x, y) / PIXEL_SCALE
        zn = s * (np.conj(zp) if conj else zp)
        out.append(
            [
                float(zn.imag * PIXEL_SCALE + (lens_n[0] - c) * PIXEL_SCALE),
                float(zn.real * PIXEL_SCALE + (lens_n[1] - c) * PIXEL_SCALE),
            ]
        )
    print(
        f"[positions] op={orientation} conj={conj} |s|={abs(s):.4f} "
        f"arg={np.degrees(np.angle(s)):.2f} deg -> {out}"
    )
    return out


def main(key: str):
    import autofit as af
    import autolens as al

    OUT.mkdir(parents=True, exist_ok=True)
    d = DATASET_DIRS[key]

    dataset = al.Imaging.from_fits(
        data_path=d / "data.fits",
        noise_map_path=d / "noise_map.fits",
        psf_path=d / "psf.fits",
        pixel_scales=PIXEL_SCALE,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=MASK_RADIUS,
    )
    dataset = dataset.apply_mask(mask=mask)

    # --- lens light: MGE, 30 linear Gaussians, tied centre + ell ----------
    lens_centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    lens_centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
    log10_sigma_lens = np.linspace(-2.0, np.log10(MASK_RADIUS), 30)
    lens_gaussians = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(30)
    )
    for i, g in enumerate(lens_gaussians):
        g.centre.centre_0 = lens_centre_0
        g.centre.centre_1 = lens_centre_1
        g.ell_comps = lens_gaussians[0].ell_comps
        g.sigma = 10 ** log10_sigma_lens[i]
    lens_bulge = af.Model(al.lp_basis.Basis, profile_list=lens_gaussians)

    lens_mass = af.Model(al.mp.Isothermal)
    lens_mass.einstein_radius = af.UniformPrior(lower_limit=1.0, upper_limit=3.5)

    lens = af.Model(
        al.Galaxy,
        redshift=Z_LENS,
        bulge=lens_bulge,
        mass=lens_mass,
        shear=af.Model(al.mp.ExternalShear),
    )

    # --- source: MGE, 20 linear Gaussians ---------------------------------
    src_centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    src_centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    log10_sigma_src = np.linspace(-2.5, 0.2, 20)
    src_gaussians = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(20)
    )
    for i, g in enumerate(src_gaussians):
        g.centre.centre_0 = src_centre_0
        g.centre.centre_1 = src_centre_1
        g.ell_comps = src_gaussians[0].ell_comps
        g.sigma = 10 ** log10_sigma_src[i]
    source_bulge = af.Model(al.lp_basis.Basis, profile_list=src_gaussians)

    source = af.Model(al.Galaxy, redshift=Z_SOURCE, bulge=source_bulge)
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    positions_yx = list(POSITIONS_ARIS)
    if key != "aris":
        positions_yx = transform_positions_to_native(positions_yx)
    positions = al.Grid2DIrregular(positions_yx)
    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[
            al.PositionsLH(positions=positions, threshold=0.7)
        ],
    )

    # n_live 100 (not the slacs1430 pair's 150): sized to run beside other
    # workloads on this laptop — parity only needs the PAIR identical.
    search = af.Nautilus(
        path_prefix="pj011646_parity",
        name="mge_sie_mge",
        unique_tag=key,
        n_live=100,
        number_of_cores=4,
    )
    result = search.fit(model=model, analysis=analysis)

    samples = result.samples
    mp = samples.median_pdf()

    def bounds(getter, sigma):
        return [
            float(getter(samples.values_at_lower_sigma(sigma))),
            float(getter(samples.values_at_upper_sigma(sigma))),
        ]

    te = lambda v: v.galaxies.lens.mass.einstein_radius
    theta_e = float(te(mp))
    e1, e2 = (float(v) for v in mp.galaxies.lens.mass.ell_comps)
    ell = np.hypot(e1, e2)
    summary = {
        "dataset": key,
        "theta_e_median": theta_e,
        "theta_e_1sigma": bounds(te, 1.0),
        "theta_e_3sigma": bounds(te, 3.0),
        "mass_q": (1 - ell) / (1 + ell),
        "mass_pa_deg": float(np.degrees(np.arctan2(e2, e1)) / 2.0) % 180.0,
        "shear": [
            float(mp.galaxies.lens.shear.gamma_1),
            float(mp.galaxies.lens.shear.gamma_2),
        ],
        "log_evidence": float(samples.log_evidence),
    }
    (OUT / f"fit_summary_{key}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import autolens.plot as aplt

    aplt.subplot_fit_imaging(
        fit=result.max_log_likelihood_fit,
        output_path=str(OUT),
        output_filename=f"fit_{key}",
        output_format="png",
    )
    print(f"PLOT: {(OUT / f'fit_{key}.png').resolve()}")


if __name__ == "__main__":
    main(sys.argv[1])
