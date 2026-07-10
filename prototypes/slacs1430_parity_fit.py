"""
Validation (issue #17): lens-model parity fit for slacs1430+4105 — fit the
legacy modeling dataset and the PyAutoReduce production reduction with an
*identical* model, mask, positions penalty and search, and compare the
inferred lens parameters. Step 3 of the design doc's "Validation — SLACS
parity study" (docs/design/hst_acs_pipeline.md): *the reduction is correct
when the science is invariant under it.*

Model: MGE lens light (30 linear Gaussians) + Isothermal + ExternalShear +
MGE source (20 linear Gaussians); Nautilus. Literature reference values
(legacy info.json): theta_E = 1.52", q = 0.68, PA = 111.7 deg.

Run:  ~/venv/PyAuto/bin/python prototypes/slacs1430_parity_fit.py legacy
      ~/venv/PyAuto/bin/python prototypes/slacs1430_parity_fit.py autoreduce
PyAutoLens + Nautilus required; unit tests never import this.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

DATASET_DIRS = {
    "legacy": Path("/mnt/c/Users/Jammy/Science/subhalo/dataset/slacs/slacs1430+4105"),
    "autoreduce": REPO / "scripts" / "output" / "slacs1430+4105",
}
LEGACY_DIR = DATASET_DIRS["legacy"]
OUT = REPO / "prototypes" / "output" / "slacs1430_parity"

PIXEL_SCALE = 0.05
MASK_RADIUS = 3.5  # arcsec, legacy info.json
Z_LENS, Z_SOURCE = 0.285, 0.575


def _centroid(img, y0, x0, w=8):
    win = img[y0 - 12 : y0 + 13, x0 - 12 : x0 + 13]
    dy, dx = np.unravel_index(np.argmax(win), win.shape)
    y0, x0 = y0 - 12 + dy, x0 - 12 + dx
    win = np.clip(img[y0 - w : y0 + w + 1, x0 - w : x0 + w + 1] - np.median(img), 0, None)
    yy, xx = np.mgrid[0 : 2 * w + 1, 0 : 2 * w + 1]
    return np.array(
        [y0 - w + (win * yy).sum() / win.sum(), x0 - w + (win * xx).sum() / win.sum()]
    )


def transform_positions_to_native(positions_yx: list) -> list:
    """Map legacy-frame (y, x) arcsec positions into the autoreduce frame.

    The legacy reduction is rot270 of the pipeline's north-up frame, no
    mirror (issue #17, ring-annulus correlation 0.997). Fit the exact
    2-point similarity transform (no reflection) anchored on the lens and
    star centroids in both frames, and assert it is rot270-like.
    """
    from astropy.io import fits

    leg = fits.getdata(LEGACY_DIR / "data.fits").astype(float)
    new = fits.getdata(DATASET_DIRS["autoreduce"] / "data.fits").astype(float)
    n = leg.shape[0]
    c = (n - 1) / 2.0
    # (row, col) guesses: lens at centre; star = brightest pixel r > 3".
    yy, xx = np.mgrid[0:n, 0:n]
    far_l = np.where(np.hypot(yy - c, xx - c) * PIXEL_SCALE > 3.0, leg, 0)
    far_n = np.where(np.hypot(yy - c, xx - c) * PIXEL_SCALE > 3.0, new, 0)
    star_l = np.unravel_index(np.argmax(far_l), leg.shape)
    star_n = np.unravel_index(np.argmax(far_n), new.shape)
    lens_l, lens_n = _centroid(leg, 140, 140), _centroid(new, 140, 140)
    star_l, star_n = _centroid(leg, *star_l), _centroid(new, *star_n)

    # Complex similarity, no reflection: z_new - lens_n = s * (z_leg - lens_l).
    def z(p, origin):
        return complex(p[1] - origin[1], p[0] - origin[0])  # x + iy in px

    s = z(star_n, lens_n) / z(star_l, lens_l)
    ang = np.degrees(np.angle(s))
    if not (0.98 < abs(s) < 1.02) or min(abs(ang - 90), abs(ang + 90)) > 3.0:
        raise RuntimeError(
            f"anchor transform not rot270-like: |s|={abs(s):.4f}, arg={ang:.2f} deg"
        )
    out = []
    for (y, x) in positions_yx:
        zl = complex(x, y) / PIXEL_SCALE  # arcsec -> px about lens centre
        zn = s * zl
        out.append(
            [
                float(zn.imag * PIXEL_SCALE + (lens_n[0] - c) * PIXEL_SCALE),
                float(zn.real * PIXEL_SCALE + (lens_n[1] - c) * PIXEL_SCALE),
            ]
        )
    print(f"[positions] |s|={abs(s):.4f} arg={ang:.2f} deg -> {out}")
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
    lens_mass.einstein_radius = af.UniformPrior(lower_limit=0.5, upper_limit=2.5)

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
    log10_sigma_src = np.linspace(-2.5, 0.0, 20)
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

    positions_yx = json.loads((LEGACY_DIR / "positions.json").read_text())
    if key != "legacy":
        positions_yx = transform_positions_to_native(positions_yx)
    positions = al.Grid2DIrregular(positions_yx)
    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[
            al.PositionsLH(positions=positions, threshold=0.7)
        ],
    )

    search = af.Nautilus(
        path_prefix="slacs1430_parity",
        name="mge_sie_mge",
        unique_tag=key,
        n_live=150,
        number_of_cores=4,
    )
    result = search.fit(model=model, analysis=analysis)

    samples = result.samples
    mp = samples.median_pdf()

    def mass_param(getter, sigma):
        lo, hi = (
            float(getter(v))
            for v in (
                samples.values_at_lower_sigma(sigma),
                samples.values_at_upper_sigma(sigma),
            )
        )
        return lo, hi

    theta_e = float(mp.galaxies.lens.mass.einstein_radius)
    te_lo1, te_hi1 = mass_param(
        lambda v: v.galaxies.lens.mass.einstein_radius, 1.0
    )
    te_lo3, te_hi3 = mass_param(
        lambda v: v.galaxies.lens.mass.einstein_radius, 3.0
    )
    e1, e2 = (float(v) for v in mp.galaxies.lens.mass.ell_comps)
    ell = np.hypot(e1, e2)
    q = (1 - ell) / (1 + ell)
    pa = float(np.degrees(np.arctan2(e2, e1)) / 2.0) % 180.0
    g1 = float(mp.galaxies.lens.shear.gamma_1)
    g2 = float(mp.galaxies.lens.shear.gamma_2)

    summary = {
        "dataset": key,
        "theta_e_median": theta_e,
        "theta_e_1sigma": [te_lo1, te_hi1],
        "theta_e_3sigma": [te_lo3, te_hi3],
        "mass_q": q,
        "mass_pa_deg": pa,
        "shear": [g1, g2],
        "log_evidence": float(samples.log_evidence),
        "reference": {"theta_e": 1.52, "q": 0.68, "pa_deg": 111.7},
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
