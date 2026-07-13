"""
Acceptance check 4 (issue #13): science invariance — fit the phase-4
B1938+666 reduction with PyAutoLens and compare the inferred Einstein radius
against the published ~0.45" (Lagattuta et al. 2012; Vegetti et al. 2012).

Practicalities, both from the SHARP playbook:
- fit at 20 mas via 2x2 binning (Chen et al. 2019's efficiency trick);
- pixel scale is the *plate-scale-corrected* as-built value: the mosaic was
  resampled assuming native 9.942 mas/pix, but the pre-2015 narrow camera is
  9.952 (Yelda et al. 2010; the raw headers agree), so one output pixel is
  truly 10 mas x (9.952/9.942) = 10.010 mas. Feeding the corrected scale
  makes theta_E physical without touching the shipped products. (Adapter fix
  is a separate, gated source change — issue #13.)

Also measures the ring radius directly (check 3's relative-geometry leg).

Run:  ~/venv/PyAuto/bin/python prototypes/b1938_lens_fit.py
PyAutoLens + Nautilus required; unit tests never import this.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "prototypes" / "output" / "b1938_keck" / "b1938+666"
OUT = REPO / "prototypes" / "output" / "b1938_keck" / "lens_fit"

SCALE_ASBUILT = 0.010  # what the mosaic header claims
SCALE_TRUE = 0.010 * (9.952 / 9.942)  # Yelda plate scale correction
BIN = 2

Z_LENS, Z_SOURCE = 0.881, 2.059
PUBLISHED_THETA_E = 0.45  # arcsec, Lagattuta+12 / Vegetti+12 regime


def bin2(a: np.ndarray, reduce: str) -> np.ndarray:
    ny, nx = (a.shape[0] // BIN) * BIN, (a.shape[1] // BIN) * BIN
    b = a[:ny, :nx].reshape(ny // BIN, BIN, nx // BIN, BIN)
    if reduce == "mean":
        return b.mean(axis=(1, 3))
    if reduce == "quad":  # noise of a mean of BIN^2 pixels
        return np.sqrt((b**2).sum(axis=(1, 3))) / (BIN * BIN)
    if reduce == "sum":
        return b.sum(axis=(1, 3))
    raise ValueError(reduce)


def measure_ring_radius(data: np.ndarray, scale: float) -> dict:
    """Azimuthal-profile ring radius about the flux-weighted galaxy centre."""
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r0 = np.hypot(yy - ny / 2, xx - nx / 2)
    w = np.clip(data, 0, None) * (r0 < 0.3 / scale)
    gy, gx = (w * yy).sum() / w.sum(), (w * xx).sum() / w.sum()
    r = np.hypot(yy - gy, xx - gx) * scale
    bins = np.arange(0.20, 0.90, 0.02)
    prof = np.array(
        [np.median(data[(r >= lo) & (r < lo + 0.02)]) for lo in bins]
    )
    peak = int(np.argmax(prof[8:])) + 8  # search beyond 0.36"
    return {
        "ring_radius_arcsec": float(bins[peak] + 0.01),
        "profile": {f"{lo+0.01:.2f}": float(p) for lo, p in zip(bins, prof)},
    }


def main():
    from astropy.io import fits

    OUT.mkdir(parents=True, exist_ok=True)

    data = fits.getdata(DATASET / "data.fits").astype(float)
    noise = fits.getdata(DATASET / "noise_map.fits").astype(float)
    psf_full = fits.getdata(DATASET / "psf_full.fits").astype(float)

    ring = measure_ring_radius(data, SCALE_TRUE)
    print(f"[check3] ring radius: {ring['ring_radius_arcsec']:.3f}\" "
          f"(published Einstein radius ~{PUBLISHED_THETA_E}\")")
    (OUT / "ring_measurement.json").write_text(json.dumps(ring, indent=2))

    # --- 2x2 bin to 20 mas (Chen 2019) and write modeling inputs ----------
    # The PSF resamples via a centred spline zoom (61 -> 31, odd, centre-
    # preserving — block-binning an odd kernel would give an even Convolver
    # kernel and a half-pixel centroid shift), then crops to 21x21.
    from scipy.ndimage import zoom

    data_b = bin2(data, "mean")
    noise_b = bin2(noise, "quad")
    psf_z = zoom(psf_full, 0.5, order=3)
    c = psf_z.shape[0] // 2
    psf_b = np.clip(psf_z[c - 10 : c + 11, c - 10 : c + 11], 0, None)
    psf_b = psf_b / psf_b.sum()
    scale_b = SCALE_TRUE * BIN
    for name, arr in [("data", data_b), ("noise_map", noise_b), ("psf", psf_b)]:
        fits.PrimaryHDU(arr.astype(np.float32)).writeto(
            OUT / f"{name}_bin2.fits", overwrite=True
        )

    import autolens as al
    import autofit as af

    dataset = al.Imaging.from_fits(
        data_path=OUT / "data_bin2.fits",
        noise_map_path=OUT / "noise_map_bin2.fits",
        psf_path=OUT / "psf_bin2.fits",
        pixel_scales=scale_b,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=1.2,
    )
    dataset = dataset.apply_mask(mask=mask)

    lens_mass = af.Model(al.mp.Isothermal)
    lens_mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    lens_mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    lens_mass.einstein_radius = af.UniformPrior(lower_limit=0.2, upper_limit=0.9)
    lens = af.Model(
        al.Galaxy,
        redshift=Z_LENS,
        bulge=af.Model(al.lp.Sersic),
        mass=lens_mass,
        shear=af.Model(al.mp.ExternalShear),
    )
    source = af.Model(
        al.Galaxy, redshift=Z_SOURCE, bulge=af.Model(al.lp.SersicCore)
    )
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    search = af.Nautilus(
        path_prefix="b1938_keck_acceptance",
        name="sie_sersic_bin2",
        unique_tag="b1938+666",
        n_live=100,
        number_of_cores=4,
    )
    analysis = al.AnalysisImaging(dataset=dataset)
    result = search.fit(model=model, analysis=analysis)

    samples = result.samples
    mp = samples.median_pdf()
    theta_e = float(mp.galaxies.lens.mass.einstein_radius)
    lo3, hi3 = (
        float(v.galaxies.lens.mass.einstein_radius)
        for v in (samples.values_at_lower_sigma(3.0),
                  samples.values_at_upper_sigma(3.0))
    )
    summary = {
        "theta_e_median": theta_e,
        "theta_e_3sigma": [lo3, hi3],
        "published_theta_e": PUBLISHED_THETA_E,
        "ring_radius_arcsec": ring["ring_radius_arcsec"],
        "pixel_scale_used": scale_b,
        "plate_scale_correction": SCALE_TRUE / SCALE_ASBUILT,
        "log_evidence": float(samples.log_evidence),
        "max_lh_chi2_info": "see output dir",
    }
    (OUT / "fit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
