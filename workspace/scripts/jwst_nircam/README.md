# JWST NIRCam

Reducing JWST/NIRCam imaging into modeling-ready lens datasets with **PyAutoReduce**:
MAST level-2 `_cal` exposures (calwebb_image2 output, MJy/sr) combined through the
official `jwst` calwebb_image3 pipeline, with noise read from the propagated ERR array
and an empirical ePSF built from the mosaic.

The anchor dataset is the COSMOS-Web ring (Mercier et al. 2024), the same lens that
`autolens_workspace/scripts/imaging/start_here.py` models.

Recommended reading order:

- `start_here.py` — the full pipeline end to end on the COSMOS-Web ring in F277W.
- `step_by_step.py` — what calwebb Detector1 / Image2 / Image3 do to the data, stage by stage.
- `multi_band.py` — all four COSMOS-Web bands (F115W/F150W/F277W/F444W) into a multi-wavelength dataset.
- `psf.py` — the JWST PSF story on the M92 stellar field: ePSFs, STARRED vs photutils, undersampling.
- `individual.py` — per-exposure `_crf` frame products instead of (as well as) the mosaic.
- `simulator.py` — synthetic-source injection into the real `_cal` frames and flux recovery.
