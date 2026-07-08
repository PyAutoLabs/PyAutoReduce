"""
PSF estimation, tiered: empirical ePSF from field stars; TinyTim + focus
model when the field is star-poor; optional high-fidelity reconstruction
back-ends (STARRED / PSFr). All tiers emit a PSF consistent with the drizzled
science mosaic (same kernel, pixfrac and pixel scale).
"""
