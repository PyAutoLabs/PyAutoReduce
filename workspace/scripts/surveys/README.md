# Surveys — ground-based colour-context cutouts

The cutout domain: fetch + package postage stamps from public survey cutout
services, **never reduce**. These are colour/context images for lens fields —
especially ALMA targets with no optical counterpart in their modeling data —
and emphatically not modeling inputs: no PSF, mostly no noise map, and a
`products_optional` provenance block that says so.

Scripts:

1. `start_here.py` — the SLACS J0008-0004 field from all three services:
   Legacy Surveys DR10 (grz, 0.262"/px, real noise map via inverse variance;
   covers the DES footprint), SDSS (gri, 0.396"/px) and Pan-STARRS
   (gri, 0.25"/px), plus the multi-survey postage-stamp contact sheet and the
   deferred-extensions list (HSC, unWISE/GALEX, survey PSFs).

Network required; failures are reported per service, never as one crash.
