"""
Acquisition: resolve a target (name / coordinates / proposal ID) against MAST,
download the calibrated exposures a reduction needs (``_flc`` / ``_flt`` +
``_asn``), and manage the transient per-lens exposure cache so full-frame
files never accumulate on disk. HAPCut/astrocut cutout services cover targets
that sit inside large survey mosaics.
"""
