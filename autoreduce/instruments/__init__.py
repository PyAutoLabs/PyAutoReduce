"""
Instrument adapters: everything instrument-specific (detector geometry,
calibration file conventions, units, recommended drizzle parameters) lives
behind an adapter so the pipeline stages stay instrument-agnostic. HST/ACS is
the first adapter; WFC3 and JWST follow (see ``docs/design/roadmap.md``).
"""
