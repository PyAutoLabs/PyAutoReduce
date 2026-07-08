"""
Alignment (design doc stage 2): trust the MAST a-priori WCS by default;
TweakReg refinement is a *triggered* fallback, not a default step.

The trigger diagnostic compares each exposure's WCS-predicted position of
the brightest compact source near the target against the stack consensus;
sub-tolerance scatter means the a-priori solutions are good enough for
drizzling and TweakReg is skipped.
"""

from pathlib import Path
from typing import Dict, List


def wcs_solution_names(exposures: List[Path]) -> Dict[str, str]:
    """Record which WCS solution each exposure carries (provenance)."""
    from astropy.io import fits

    names = {}
    for path in exposures:
        with fits.open(path) as hdul:
            header = hdul["SCI", 1].header
            names[Path(path).name] = header.get("WCSNAME", "unknown")
    return names


def run_tweakreg(exposures: List[Path]) -> None:
    """Relative alignment refinement. Only called when the trigger demands."""
    from drizzlepac import tweakreg

    tweakreg.TweakReg(
        [str(p) for p in exposures],
        interactive=False,
        updatehdr=True,
        shiftfile=False,
    )
