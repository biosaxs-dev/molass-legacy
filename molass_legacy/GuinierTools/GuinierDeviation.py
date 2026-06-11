"""
GuinierTools.GuinierDeviation — re-exports from molass-library canonical location.

The active implementation lives in molass.Guinier.GuinierDeviation.
This module exists for backward compatibility with callers that import from
molass_legacy.GuinierTools.GuinierDeviation.

Canonical usage::

    from molass.Guinier.GuinierDeviation import GuinierDeviation
    from molass.Guinier.GuinierDeviation import USE_NORMALIZED_RMSD_FOR_RGCURVES

Legacy usage (still works)::

    from molass_legacy.GuinierTools.GuinierDeviation import GuinierDeviation
    from molass_legacy.GuinierTools.GuinierDeviation import USE_NORMALIZED_RMSD_FOR_RGCURVES
"""
from molass.Guinier.GuinierDeviation import (  # noqa: F401
    GuinierDeviation,
    USE_NORMALIZED_RMSD_FOR_RGCURVES,
    VERY_SMALL_VALUE,
    GN_DEV_LOWER_LIMIT,
    RG_DEV_LOWER_LIMIT,
    MAXIMIZE_ADJUST,
    NEGATIVE_SLOPE_SCALE,
    QRG_UPPER_BOUND,
    USE_GUINIER_RGS,
    BAD_RG_VALUE,
    MIN_RG,
)

__all__ = [
    "GuinierDeviation",
    "USE_NORMALIZED_RMSD_FOR_RGCURVES",
]