"""
GuinierTools.RgCurveUtils — re-exports from molass-library canonical location.

The active computation functions live in molass.Guinier.RgCurveUtils.
This module exists for backward compatibility and provides rg_deviation_inspect_impl
which uses the legacy debug-plot toolkit (plt.Dp()) and therefore stays here.

Canonical usage::

    from molass.Guinier.RgCurveUtils import (
        ValidBools, VALID_BASE_QUALITY,
        get_connected_curve_info, convert_to_milder_qualities,
        get_reconstructed_curve, compute_rg_curves, plot_rg_curves,
    )
"""
import molass_legacy.KekLib.DebugPlot as plt  # needed by rg_deviation_inspect_impl below

from molass.Guinier.RgCurveUtils import (  # noqa: F401
    ValidBools,
    VALID_QUIALTY_LIMIT,
    VALID_BASE_QUALITY,
    convert_to_milder_qualities,
    get_connected_curve_info,
    get_reconstructed_curve,
    compute_rg_curves,
    plot_rg_curves,
)

__all__ = [
    "ValidBools",
    "VALID_QUIALTY_LIMIT",
    "VALID_BASE_QUALITY",
    "convert_to_milder_qualities",
    "get_connected_curve_info",
    "get_reconstructed_curve",
    "compute_rg_curves",
    "plot_rg_curves",
    "rg_deviation_inspect_impl",
]


def rg_deviation_inspect_impl(gdev, valid_components, rg_params, lrf_rgs):
    """Debug inspection plot for GuinierDeviation.compute_deviation().

    Uses the legacy plt.Dp() debug-plot context manager, so it must stay in
    molass-legacy.  Called only when compute_deviation(debug=True).
    """
    print("rg_deviation_inspect_impl")
    print("valid_components=", valid_components)
    print("rg_params=", rg_params)
    print("lrf_rgs=", lrf_rgs)

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        ax1.set_title("Rg Plot", fontsize=16)
        ax2.set_title("Rg Curve", fontsize=16)
        ax1.plot(rg_params, "o", label="Rg params")
        ax1.plot(lrf_rgs, "o", label="LRF Rgs")
        ax1.legend()
        fig.tight_layout()
        plt.show()