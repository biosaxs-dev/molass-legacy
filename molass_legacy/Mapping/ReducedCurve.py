"""

    ReducedCurve.py

    Copyright (c) 2023, SAXS Team, KEK-PF

"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from molass_legacy.Elution.CurveUtils import simple_plot

RECUDE_LIMIT_RATIO = 0.1

def make_reduced_curve(curve, debug=False):
    """
    temporary coping with the 20201007_2 trouble
    """
    red_curve = ElutionCurve(curve.y, x=curve.x)

    if debug:
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("make_reduced_curve entry")
            simple_plot(ax1, curve)
            simple_plot(ax2, red_curve)
            fig.tight_layout()
            plt.show()

    max_y = curve.max_y
    red_peak_top_x = []
    red_peak_info = []
    red_boundaries = []
    primary_index = None
    max_py = None
    i = 0
    for k, info in enumerate(curve.peak_info):
        px = info[1]
        py = curve.spline(px)
        ratio = py/max_y
        print([k], "ratio=", ratio)
        if ratio < RECUDE_LIMIT_RATIO or px/curve.x[-1] > 0.9:
            # px/curve.x[-1] > 0.9 for ignoring the last peak of uv_curve in 20201007_2
            continue

        if max_py is None or py > max_py:
            max_py = py
            primary_index = i

        i += 1
        red_peak_top_x.append(curve.peak_top_x[k])
        red_peak_info.append(info)
        if k < len(curve.boundaries):
            red_boundaries.append(curve.boundaries[k])

    red_curve.primary_peak_no = primary_index
    red_curve.peak_top_x = np.array(red_peak_top_x)
    red_curve.peak_info = red_peak_info
    red_curve.boundaries = red_boundaries

    if debug:
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("make_reduced_curve return")
            simple_plot(ax1, curve)
            simple_plot(ax2, red_curve)
            fig.tight_layout()
            plt.show()

    return red_curve
