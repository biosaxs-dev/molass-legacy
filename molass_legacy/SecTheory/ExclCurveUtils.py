"""
    SecTheory.ExclCurveUtils.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.interpolate import UnivariateSpline

def safely_create_spline_from_points(points):
    x = points[:,0]
    y = points[:,1]

    diff_x = np.diff(x)
    strictly_increasing = np.concatenate([[0], np.where(diff_x > 1e-6)[0] + 1])

    x_ = x[strictly_increasing]
    y_ = y[strictly_increasing]

    try:
        spline = UnivariateSpline(x_, y_, s=0)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        import molass_legacy.KekLib.DebugPlot as plt
        log_exception(None, "UnivariateSpline: ")

        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("safely_create_spline_from_points: debug")
            ax.plot(x_, y_, "o")
            fig.tight_layout()
            plt.show()

    return spline

if __name__ == '__main__':
    import os
    import sys

    this_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.dirname(this_dir)
    sys.path.append(lib_dir)
