"""
    SpecialSigmoid.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.patches import Rectangle
import molass_legacy.KekLib.DebugPlot as plt
from .Sigmoid import fit_bent_sigmoid, ex_sigmoid

def guess_special_sigmoid_params(fc, debug=False):
    ecurve = fc.a_curve

    assert len(ecurve.peak_info) == 1

    ecurve2 = fc.a_curve2
    x = ecurve.x
    y = ecurve.y
    x2 = ecurve2.x
    y2 = ecurve2.y

    f, t = ecurve.get_peak_region_sigma()

    y_ = y.copy()
    y_[np.logical_and(x > f, x < t)] = 0      # better 

    px = (f + t)/2
    popt, pcov = fit_bent_sigmoid(x, y_, x0=px)

    if debug:
        from molass_legacy.Elution.CurveUtils import simple_plot
        with plt.Dp():
            fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(6,8))
            fig.suptitle("guess_special_sigmoid_params: debug")

            simple_plot(ax1, fc.a_curve)
            ymin, ymax = ax1.get_ylim()

            p = Rectangle(
                    (f, ymin),      # (x,y)
                    t - f,          # width
                    ymax - ymin,    # height
                    facecolor   = 'cyan',
                    alpha       = 0.2,
                )

            ax1.add_patch(p)

            ax1.plot(x, ex_sigmoid(x, *popt))

            ax2.plot(x2, y2)

            fig.tight_layout()
            plt.show()

    return [popt]
