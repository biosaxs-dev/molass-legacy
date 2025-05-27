"""
    SecTheory.MomentAnalysis.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import root
from molass_legacy.Peaks.ElutionModels import compute_moments
from .BasicModels import robust_single_pore_pdf_scaled as elutionmodel_func
import molass_legacy.KekLib.DebugPlot as plt

def estimate_sec_params(ecurve, debug=False):
    x = ecurve.x
    y = ecurve.y

    M = compute_moments(x, y)
    m = M[0]
    s = np.sqrt(M[1])

    """
        t0, n, t
    """

    def fun(x):
        return [
            x[0] + x[1]*x[2] - M[0],
            2*x[1]*x[2]**2 - M[1],
            6*x[1]*x[2]**3 - M[2],
            ]

    t0_init = 0
    n_init = M[0]
    t_init = 1

    ret = root(fun, (t0_init, n_init, t_init))

    if debug:
        print("ret.success=", ret.success)
        print("ret.message=", ret.message)
        print("ret.x=", ret.x)
        t0, np_, tp_ = ret.x

        with plt.Dp():
            fig, ax = plt.subplots()

            ax.plot(x, y)

            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            for px in [m-s, m, m+s]:
                ax.plot([px, px], [ymin, ymax], color="green")

            ax.plot(x, elutionmodel_func(x - t0, np_, tp_))

            fig.tight_layout()
            plt.show()
