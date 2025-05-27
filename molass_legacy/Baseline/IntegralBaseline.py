"""
    IntegralBaseline.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt
from .LinearBaseline import USE_END_PARAMS
if not USE_END_PARAMS:
    from .Constants import SLOPE_SCALE

OPTIMIZE_AB = True

class IntegralBaseline:
    def __init__(self, x=None, y=None, debug=False):
        if debug:
            from importlib import reload
            import Baseline.Baseline
            reload(Baseline.Baseline)            
        from .Baseline import compute_baseline

        if x is None:
            return

        yb, (a, b, r) = compute_baseline(y, x=x, integral=True, return_params=True, debug=debug)
        self.x = x
        self.y = y
        self.yb = yb
        self.x1 = x[0]
        self.x2 = x[-1]

        if OPTIMIZE_AB:
            y_ = y - yb

            def func(p):
                a_, b_ = p
                params = [a_*SLOPE_SCALE, b_, r]
                yb_ = self.__call__(x, params, y_, [y_])
                return np.sum((yb_ - yb)**2)

            ret = minimize(func, (a,b))
            a_, b_ = ret.x
        else:
            a_, b_ = a, b

        self.params = [a_*SLOPE_SCALE, b_, r]

        if debug:
            yb_ = self.__call__(x, self.params, y_)
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("IntegralBaseline.__init__")
                ax.plot(x, y)
                ax.plot(x, yb, ":")
                ax.plot(x, yb_, ":")
                fig.tight_layout()
                plt.show()

        y1, y2 = [a_*px+ b_ for px in x[[0,-1]]]
        self.end_params = [y1, y2, r]

    def __call__(self, x, params, y_, cy_list):
        # cy_list is not used in IntegralBaseline

        if USE_END_PARAMS:
            y1, y2, r = params
            k = (y2 - y1)/(self.x2 - self.x1)
            y_linear = y1 + k*(x - self.x1)
        else:
            a, b, r = params
            y_linear = x*a/SLOPE_SCALE + b
        y_integral = r*np.cumsum(y_)
        return y_linear + y_integral

    def get_baseplane(self, D, j1, j2, debug=True):
        bz = D[:,j2] - D[:,j1]
        scale = 1/(self.yb[j2] - self.yb[j1])
        B = scale * bz[:,np.newaxis] @ self.yb[np.newaxis,:]
        if debug:
            from MatrixData import simple_plot_3d
            with plt.Dp():
                fig = plt.figure(figsize=(18,5))
                ax1 = fig.add_subplot(131)
                ax2 = fig.add_subplot(132)
                ax3 = fig.add_subplot(133, projection="3d")
                ax1.plot(self.x, self.y)
                ax2.plot(self.yb)
                simple_plot_3d(ax3, B)
                fig.tight_layout()
                plt.show()
        return B
