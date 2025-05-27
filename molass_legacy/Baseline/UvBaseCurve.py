"""
    UV.UvBaseCurve.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.UV.PlainCurve import make_secondary_e_curve_at
from molass_legacy.Trimming.Sigmoid import guess_bent_sigmoid, ex_sigmoid
from molass_legacy.Peaks.EghSupples import egh, d_egh
from molass_legacy.KekLib.SciPyCookbook import smooth

SMOOTHING = True

class UvBaseCurve:
    def __init__(self, sd, curve1, n_sigmchain, n_peaks):
        self.curve1 = curve1
        pn = curve1.primary_peak_no
        emg_peaks = curve1.get_emg_peaks()
        epk = emg_peaks[pn]
        self.peak_params = epk.get_params()
        self.n_sigmchain = n_sigmchain
        self.n_peaks = n_peaks

    def guess_params(self, curve2, pp, slice_, debug=False):
        x = curve2.x
        y = curve2.y

        x_ = x[slice_]
        if SMOOTHING:
            y_ = smooth(y[slice_])
        else:
            y_ = y[slice_]

        def obj_func(p):
            fy = self.__call__(x_, p)
            return np.sum((fy - y_)**2)

        x0 = pp[0]
        start = max(0, x0 - 50)
        stop = min(len(x), x0 + 50)
        guess_slice = slice(start, stop)
        sigm_params = guess_bent_sigmoid(x[guess_slice], y[guess_slice], x0=x0)

        opt_ret = None
        for scale in [-1, 1]:
            init_params = (*sigm_params, scale)
            ret = minimize(obj_func, init_params)
            # ret = basinhopping(obj_func, init_params)
            print("ret.x=", ret.x)
            if opt_ret is None or ret.fun < opt_ret.fun:
                opt_ret = ret

        if debug:
            with plt.Dp():
                from molass_legacy.Elution.CurveUtils import simple_plot
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("debug plot at guess_params")
                simple_plot(ax1, self.curve1)
                ax1.plot(x, egh(x, *self.peak_params))
                ax2.plot(x, y)
                if SMOOTHING:
                    ax2.plot(x_, y_)
                fy = self.__call__(x, opt_ret.x)
                ax2.plot(x, fy)
                ax3.plot()
                fig.tight_layout()
                plt.show()

        return opt_ret.x

    def __call__(self, x, params):
        """
            requires 6 + 1 params when n_sigmchain=1, n_peaks=1
        """
        y = np.zeros(len(x))
        if self.n_sigmchain == 1:
            start = 6
            y += ex_sigmoid(x, *params[0:start])
        else:
            # SigmoidChain
            assert False
        h, m, s, t = self.peak_params
        for k in range(self.n_peaks):
            k_ = start + k
            y += d_egh(x, h*params[k], m, s, t)
        return y
