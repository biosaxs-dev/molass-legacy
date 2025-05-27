"""
    SigmoidChain.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from .Sigmoid import fit_bent_sigmoid, ex_sigmoid

HW = 50
JOINT_PENALTY_SCALE = 1e5

class SigmoidChain:
    def __init__(self, x, y, ppk, debug=False):
        self.x = x
        self.y = y
        self.ppk = ppk
        init_params = []
        split_slices = []

        sorted_ppk = sorted(ppk)
        if debug:
            print("ppk=", ppk, "sorted_ppk=", sorted_ppk)

        start = 0
        last_p = None
        for p in sorted_ppk:
            slice_ = slice(max(0, p - HW), min(len(x), p + HW))
            x_ = x[slice_]
            y_ = y[slice_]
            try:
                popt, pcov = fit_bent_sigmoid(x_, y_, p)
                init_params.append(popt)
                p_ = popt[1]
                if last_p is not None:
                    stop = int(round((last_p + p_)/2))
                    split_slices.append(slice(start, stop))
                    start = stop
                last_p = p_
            except:
                log_exception(None, "fit_bent_sigmoid failure: ")

        split_slices.append(slice(start, None))
        self.split_slices = split_slices

        major_params = []
        slope_params = []
        last_s2 = None
        for params in init_params:
            major_params.append(params[0:4])
            s1, s2 = params[4:]
            if last_s2 is None:
                slope_params.append(s1)
            else:
                slope_params.append((last_s2+s1)/2)
            last_s2 = s2
        slope_params.append(last_s2)

        if debug:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("before minimize")
                ax.plot(x, y)
                ax.plot(x, self.evaluate(x, slope_params))
                fig.tight_layout()
                plt.show()

        def obj_func(p):
            y_ = np.zeros(len(x))
            joint_penalty = 0
            for k, slice_ in enumerate(self.split_slices):
                m_param = major_params[k]
                s_param = p[k:k+2]
                y_[slice_] += ex_sigmoid(x[slice_], *m_param, *s_param)
                if k > 0:
                    i = slice_.start
                    joint_penalty += JOINT_PENALTY_SCALE*(y_[i-1] - y_[i])**2
            return np.sum((y_ - y)**2) + joint_penalty

        result = minimize(obj_func, slope_params)
        popt = result.x

        if debug:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("after minimize")
                ax.plot(x, y)
                ax.plot(x, self.evaluate(x, popt))
                fig.tight_layout()
                plt.show()

        self.major_params = major_params
        self.slope_params = popt

    def evaluate(self, x, slope_params=None):
        if slope_params is None:
            slope_params = self.slope_params

        y = np.zeros(len(x))
        k = 0
        for slice_, m_param in zip(self.split_slices, self.major_params):
            y[slice_] += ex_sigmoid(x[slice_], *m_param, *slope_params[k:k+2])
            k += 1
        return y

    def __call__(self, x):
        return self.evaluate(x, self.slope_params)

    def plot_segments(self, ax, label=None, **kwargs):
        k = 0
        for slice_, m_param in zip(self.split_slices, self.major_params):
            x_ = self.x[slice_]
            label_ = label if label is None else label % k
            y_ = ex_sigmoid(x_, *m_param, *self.slope_params[k:k+2])
            ax.plot(x_, y_, label=label_, **kwargs)
            k += 1
