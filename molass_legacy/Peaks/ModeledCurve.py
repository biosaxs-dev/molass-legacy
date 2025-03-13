# coding: utf-8
"""

    ModeledCurve.py

        evaluation of simularity between curves

    Copyright (c) 2021, SAXS Team, KEK-PF

"""
import numpy as np
from scipy.optimize import minimize
from matplotlib.patches import Rectangle
import molass_legacy.KekLib.DebugPlot as plt
from .RobustCurve import RobustCurve, plot_curve
from .ElutionModels import egh, emg

class Model:
    def __init__(self, name, func, params):
        self.name = name
        self.func = func
        self.params = params

    def __call__(self, x):
        return self.func(x, *self.params)

class FitHistory:
    def __init__(self, x, slice_history, model_history, params_history):
        self.x = x
        self.slice_history = slice_history
        self.model_history = model_history
        self.params_history = params_history

    def get_params_history(self):
        return self.params_history

    def get_shot(self, n):
        return self.slice_history[n], self.model_history[n]

    def get_peaktop_index(self):
        slice_ = self.slice_history[0]
        pti = (slice_.start + slice_.stop)//2
        return pti

    def plot_params_history(self, ax):
        params_array = np.array(self.params_history)
        pti = self.get_peaktop_index()
        num_iter = params_array.shape[0]
        for j in range(4):
            ax.plot(self.x[pti:pti+num_iter], params_array[:,j]/params_array[0,j], ':')

class ModeledCurve(RobustCurve):
    def __init__(self, x, y, depth=2, debug=False):
        RobustCurve.__init__(self, x, y, debug=debug)
        self.depth = depth
        self.children = []
        self.pgs = self.get_peak_models()

    def fit_peak(self, pno, func, allow=0.005):
        return self.fit_locally(self.pgs[pno], func, allow)

    def fit_locally(self, pg, func, allow):
        f, t = self.get_next_range(pg.pt_slice.start, pg.pt_slice.stop)
        max_iter = len(self.y)//2
        print("pg.params=", pg.params)
        h, m, s = pg.params
        pti = (f+t)//2
        params = (self.y[pti], m, s, 0)
        bounds = ((0, None), (m-s, m+s), (0, None), (-5*s, 5*s))
        slice_hishoty = []
        model_history = []
        params_history = []
        for k in range(max_iter):
            nf, nt = self.get_next_range(f, t)
            print([k], (f,t), (nf, nt))
            if nf == f and nt == t:
                break
            slice_ = slice(f, t+1)
            x_ = self.x[slice_]
            y_ = self.y[slice_]

            def objective(p):
                return np.sum((func(x_, *p) - y_)**2)

            result = minimize(objective, params, bounds=bounds)
            slice_hishoty.append(slice_)
            model_history.append(Model('egh', egh, result.x))
            params_history.append(result.x)
            max_deviation = np.max(np.abs(result.x/params_history[0] - 1))
            print([k], max_deviation)
            if max_deviation > allow:
                break
            params = result.x
            f, t = nf, nt
        return FitHistory(self.x, slice_hishoty, model_history, params_history)

    def get_side_residual(self, side, history):
        pti = history.get_peaktop_index()
        slice_ = slice(0, pti) if side == 0 else slice(pti, len(self.x))
        x_ = self.x[slice_]
        _, model = history.get_shot(0)
        y_ = self.y[slice_] - model(x_)
        return ModeledCurve(x_, y_, debug=False)

    def get_next_range(self, f, t):
        nf = max(0, f-1)
        nt = min(len(self.y)-1, t+1)
        while self.y[nf] > self.y[nt]:
            if nf > 0:
                nf -= 1
            else:
                break
        while self.y[nf] < self.y[nt]:
            if nt < len(self.y)-1:
                nt += 1
            else:
                break
        return nf, nt

    def plot_shot(self, ax, shot):
        x = self.x
        y = self.y
        ax.plot(x, y)
        slice_, model = shot
        my = model(x)
        ax.plot(x, my)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
        f = x[slice_.start]
        t = x[slice_.stop-1]
        p = Rectangle(
                (f, ymin),      # (x,y)
                t - f,          # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        ax.add_patch(p)

        ry = y - my
        ax.plot(x, ry)

    def plot(self, ax):
        plot_curve(ax, self)
