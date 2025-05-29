"""
    Models/RateTheory/RobustEDM.py

    Copyright (c) 2023-2025, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from matplotlib.widgets import Slider
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SecTheory.Edm import edm_func
from molass_legacy.Models.Tentative import Model
from molass_legacy.Models.ElutionModelUtils import compute_4moments, get_xies_from_height_ratio, x_from_height_ratio_impl

VERY_SMALL_VALUE = 1e-8

def edm_impl(x, t0, u, a, b, e, Dz, cinj):
    return edm_func(x-t0, u, a, b, e, Dz, cinj)

def guess_init_params(M):
    M1 = M[1]
    M2 = M[2]
    t0 = M1/2
    u = 30/M1*3
    a = 0.5
    b = -4.0
    e = 0.4
    print("guess_init_params: M2=", M2)
    Dz = 0.2
    cinj = M[0]/2.0 * 0.2
    return np.array([t0, u, a, b, e, Dz, cinj])

def try_optimize(x, y, init_params, debug=False):

    def objective(p):
        y_ = edm_impl(x, *p)
        return np.sum((y_ - y)**2)

    ret = minimize(objective, init_params)
    score = np.sqrt(ret.fun)/np.max(y)

    if debug:
        print("init_params=", init_params)
        print("ret.x=", ret.x)
        print("score=", score)


        def add_sliders(fig, line, slider_axes, sliders, left, width, slider_params):
            slider_specs = [    ("t0", -50, 200, slider_params[0]),
                                ("u", 0, 5, slider_params[1]),
                                ("a", 0, 2, slider_params[2]),
                                ("b", -4, 1, slider_params[3]),
                                ("e", 0, 2, slider_params[4]),
                                ("Dz", 0, 1.5, slider_params[5]),
                                ("cinj", 0, 3, slider_params[6]),
                                # ("tinj", 0, 10, slider_params[7]),
                                ]

            def slider_update(k, val):
                # print([k], "slider_update", val)
                init_params[k] = val
                y_ = edm_impl(x, *init_params)
                line.set_data(x, y_)
                dp = plt.get_dp()
                dp.draw()

            for k, (label, valmin, valmax, valinit) in enumerate(slider_specs, start=0):
                ax_ = fig.add_axes([left, 0.8 - 0.08*k, width, 0.03])
                slider  = Slider(ax_, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
                slider.on_changed(lambda val, k_=k: slider_update(k_, val))
                slider_axes.append(ax_)
                sliders.append(slider)

        with plt.Dp():
            fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            ax.plot(x, y)

            edm_line1, = ax.plot(x, edm_impl(x, *init_params))
            ax2.plot(x, y)
            edm_line2, = ax2.plot(x, edm_impl(x, *ret.x))

            slider_axes = []
            sliders = []
            add_sliders(fig, edm_line1, slider_axes, sliders, 0.15, 0.25, init_params)
            add_sliders(fig, edm_line2, slider_axes, sliders, 0.65, 0.25, ret.x)

            fig.tight_layout()
            plt.show()

    return ret.x, score

def guess_multiple(x, y, debug=False):
    M = compute_4moments(x, y)
    init_params = guess_init_params(M)
    params, score = try_optimize(x, y, init_params, debug=True)

    y_ = edm_impl(x, *params)

    try_narrower = False

    if try_narrower:
        max_y = np.max(y)
        xL, xR = get_xies_from_height_ratio(0.3, x, y_, max_y=max_y, debug=False)

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(x, y_)
        if try_narrower:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            for px in xL, xR:
                ax.plot([px, px], [ymin, ymax], ":", color="gray")
        fig.tight_layout()
        plt.show()

    if False:

        i, j = [ int(round(px - x[0])) for px in (xL, xR) ]
        slice_ = slice(i,j+1)
        sx = x[slice_]
        sy = y[slice_]
        better_params, score = try_optimize(sx, sy, params)

        with plt.Dp():
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.plot(x, y_)
            ax.plot(x, edm_impl(x, *better_params))
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            for px in xL, xR:
                ax.plot([px, px], [ymin, ymax], ":", color="gray")
            fig.tight_layout()
            plt.show()

class EDM(Model):
    def __init__(self, **kwargs):
        super(EDM, self).__init__(edm_impl, **kwargs)

    def get_name(self):
        return "EDM"

    def is_traditional(self):
        return False

    def eval(self, params=None, x=None):
        return self.func(x, *params)

    def x_from_height_ratio(self, ecurve, ratio, params):
        return x_from_height_ratio_impl(edm_impl, ecurve, ratio, *params, needs_ymax=True, full_params=True)

    def get_params_string(self, params):
        return 't0=%g, u=%g, a=%g, b=%g, e=%g, Dz=%g, cinj=%g' % tuple(params)
