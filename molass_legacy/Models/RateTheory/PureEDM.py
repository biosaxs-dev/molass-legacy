"""
    Models/RateTheory/EDM.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from matplotlib.widgets import Slider
import molass_legacy.KekLib.DebugPlot as plt
from SecTheory.Edm import edm_func
from molass_legacy.QuickAnalysis.ModeledPeaks import recognize_peaks
from molass_legacy.Models.ElutionCurveModels import egha
from molass_legacy.Models.Tentative import Model
from molass_legacy.Models.ElutionModelUtils import x_from_height_ratio_impl
from molass_legacy.KekLib.BasicUtils import Struct
from molass_legacy.Peaks.ElutionModels import compute_moments

VERY_SMALL_VALUE = 1e-8

def edm_impl(x, t0, u, a, b, e, Dz, cinj):
    return edm_func(x-t0, u, a, b, e, Dz, cinj)

"""
GI      params= [  0.11655683  0.79493376  1.41901069  -0.38470116  0.29435237  0.15273794  1.33639523 ]
GI+ 50  params= [ 50.11653983  0.79492356  1.41896109  -0.38470303  0.29434855  0.1527366   1.33639626 ]
GI+100  params= [100.11653648  0.79494333  1.41899857  -0.38470053  0.29434736  0.15274067  1.33638596 ]
"""

def guess_x0_u(x0, top_x, top_y):
    if top_x < 200:
        x0_ = 0
        u = 0.7         # ok for sample_data
        # u = 1.0         # ok for HasA
    elif top_x < 300:
        x0_ = 0 if x0 < 200 else x0/2
        u = 0.7
        # u = 0.5
    else:
        x0_ = top_x/3
        u = 0.5
    return x0_, u

GUESS_PARAMS = np.array([
    [-0.403, 0.00278, -8.7e-05, -6.21e-06],
    [1.3, -0.00749, 0.000712, -1.02e-05],
    [1.42, 0.00207, -0.000579, 2.38e-05],
    [-12.1, 0.063, -0.00155, -0.000146],
    [0.477, 0.00916, -0.00199, 1.79e-05],
    [0.496, -0.00305, 0.000205, 1.31e-06],
    [-2.18, 0.0203, -0.00129, 4.19e-06],
    ])

def guess_init_params(moments):
    return [GUESS_PARAMS[k,0] + np.sum(GUESS_PARAMS[k,1:] * np.array(moments)) for k in range(7)]

def guess(x, y, moments=None, init_params=None, debug=False):
    if moments is None:
        if init_params is None:
            j = np.argmax(y)
            top_y = y[j]
            top_x = x[j]
            print("top_y=", top_y)
            x0, u = guess_x0_u(x[0], top_x, top_y)
            cinj_init = min(1.0, 1.32 * top_y/0.291)
            init_params = [x0, u, 1.5, -3.0, 0.4, 0.06, cinj_init]
    else:
        init_params = guess_init_params(moments)

    print("init_params=", init_params)

    if debug:
        slider_specs = [    ("t0", -50, 200, init_params[0]),
                            ("u", 0, 2, init_params[1]),
                            ("a", 0, 2, init_params[3]),
                            ("b", -4, 1, init_params[3]),
                            ("e", 0, 2, init_params[4]),
                            ("Dz", 0, 1, init_params[5]),
                            ("cinj", 0, 3, init_params[6]),
                            # ("tinj", 0, 10, init_params[7]),
                            ]

        with plt.Dp():
            fig, ax = plt.subplots()
            ax.plot(x, y)

            edm_line, = ax.plot(x, edm_impl(x, *init_params))

            def slider_update(k, val):
                # print([k], "slider_update", val)
                init_params[k] = val
                y_ = edm_impl(x, *init_params)
                edm_line.set_data(x, y_)
                dp = plt.get_dp()
                dp.draw()

            slider_axes = []
            sliders = []
            for k, (label, valmin, valmax, valinit) in enumerate(slider_specs, start=0):
                ax_ = fig.add_axes([0.15, 0.8 - 0.08*k, 0.25, 0.03])
                slider  = Slider(ax_, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
                slider.on_changed(lambda val, k_=k: slider_update(k_, val))
                slider_axes.append(ax_)
                sliders.append(slider)

            fig.tight_layout()
            plt.show()

    def objective(p):
        y_ = edm_impl(x, *p)
        return np.sum((y_ - y)**2)

    ret = minimize(objective, init_params)

    if debug:
        M0 = np.sum(y)
        print("ret.x=", ret.x, M0/30)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.plot(x, edm_impl(x, *ret.x))
            fig.tight_layout()
            plt.show()

    return ret.x

def guess_multiple(x, y, num_peaks, fh=None, debug=False):

    M = compute_moments(x, y)
    params = guess(x, y, moments=M, debug=True)

    if fh is not None:
        fh.write(",".join(["%.3g" % v for v in [*M, *params]]) + "\n")

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.plot(x, y)
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
