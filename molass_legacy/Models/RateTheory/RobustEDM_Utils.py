"""
    Models/RateTheory/RobustEDM_Utils.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from matplotlib.widgets import Slider
import molass_legacy.KekLib.DebugPlot as plt
from SecTheory.Edm import edm_func
from .RobustEDM import edm_impl
from molass_legacy.Peaks.MomentsUtils import compute_moments      # for backward compatibility

DRIVE ="D:"
DATA_FOLDER = DRIVE + r"\PyTools\Data"
SAVE_FOLDER = DRIVE + r"\TODO\20230807\training_data"

def guess_init_params(M):
    t0 = M[1]/2
    u = 30/M[1]*3
    a = 0.2
    b = 0.2
    e = 0.4
    Dz = 0.02
    cinj = M[0]/2.0 * 0.2
    return np.array([t0, u, a, b, e, Dz, cinj])

def save_training_data():
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    print("save_training_data")

    data_list = [
        r"\20161104\BL-6A\AhRR",
        r"\20161104\BL-10C\OA",
        r"\20161104\BL-10C\Ald",
        r"\20161216\Backsub",
        r"\20170512\Backsub",
        # r"\20171203",
        r"\20180526\GI",
        r"\20180526\OA",
        r"\20180602",
        r"\20190309_1",
        ]

    sp = StandardProcedure()

    for k, node in enumerate(data_list):
        in_folder = DATA_FOLDER + node
        print([k], node)
        sd = sp.load_old_way(in_folder)
        xr_ecurve = sd.get_xray_curve()
        uv_ecurve = sd.get_uv_curve()

        for type_, ecurve in ('xr', xr_ecurve),('uv', uv_ecurve):
            x = ecurve.x
            y = ecurve.y
            path = r"%s\%04d-%s.dat" % (SAVE_FOLDER, k, type_)
            np.savetxt(path, np.array([x, y]).T)

def try_optimize(x, y, init_params, debug=True):

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
                                ("u", 0, 2, slider_params[1]),
                                ("a", 0, 2, slider_params[3]),
                                ("b", -4, 1, slider_params[3]),
                                ("e", 0, 2, slider_params[4]),
                                ("Dz", 0, 1, slider_params[5]),
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

def train():
    from glob import glob
    fh = open("moments-params.csv", "w")

    for file in glob(SAVE_FOLDER + r"\*.dat"):
        print(file)
        x, y = np.loadtxt(file).T
        M = compute_moments(x, y)
        init_params = guess_init_params(M)
        params, score = try_optimize(x, y, init_params)
        if score < 1:
            fh.write(",".join(["%.3g" % v for v in [*M, *params]]) + "\n")

    fh.close()
