"""
    Models/Stochastic/TriporeImpl.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from matplotlib.widgets import Slider, Button
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.QuickAnalysis.ModeledPeaks import recognize_peaks
from molass_legacy.Models.ElutionCurveModels import egha
from molass_legacy.Models.Tentative import Model
from molass_legacy.Models.ElutionModelUtils import x_from_height_ratio_impl
from molass_legacy.KekLib.BasicUtils import Struct
from molass_legacy.SecTheory.SecCF import gec_tripore_phi
from molass_legacy.SecTheory.SecPDF import FftInvPdf

PORESIZES = np.array([170,  55,  40])   # see meeting-docs/20231218/index.html
POREPROPS = np.array([0.1, 0.3, 0.6])   # see as above
# N = 1000
# T = 1.2

tripore_pdf = FftInvPdf(gec_tripore_phi)

def tripore_func(x, scale, *params):
    """
    all: N, T, me, mp, pore-distribution, Rg, scale, x0
    pore-distribution:
        pore sizes
        pore proportions

    mid: N, T, Rg, scale, x0  ---- implemented below

    min: Rg, scale, x0
    """

    N, T, Rg, x0 = params

    me = 1.5
    mp = 1.5
    rho = Rg/PORESIZES
    rho[rho > 1] = 1
    np_ = N * POREPROPS * (1 - rho)**me
    tp_ = T*(1 - rho)**mp

    return scale*tripore_pdf(x, np_[0], tp_[0], np_[1], tp_[1], np_[2], tp_[2], x0)

VERY_SMALL_VALUE = 1e-8
if False:
    save_reg_data_fh = open("reg-data.csv", "w")
else:
    save_reg_data_fh = None

def guess_init_params_better(x, y, M):
    area = np.sum(y)
    score_list = []
    params_list = []
    for uscale in [3]:
        t0 = M[1]/2
        u = 30/M[1]*uscale
        a = 0.5
        if M[3] < 0:
            b = -4.0
        else:
            b = -4.0
        e = 0.4
        Dz = 0.02
        cinj = M[0]/2.0 * 0.2
        params = np.array([t0, u, a, b, e, Dz, cinj])
        params_list.append(params)
        y_ = tripore_func(x, *params)
        area_ = np.sum(y_)
        score = abs(1 - area_/area)

        if np.isnan(score):
            score = np.inf

        score_list.append(score)

    k = np.argmin(score_list)
    return params_list[k]

def guess_init_params(x, y, mu, M, rg, debug=False):
    N = 2000
    T = 8
    x0 = 50
    def objective(p, title=None):
        scale, N, T, rg, x0 = p
        y_ = tripore_func(x, scale, N, T, rg, x0)
        if title is not None:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title(title)
                ax.plot(x, y)
                ax.plot(x, y_)
                fig.tight_layout()
                plt.show()
        return np.sum((y_ - y)**2)

    print("mu/M[1]=", mu/M[1])
    init_params = (1, N, T, rg, x0)
    if debug:
        objective(init_params, title="guess_init_params: before minimize")

    ret = minimize(objective, init_params)

    if debug:
        objective(ret.x, title="guess_init_params: after minimize")
    scale_, N_, T_, rg_, x0_ =  ret.x

    return np.array([scale_, N_, T_, rg_, x0_])

def guess(x, y, mu, init_params=None, debug=False, debug_info=None):
    from molass_legacy.Models.ElutionModelUtils import compute_4moments

    if debug:
        # y = tripore_func(x, scale, np1, tp1, np2, tp2, np3, tp3, x0)
        def debug_plot_with_sliders():
            slider_specs = [    ("scale", 0, 10, init_params[0]),
                                ("N", 0, 2000, init_params[1]),
                                ("T", 0, 10, init_params[2]),
                                ("rg", 0, 100, init_params[3]),
                                ("x0",  -100, 100, init_params[4]),
                                ]

            with plt.Dp():
                fig, ax = plt.subplots()
                dp = plt.get_dp()

                ax.set_title("guess: before minimize with sliders")
                ax.plot(x, y)

                stc_line, = ax.plot(x, tripore_func(x, *init_params))

                def slider_update(k, val):
                    # print([k], "slider_update", val)
                    init_params[k] = val
                    y_ = tripore_func(x, *init_params)
                    stc_line.set_data(x, y_)
                    dp.draw()

                slider_axes = []
                sliders = []
                for k, (label, valmin, valmax, valinit) in enumerate(slider_specs, start=0):
                    ax_ = fig.add_axes([0.15, 0.8 - 0.08*k, 0.25, 0.03])
                    slider  = Slider(ax_, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
                    slider.on_changed(lambda val, k_=k: slider_update(k_, val))
                    slider_axes.append(ax_)
                    sliders.append(slider)

                def reset_slider_valules(event):
                    for slider in sliders:
                        slider.reset()
                    dp.draw()

                axreset = fig.add_axes([0.9, 0.0, 0.1, 0.075])
                breset = Button(axreset, 'Reset')
                breset.on_clicked(reset_slider_valules)

                fig.tight_layout()
                applied = plt.show()
                if applied:
                    params = np.array([slider.val for slider in sliders])
                else:
                    params = init_params

            return params

        def debug_plot_params(x, y, params, title):
            print("params=", params)
            wx, wy, wy_resid, range_ = debug_info
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(12,5))
                fig.suptitle(title, fontsize=20)
                ax1.set_title("Guessing Range", fontsize=16)
                ax2.set_title("Guess in the Range", fontsize=16)

                ax1.plot(wx, wy)
                ax1.plot(wx, wy_resid)
                ax1.plot(x, y)
                ymin, ymax = ax1.get_ylim()
                ax1.set_ylim(ymin, ymax)
                for px in x[[0,-1]]:
                    ax1.plot([px, px], [ymin, ymax], ":", color="gray")

                ax2.plot(x, y)
                ax2.plot(x, tripore_func(x, *params))
                fig.tight_layout()
                plt.show()

    if init_params is None:
        M = compute_4moments(x, y)
        # init_params = guess_init_params_better(x, y, M)
        init_params = guess_init_params(x, y, mu, M, 60)
        area = np.sum(y)
        y_i = tripore_func(x, *init_params)
        area_i = np.sum(y_i)
        ratio = area_i/area
        print("area ratio=", ratio)
        if debug:
            print("M=", M)
            print("init_params=", init_params)
            changed_params = debug_plot_with_sliders()
            print("changed_params=", changed_params)
            changed_y_i = tripore_func(x, *changed_params)
            changed_area_i = np.sum(changed_y_i)
            changed_ratio = changed_area_i/area
            print("changed area ratio=", changed_ratio)
            if ratio < 0.1:
                print("changed_params=", changed_params)
                init_params = changed_params
            debug_plot_params(x, y, init_params, "guess: before minimize")

    def objective(p):
        y_ = tripore_func(x, *p)
        return np.sum((y_ - y)**2)

    ret = minimize(objective, init_params)

    if debug:
        print("M=", M)
        debug_plot_params(x, y, ret.x, "guess: after minimize")

    return ret.x

def guess_multiple_impl(x, y, num_peaks, debug=False):
    egha_params_array = np.array(recognize_peaks(x, y, exact_num_peaks=num_peaks, affine=True))

    cy_list = []
    for params in egha_params_array:
        cy = egha(x, *params)
        cy_list.append(cy)

    if debug:
        # np.savetxt("temp.dat", np.array([x, y]).T)
        ty = np.sum(cy_list, axis=0)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("guess_multiple: EGH decomposition")
            ax.plot(x, y)
            for cy in cy_list:
                ax.plot(x, cy, ":")
            ax.plot(x, ty, ":", color="red", lw=3)
            fig.tight_layout()
            plt.show()

    # sort in the order of peak heights
    exec_recs = sorted(np.array([np.arange(num_peaks), egha_params_array[:,0]]).T, key=lambda x: -x[1])
    print("exec_recs=", exec_recs)

    ecurve = Struct(x=x, y=y)   # dummy will sffice here
    stc_params_list = [None] * num_peaks
    y_resid = y.copy()
    edm_cy_list = [None] * num_peaks

    num_failures = 0
    for k, h_ in exec_recs:
        i = int(k)
        egha_params = egha_params_array[i,:]
        try:
            ret_xes = x_from_height_ratio_impl(egha, ecurve, 0.1, *egha_params[1:])
            # print([i], egha_params, ret_xes)
        except:
            # as with pH6
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(None, "a possible component peak ignored due to x_from_height_ratio_impl failure: ")
            num_failures += 1
            continue

        range_ = np.logical_and(x > ret_xes[0], x < ret_xes[1])
        x_ = x[range_]
        y_ = y_resid[range_]
        if debug:
            debug_info = (x, y, y_resid, range_)
        else:
            debug_info = None

        if np.sum(y_) < 0:
            # avoid computing for negative regions
            num_failures += 1
            continue

        mu = egha_params[1]
        params = guess(x_, y_, mu, debug=debug, debug_info=debug_info)
        print([i], "params=", params)
        stc_params_list[i] = params
        cy = tripore_func(x, *params)
        edm_cy_list[i] = cy
        y_resid -= cy

    if debug:
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("guess_multiple: STC decomposition")
            ax.plot(x, y)

            cy_list = []
            for cy in edm_cy_list:
                if cy is not None:
                    ax.plot(x, cy, ":")
                    cy_list.append(cy)

            ty = np.sum(cy_list, axis=0)
            ax.plot(x, ty, ":", color="red", lw=3)

            fig.tight_layout()
            plt.show()

    if num_failures > 0:
        ret_list = []
        for p in stc_params_list:
            if p is not None:
                ret_list.append(p)
    else:
        ret_list = stc_params_list

    return np.array(ret_list)