"""
    Models/RateTheory/EDM.py

    Copyright (c) 2023-2025, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from matplotlib.widgets import Slider, Button
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SecTheory.Edm import edm_func
from molass_legacy.QuickAnalysis.ModeledPeaks import recognize_peaks
from molass_legacy.Models.ElutionCurveModels import egha
from molass_legacy.Models.Tentative import Model
from molass_legacy.Models.ElutionModelUtils import compute_4moments, x_from_height_ratio_impl
from molass_legacy.KekLib.BasicUtils import Struct

VERY_SMALL_VALUE = 1e-8
MIN_CINJ = 1e-6
MAX_CINJ = 5.0

if False:
    save_reg_data_fh = open("reg-data.csv", "w")
else:
    save_reg_data_fh = None

def edm_impl(x, t0, u, a, b, e, Dz, cinj):
    return edm_func(x-t0, u, a, b, e, Dz, cinj)

def edm_full_impl(x, t0, u, a, b, e, Dz, cinj, cinit=0, c0=0.0001, tinj=2.0, L=30, z=30):
    return edm_func(x-t0, u, a, b, e, Dz, cinj, cinit, c0, tinj, L, z)

def debug_plot_with_sliders_impl(x, y, init_params, title=None):
    plot_params = init_params.copy()
    slider_specs = [    ("t0", -50, 200, plot_params[0]),
                        ("u", 0, 3, plot_params[1]),
                        ("a", 0, 2, plot_params[2]),
                        ("b", -4, 1, plot_params[3]),
                        ("e", 0, 2, plot_params[4]),
                        ("Dz", 0, 1, plot_params[5]),
                        ("cinj", 0, 3, plot_params[6]),
                        # ("tinj", 0, 10, init_params[7]),
                        ]

    with plt.Dp():
        fig, ax = plt.subplots()
        dp = plt.get_dp()
        if title is None:
            title = "debug_plot_with_sliders_impl"
        ax.set_title(title)
        ax.plot(x, y)

        edm_line, = ax.plot(x, edm_impl(x, *init_params))

        def slider_update(k, val):
            # print([k], "slider_update", val)
            plot_params[k] = val
            y_ = edm_impl(x, *plot_params)
            edm_line.set_data(x, y_)
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

def guess_params_from_sdm(x, y, n1, t1, Nd, t0, debug=True):
    # Nd ≡ N0
    M1 = t0 + n1*t1
    M2 = 2*n1*t1**2 + (M1)**2/Nd
    M3 = 6*n1*t1**2*(Nd*t1 + n1*t1 + t0)/Nd
    # Nd = Td**2/M2 
    Td = np.sqrt(Nd*M2)
    area = np.sum(y)
    print("M1, M2, Td, area=", M1, M2, Td, area)
    # u = 30/Td
    u = 0.5
    # task: compute a, b using M3
    a = 0.5
    b = -4.0
    e = 0.4
    Dz = 0.02
    cinj = 0.1
    """
    t0, u, a, b, e, Dz, cinj
    init_params= [ 5.e+01  5.e-01  5.e-01 -4.e+00  4.e-01  2.e-02  1.e-01]
    params= [ 5.00122462e+01  2.63754524e-01  5.29024183e-02 -3.80947097e+00   5.70061653e-01  2.04147889e-02  4.78377216e-01]
    """
    def objective(params):
        t0, u, a, b, e, Dz, cinj = params
        y_ = edm_impl(x, t0, u, a, b, e, Dz, cinj)
        return np.sum((y_ - y)**2)
    
    init_params = np.array([t0, u, a, b, e, Dz, cinj])
    if debug:
        print("init_params=", init_params)
        debug_plot_with_sliders_impl(x, y, init_params, title="guess_params_from_sdm: before minimize")
    res = minimize(objective, init_params)
    if debug:
        debug_plot_with_sliders_impl(x, y, res.x, title="guess_params_from_sdm: after minimize")
    return res.x

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
        y_ = edm_impl(x, *params)
        area_ = np.sum(y_)
        score = abs(1 - area_/area)

        if np.isnan(score):
            score = np.inf

        score_list.append(score)

    k = np.argmin(score_list)
    return params_list[k]

def guess(x, y, init_params=None, debug=False, debug_info=None):
    if debug:
        from importlib import reload
        import Models.RateTheory.RobustEDM
        reload(Models.RateTheory.RobustEDM)
    from molass_legacy.Models.RateTheory.RobustEDM import guess_init_params

    if debug:
        def debug_plot_with_sliders():
            return debug_plot_with_sliders_impl(x, y, init_params, title="guess: before minimize with sliders")


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
                ax2.plot(x, edm_impl(x, *params))
                fig.tight_layout()
                plt.show()

    if init_params is None:
        M = compute_4moments(x, y)
        # init_params = guess_init_params_better(x, y, M)
        init_params = guess_init_params(M)
        area = np.sum(y)
        y_i = edm_impl(x, *init_params)
        area_i = np.sum(y_i)
        ratio = area_i/area
        print("area ratio=", ratio)
        if debug:
            print("M=", M)
            print("init_params=", init_params)
            changed_params = debug_plot_with_sliders()
            print("changed_params=", changed_params)
            changed_y_i = edm_impl(x, *changed_params)
            changed_area_i = np.sum(changed_y_i)
            changed_ratio = changed_area_i/area
            print("changed area ratio=", changed_ratio)
            if ratio < 0.1:
                print("changed_params=", changed_params)
                init_params = changed_params
            debug_plot_params(x, y, init_params, "guess: before minimize")

    def objective(p):
        y_ = edm_impl(x, *p)
        return np.sum((y_ - y)**2)

    ret = minimize(objective, init_params)

    if debug:
        print("M=", M)
        debug_plot_params(x, y, ret.x, "guess: after minimize")

    return ret.x

def guess_multiple_impl(x, y, num_peaks, respect_egh=False, debug=False):
    egha_params_array = np.array(recognize_peaks(x, y, exact_num_peaks=num_peaks, affine=True))

    cy_list = []
    for params in egha_params_array:
        cy = egha(x, *params)
        cy_list.append(cy)

    if debug:
        ty = np.sum(cy_list, axis=0)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("guess_multiple: EGH decomposition")
            ax.plot(x, y)
            for k, cy in enumerate(cy_list):
                ax.plot(x, cy, ":", label="component-%d" % k)
            ax.plot(x, ty, ":", color="red", lw=3, label="total")
            ax.legend()
            fig.tight_layout()
            plt.show()

    # sort in the order of peak heights
    exec_recs = sorted(np.array([np.arange(num_peaks), egha_params_array[:,0]]).T, key=lambda x: -x[1])
    print("exec_recs=", exec_recs)

    ecurve = Struct(x=x, y=y)   # dummy will sffice here
    edm_params_list = [None] * num_peaks
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
        if respect_egh:
            egh_cy = cy_list[i]
            y_ = egh_cy[range_]
        else:
            y_ = y_resid[range_]
        if debug:
            debug_info = (x, y, y_resid, range_)
        else:
            debug_info = None

        if debug:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("guess_multiple: range")
                ax.plot(x, y_resid)
                ax.plot(x_, y_)
                fig.tight_layout()
                plt.show()

        sum_y = np.sum(y_)
        if sum_y < 0:
            # avoid computing for negative regions
            print("avoid computing for negative regions: np.sum(y_)=%g < 0" % sum_y)
            num_failures += 1
            continue

        params = guess(x_, y_, debug=debug, debug_info=debug_info)
        edm_params_list[i] = params
        if save_reg_data_fh is not None:
            j = np.argmax(y_)
            top_x = x_[j]
            top_y = y_[j]
            save_reg_data_fh.write(",".join(["%.3g" % v for v in [x_[0], top_x, top_y, *params]]) + "\n")
        cy = edm_impl(x, *params)
        edm_cy_list[i] = cy
        y_resid -= cy

        if debug:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("guess_multiple: cy, y_resid")
                ax.plot(x_, y_)
                ax.plot(x, cy, ":")
                ax.plot(x, y_resid)
                fig.tight_layout()
                plt.show()

    if debug:
        print("num_failures=", num_failures)
        def draw_edm_cy_list(title, edm_cy_list):
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title(title)
                ax.plot(x, y)

                cy_list = []
                for k, cy in enumerate(edm_cy_list):
                    if cy is not None:
                        ax.plot(x, cy, ":", label="component-%d" % k)
                        cy_list.append(cy)

                ty = np.sum(cy_list, axis=0)
                ax.plot(x, ty, ":", color="red", lw=3, label="total")
                ax.legend()

                fig.tight_layout()
                ret = plt.show()
            return ret

        draw_edm_cy_list("guess_multiple: EDM decomposition before cinj optimization", edm_cy_list)
    # 
    def cinj_ovjective(p, return_cy_list=False):
        cy_list = []
        for i, params in enumerate(edm_params_list):
            params_ = params.copy()
            params_[6] = p[i]
            cy = edm_impl(x, *params_)
            cy_list.append(cy)
        if return_cy_list:
            return cy_list
        ty = np.sum(cy_list, axis=0)
        return np.sum((y - ty)**2)

    init_cinjs = [p[6] for p in edm_params_list]
    bounds = [(MIN_CINJ, MAX_CINJ)] * num_peaks
    ret = minimize(cinj_ovjective, init_cinjs, method="Nelder-Mead", bounds=bounds)
    edm_cy_list = cinj_ovjective(ret.x, return_cy_list=True)
    if debug:
        draw_edm_cy_list("guess_multiple: EDM decomposition after cinj optimization", edm_cy_list)

    peak_pos = []
    for i, params in enumerate(edm_params_list):
        params[6] = ret.x[i]
        m = np.argmax(edm_cy_list[i])
        peak_pos.append(x[m])
    sort_pairs = sorted(zip(peak_pos, edm_params_list), key=lambda x: x[0])
    final_params_list = [pair[1] for pair in sort_pairs]

    if num_failures > 0:
        assert not respect_egh
        print("try guess_multiple_impl respect_egh=True")
        try:
            return guess_multiple_impl(x, y, num_peaks, respect_egh=True)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(None, "guess_multiple_impl respect_egh=True failed: ")

        ret_list = []
        for p in final_params_list:
            if p is not None:
                ret_list.append(p)
    else:
        ret_list = final_params_list

    return np.array(ret_list)

class EDM(Model):
    def __init__(self, **kwargs):
        super(EDM, self).__init__(edm_impl, **kwargs)

    def get_name(self):
        return "EDM"

    def is_traditional(self):
        return False

    def guess_multiple(self, x, y, num_peaks, debug=False):
        return guess_multiple_impl(x, y, num_peaks, debug=debug)

    def eval(self, params=None, x=None):
        return self.func(x, *params)

    def x_from_height_ratio(self, ecurve, ratio, params):
        return x_from_height_ratio_impl(edm_impl, ecurve, ratio, *params, needs_ymax=True, full_params=True)

    def get_params_string(self, params):
        return 't0=%g, u=%g, a=%g, b=%g, e=%g, Dz=%g, cinj=%g' % tuple(params)

    def adjust_to_xy(self, params_list, x, y, props=None, devel=False):
        if props is None:
            areas = []
            for p in params_list:
                cy = edm_impl(x, *p)
                areas.append(np.sum(cy))
            props = np.array(areas)/np.sum(areas)

        print("props=", props)

        if devel:
            def plot_with_params_list(title, params_list):
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.set_title(title)
                    ax.plot(x, y)
                    cy_list = []
                    for p in params_list:
                        cy = edm_impl(x, *p)
                        cy_list.append(cy)
                        ax.plot(x, cy, ":")
                    ty = np.sum(cy_list, axis=0)
                    ax.plot(x, ty, ":", color="red")
                    fig.tight_layout
                    plt.show()
            plot_with_params_list("before conversion", params_list)

        params_array = np.array(params_list)

        def objective(p):
            cy_list = []
            areas = []
            for params in p.reshape(params_array.shape):
                cy = edm_impl(x, *params)
                cy_list.append(cy)
                areas.append(np.sum(cy))
            ty = np.sum(cy_list, axis=0)
            props_ = np.array(areas)/np.sum(areas)
            fv = np.sum((ty - y)**2) + np.sum((props_ - props)**2)
            return fv

        ret = minimize(objective, params_array.flatten(), method='Nelder-Mead')
        print("ret.success=", ret.success)
        converted_array = ret.x.reshape(params_array.shape)

        if devel:
            plot_with_params_list("after conversion", converted_array)
        return converted_array

def test_it_from_editor_frame(editor):
    frame = editor.get_current_frame()
    model = frame.model
    print("edm_text_from_editor_frame: ", model.get_name())
    params_list = []
    for rec in frame.opt_recs:
        params = rec.get_params()
        params_list.append(params)

    fx = frame.fx
    x = frame.x
    y = frame.y
    uv_y = frame.uv_y

    converted_list = model.adjust_to_xy(params_list, fx, uv_y)

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        ax1.plot(x, uv_y, color="blue")
        for params in converted_list:
            cy = edm_impl(fx, *params)
            ax1.plot(x, cy, ":")
        ax2.plot(x, y, color="orange")
        for params in params_list:
            cy = edm_impl(fx, *params)
            ax2.plot(x, cy, ":")
        fig.tight_layout()
        plt.show()