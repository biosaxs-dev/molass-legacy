"""
    Models.Stochastic.DispersiveUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.Stochastic.DispersivePdf import dispersive_monopore_pdf, DEFUALT_TIMESCALE

NUM_SDMCOL_PARAMS = 5
NUM_SDMCUV_PARAMS = 6

def compute_elution_curves(x, sdm_params, peak_rgs, timescale=DEFUALT_TIMESCALE, debug=False):
    me = 1.5
    mp = 1.5
    N_, K_, x0_, poresize_, N0_, tI_ = sdm_params[0:NUM_SDMCUV_PARAMS]
    # print("sdm_params[0:NUM_SDMCUV_PARAMS]=", sdm_params[0:NUM_SDMCUV_PARAMS])
    T_ = K_/N_
    # print("N_, T_, N0_, tI_=", N_, T_, N0_, tI_)
    scales_ = sdm_params[NUM_SDMCUV_PARAMS:]
    rhov = peak_rgs/poresize_
    rhov[rhov > 1] = 1
    cy_list = []
    t0_ = x0_ - tI_
    x_ = x - tI_
    if debug:
        print("x_[0]=", x_[0])
        moments_list = []
    for k, (rho, scale) in enumerate(zip(rhov, scales_)):
        ni_ = N_*(1 - rho)**me
        ti_ = T_*(1 - rho)**mp
        if debug:
            M1_ = x0_ + ni_*ti_
            M2_ = np.sqrt(2*ni_*ti_**2 + (t0_ + ni_*ti_)**2/N0_)
            moments_list.append([M1_, M2_])
            print([k], ni_, ti_, N0_, t0_)
        cy = scale*dispersive_monopore_pdf(x_, ni_, ti_, N0_, t0_, timescale=timescale)
        cy_list.append(cy)    
    ty = np.sum(cy_list, axis=0)

    if debug:
        print("compute_elution_curves: scales_", scales_)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("compute_elution_curves")
            ax.plot(x, ty)
            for cy, (m1, s) in zip(cy_list, moments_list):
                ax.plot(x, cy, ":")
                ax.axvline(x=m1, color='yellow')
                ax.axvspan(m1-s, m1+s, color='yellow', alpha=0.3)
            ax.axvline(x=t0_, color='red')
            fig.tight_layout()
            plt.show()

    return cy_list, ty

def investigate_sdm_params(x, y, sdm_params, num_unreliables, peak_rgs, bounds):
    from matplotlib.widgets import Slider, Button

    # print("sdm_params=", sdm_params)
    print("peak_rgs=", peak_rgs)

    N_, K_, x0_, poresize_, N0_, tI_  = sdm_params[0:NUM_SDMCUV_PARAMS]
    print("K_=", K_)
    print("poresize_=", poresize_)
    print("N0_, tI_=", N0_, tI_)
    T_ = K_/N_
    me = 1.5
    mp = 1.5
    t0 = x0_ - tI_
    print("t0=", t0)
    slider_params = np.concatenate([sdm_params, [DEFUALT_TIMESCALE]])

    with plt.Dp():
        fig, ax = plt.subplots(figsize=(18, 5))
        ax.set_title("investigate_sdm_params")
        ax.plot(x, y)
        curves = []
        cy_list, ty = compute_elution_curves(x, slider_params[:-1], peak_rgs, timescale=slider_params[-1], debug=False)
        print("len(cy_list)=", len(cy_list))
        for cy, rg in zip(cy_list, peak_rgs):
            curve, = ax.plot(x, cy, ":", label="$R_g=$ %.3g" % rg)
            curves.append(curve)
        tcurve, = ax.plot(x, ty, ":", label="total", color="red")
        ax.legend()
        lines = []
        line = ax.axvline(x=tI_, color='gray')
        lines.append(line)
        line = ax.axvline(x=x0_, color='red')
        lines.append(line)
        line = ax.axvline(x=x0_ + K_, color='green')
        lines.append(line)
        fig.tight_layout()

        fig.subplots_adjust(right=0.7)

        slider_specs = [    (r"$N$", *bounds[0], slider_params[0]),
                            (r"$K$", *bounds[1], slider_params[1]),
                            (r"$x_0$", *bounds[2], slider_params[2]),
                            (r"poresize", *bounds[3], slider_params[3]),
                            (r"$N_0$", 1000, 80000, slider_params[4]),
                            (r"$t_I$", *bounds[4], slider_params[5]),
                            (r"$timescale$", 0.01, 1, slider_params[-1]),
                            ]

        def slider_update(k, val):
            # print([k], "slider_update", val)
            slider_params[k] = val
            N_, K_, x0_, poresize_, N0_, tI_ = slider_params[0:NUM_SDMCUV_PARAMS]
            timescale = slider_params[-1]
            cy_list, ty = compute_elution_curves(x, slider_params[:-1], peak_rgs, timescale=timescale)
            for curve, cy in zip(curves, cy_list):
                curve.set_data(x, cy)
            tcurve.set_data(x, ty)
            for line, pos in zip(lines, [tI_, x0_, x0_ + K_]):
                line.set_xdata(pos)
            fig.canvas.draw_idle()

        left = 0.75
        width = 0.2
        slider_axes = []
        sliders = []

        for k, (label, valmin, valmax, valinit) in enumerate(slider_specs, start=0):
            ax_ = fig.add_axes([left, 0.8 - 0.08*k, width, 0.03])
            slider  = Slider(ax_, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
            slider.on_changed(lambda val, k_=k: slider_update(k_, val))
            slider_axes.append(ax_)
            sliders.append(slider)

        def reset(event):
            print("reset")
            for k, slider in enumerate(sliders):
                slider.reset()

        button_ax = fig.add_axes([0.85, 0.2, 0.12, 0.05])
        debug_btn = Button(button_ax, 'Reset', hovercolor='0.975')
        debug_btn.on_clicked(reset)

        ret = plt.show()
    
    return ret

def convert_to_sdm_guess_params(optimizer, p):
    xr_params, xr_baseparams, rg_params, (a, b), uv_params, uv_baseparams, (c, d), sdmcol_params = optimizer.split_params_simple(p)
    sdm_params = np.concatenate([sdmcol_params, xr_params])
    num_colparams = len(sdmcol_params)
    start_index = len(xr_params) + len(xr_baseparams)
    num_components = len(rg_params)
    bounds = list(optimizer.real_bounds[-num_colparams:]) + list(optimizer.real_bounds[start_index:start_index+num_components])
    return sdm_params, rg_params, bounds

def investigate_sdm_params_from_optimizer_params(optimizer, p):
    xr_curve = optimizer.xr_curve
    x = xr_curve.x
    y = xr_curve.y
    sdm_params, rg_params, bounds = convert_to_sdm_guess_params(optimizer, p)
    investigate_sdm_params(x, y, sdm_params, 0, rg_params, bounds)

def investigate_sdm_lrf_params_from_optimizer_params(optimizer, p):
    xr_curve = optimizer.xr_curve
    x = xr_curve.x
    y = xr_curve.y
    sdm_params, rg_params, bounds = convert_to_sdm_guess_params(optimizer, p)
    investigate_sdm_params(x, y, sdm_params, 0, rg_params, bounds)

def investigate_sdm_params_from_v2_result(caller):
    print("investigate_sdm_params_from_v2_result")
    # investigate_sdm_params(x, y, init_cuvparams, num_unreliables, temp_rgs, bounds)