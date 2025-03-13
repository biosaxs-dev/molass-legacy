"""
    Models.Stochastic.LnporeUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np

from molass_legacy.Models.Stochastic.LognormalUtils import compute_mode

def compute_lnp_elutions(x, lnp_params, rgs):
    from molass_legacy.Models.Stochastic.LognormalPoreFunc import lognormal_pore_func
    N, T, x0, me, mp, mu, sigma = lnp_params[0:7]
    scales = lnp_params[7:]
    cy_list = []
    for scale, rg in zip(scales, rgs):
        cy = lognormal_pore_func(x, scale, N, T, me, mp, mu, sigma, rg, x0) 
        cy_list.append(cy)
    ty = np.sum(cy_list, axis=0)
    return cy_list, ty

def plot_elution_exclusion_impl(ax1, x, y, lnp_params, rgs):

    from molass_legacy.Models.Stochastic.MonoporeUtils import draw_exclusion_cuve
    cy_list, ty = compute_lnp_elutions(x,lnp_params, rgs)
    ax1.set_title("Elution Curve with Exclusion Curve", fontsize=16)
    ax1.set_xlabel("Time (frames)")
    ax1.set_ylabel("Intensity")
    ax1.plot(x, y, color='orange',label="data")
    for k, (cy, rg) in enumerate(zip(cy_list, rgs)):
        ax1.plot(x, cy, ':', label="component-%d, Rg=%.3g" % (k, rg))
    ax1.plot(x, ty, ':', color="red", label="model total")
    ax1.legend()
    axt = ax1.twinx()
    axt.grid(False)
    trs = [x[0] + np.argmax(cy) for cy in cy_list]
    N, T, x0, me, mp, mu, sigma = lnp_params[0:7]
    mode = compute_mode(mu, sigma)
    draw_exclusion_cuve(axt, (N, T, x0, me, mp, mode), trs, rgs)

def compute_psd_curve(mu, sigma):
    from molass_legacy.Models.Stochastic.ParamLimits import PORESIZE_INTEG_LIMIT
    from molass_legacy.Models.Stochastic.LognormalPoreFunc import distr_func
    pv = np.linspace(0, PORESIZE_INTEG_LIMIT, 100)
    pd = distr_func(pv, mu, sigma)
    return pv, pd

def plot_lognormal_psd_impl(ax2, mu, sigma, return_artists=False):
    mode = compute_mode(mu, sigma)
    ax2.set_title("Lognormal Pore Size Distribution", fontsize=16)
    ax2.set_xlabel(r"Pore Size ($\AA$)")
    ax2.set_ylabel("Density")
    pv, pd = compute_psd_curve(mu, sigma)
    psd_curve, = ax2.plot(pv, pd, label="lognormal pdf")
    psd_mode = ax2.axvline(x=mode, ls=':', color="blue", label="poresize (%d)" % int(mode))
    ax2.legend()
    if return_artists:
        from molass_legacy.KekLib.MatplotlibUtils import get_labeltext_from_line
        artists = psd_curve, psd_mode
        labels = [get_labeltext_from_line(a) for a in artists]
        return *artists, *labels

def plot_lnpore_component_with_sliders_impl(x, y, lnp_params, rgs, title=None):
    from matplotlib.gridspec import GridSpec
    from matplotlib.widgets import Slider, Button
    import molass_legacy.KekLib.DebugPlot as plt
    from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
    from molass_legacy.Models.Stochastic.LognormalUtils import compute_mode, compute_stdev, compute_mu_sigma

    plot_params = lnp_params.copy()
    mu, sigma = lnp_params[5:7]
    mode = compute_mode(mu, sigma)
    stdev = compute_stdev(mu, sigma)
    plot_params[5:7] = mode, stdev
    slider_specs = [    ("N", 0, 6000, plot_params[0]),
                        ("T", 0, 3, plot_params[1]),
                        ("t0",  -1000, 1000, plot_params[2]),
                        ("me", 0, 3, plot_params[3]),
                        ("mp", 0, 3, plot_params[4]),
                        ("mode", 20, 400, plot_params[5]),
                        ("stdev", 1, 100, plot_params[6]),
                        ]

    with plt.Dp(button_spec=["OK", "Cancel"]):
        fig = plt.figure(figsize=(18,5))
        gs = GridSpec(1,5)
        ax1 = fig.add_subplot(gs[0,0:3])
        ax2 = fig.add_subplot(gs[0,3:5])
        if title is None:
            title = "Lognormal PSD Elution Components for %s" % get_in_folder()
        fig.suptitle(title, fontsize=20, x=0.35)
        ax1.set_title("Component Elution Curves", fontsize=16)
        ax1.plot(x, y, color='orange',label="data")
        cy_list, ty = compute_lnp_elutions(x, lnp_params, rgs)
        ecurves = []
        for k, (cy, rg) in enumerate(zip(cy_list, rgs)):
            curve, = ax1.plot(x, cy, ':', label="component-%d, Rg=%.3g" % (k, rg))
            ecurves.append(curve)
        curve, = ax1.plot(x, ty, ':', color="red", label="model total")
        ecurves.append(curve)
        ax1.legend()
        psd_curve, psd_mode, psd_curve_label, psd_mode_label = plot_lognormal_psd_impl(ax2, mu, sigma, return_artists=True)

        data_ymin = y.min()
        data_ymax = y.max()
        w = -0.05
        ax1_ymin = data_ymin*(1 - w) + data_ymax*w
        w = 1.1
        ax1_ymax = data_ymin*(1 - w) + data_ymax*w

        def slider_update(k, val):
            # print([k], "slider_update", val)
            plot_params[k] = val
            mode, stdev = plot_params[5:7]
            mu, sigma = compute_mu_sigma(mode, stdev)
            pv, pd = compute_psd_curve(mu, sigma)
            psd_curve.set_ydata(pd)
            psd_mode.set_xdata(mode)
            psd_mode_label.set_text("poresize (%d)" % int(mode))
            lnp_params_temp = plot_params.copy()
            lnp_params_temp[5:7] = mu, sigma
            cy_list, ty = compute_lnp_elutions(x, lnp_params_temp, rgs)
            for cy, ecurve in zip(cy_list + [ty], ecurves):
                ecurve.set_ydata(cy)
            tymin = ty.min()
            tymax = ty.max()
            ax1.set_ylim(min(ax1_ymin, tymin), max(ax1_ymax, tymax))
            ymin, ymax = ax2.get_ylim()
            ax2.set_ylim(ymin, pd.max()*1.2)
            fig.canvas.draw_idle()

        slider_axes = []
        sliders = []
        for k, (label, valmin, valmax, valinit) in enumerate(slider_specs, start=0):
            ax_ = fig.add_axes([0.75, 0.8 - 0.08*k, 0.2, 0.03])
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

        fig.tight_layout()
        fig.subplots_adjust(right=0.7)
        ret = plt.show()
    return ret