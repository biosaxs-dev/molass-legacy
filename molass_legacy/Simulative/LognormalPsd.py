"""
    Simulative/LognormalPsd.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button

def lognormalpore_model_interactive_impl(x, y, params, rgs,
                                         title=None, use_ty_as_data=False, plot_ty=None,
                                         plot_mnp=True,
                                         window_title=None, parent=None):
    import molass_legacy.KekLib.DebugPlot as plt
    from molass_legacy.KekLib.MatplotlibUtils import get_labeltext_from_line
    from molass_legacy.Models.Stochastic.LognormalPoreFunc import lognormal_pore_pdf, distr_func, compute_trvec
    from molass_legacy.Models.Stochastic.LognormalUtils import compute_mode, compute_stdev, compute_mu_sigma
    from SecTheory.BasicModels import robust_single_pore_pdf


    if plot_ty is None:
        plot_ty = not use_ty_as_data

    N, T, t0, me, mp, mu, sigma = params[0:7]
    scales = params[7:]

    rgv = np.flip(np.linspace(10, 200, 100))
    trv = compute_trvec(N, T, t0, me, mp, mu, sigma, rgv)
    # plot_rv_list()

    with plt.Dp(parent=parent, window_title=window_title, button_spec=["OK", "Cancel"]):
        fig = plt.figure(figsize=(20,4.5))
        gs = GridSpec(1,4)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1:3])
        ax3 = fig.add_subplot(gs[0,3])
    
        if title is None:
            title = "Simulative Use Demo with Lognormalpore vs. Monopore Comparison"
        fig.suptitle(title, fontsize=20)
        ax1.set_title("Experiment Data", fontsize=16)
        ax2.set_title("Simulated Data", fontsize=16)
        ax3.set_title("Pore Size Distribution", fontsize=16)

        def compute_lnp_curves(x, N, T, me, mp, mu, sigma, t0):
            cy_list = []
            for rg, scale in zip(rgs, scales):
                cy = lognormal_pore_pdf(x, N, T, me, mp, mu, sigma, rg, t0)*scale
                cy_list.append(cy)
            ty = np.sum(cy_list, axis=0)
            return cy_list, ty

        def compute_mnp_curves(x, N, T, me, mp, poresize, t0):
            cy_list = []
            for rg, scale in zip(rgs, scales):
                rho = min(1, rg/poresize)
                ni_ = N * (1 - rho)**me
                ti_ = T * (1 - rho)**mp
                cy = robust_single_pore_pdf(x - t0, ni_, ti_)*scale
                cy_list.append(cy)
            ty = np.sum(cy_list, axis=0)
            return cy_list, ty

        cy_list, ty = compute_lnp_curves(x, N, T, me, mp, mu, sigma, t0)
        if use_ty_as_data:
            y = ty
        if y is not None:
            ax1.plot(x, y, label="data")
        for cy, rg in zip(cy_list, rgs):
            ax1.plot(x, cy, ":", label="rg=%.3g" % rg)
        if plot_ty:
            ax1.plot(x, ty, ":", label="model total", color="red")

        # ax1.legend()

        mode = compute_mode(mu, sigma)
        stdev = compute_stdev(mu, sigma)
        print("mode, stdev=", mode, stdev)
        stdev_percent = stdev/mode*100

        gen_params = np.array([N, T, t0, mode, stdev, stdev_percent])
        slider_specs = [    ("N", 0, 2000, gen_params[0]),
                            ("T", 0, 2, gen_params[1]),
                            ("t0",  -500, 500, gen_params[2]),
                            ("poresize", 10, 600, gen_params[3]),
                            ("stdev", 1, 100, gen_params[4]),
                            ("stdev_percent", 1, 50, gen_params[5]),
                            ]
        x_ = np.arange(-200, 600)
        cy_list, ty = compute_lnp_curves(x_, N, T, me, mp, mu, sigma, t0)

        t0_line = ax2.axvline(t0, color="red", label="t0")
        axt = ax2.twinx()
        axt.grid(False)
        excl_line, = axt.plot(trv, rgv, color="yellow")
        ax2.plot(x_, ty, color="gray", alpha=0.2)
        g_curves = []
        for cy, rg in zip(cy_list, rgs):
            curve, = ax2.plot(x_, cy, ":", label="rg=%.3g" % rg)
            g_curves.append(curve)
        curve, = ax2.plot(x_, ty, label="ln pore total", color="orange")
        g_curves.append(curve)
 
        if plot_mnp:
            poresize = compute_mode(mu, sigma)
            ty_mnp = compute_mnp_curves(x_, N, T, me, mp, poresize, t0)[1]
            mnp_curve, = ax2.plot(x_, ty_mnp, "-", label="mono pore total", color="green", alpha=0.5)

        ax2.legend()

        r = np.arange(0, 600)
        psd_y = distr_func(r, mu, sigma)
        ax3.plot(r, psd_y, color="gray", alpha=0.2)
        psd_curve, = ax3.plot(r, psd_y, alpha=0.5, label="Lognormal(%.3g, %.3g)" % (mu, sigma))
        psd_mode = ax3.axvline(mode, linestyle=":", color="green", label="mode")

        ymin, ymax = ax3.get_ylim()
        init_ymax = ymax*1.3
        ax3.set_ylim(ymin, init_ymax)
        ax3.legend(loc="upper left")    # need to update this if you show the legend

        psd_curve_label = get_labeltext_from_line(psd_curve)

        def slider_update(k, val):
            # print([k], "slider_update", val)
            # for eventson below see https://stackoverflow.com/questions/64420927/matplotlib-update-one-slider-based-on-change-in-another-slider
            gen_params[k] = val
            t0_line.set_xdata(gen_params[2])
            slider = None
            if k in [3, 4, 5]:
                if k in [3, 5]:
                    stdev = gen_params[5]/100*gen_params[3]
                    slider = sliders[4]
                    slider.eventson = False
                    slider.set_val(stdev)
                elif k == 4:
                    stdev = gen_params[4]
                    stdev_percent = gen_params[5]/gen_params[3]*100
                    slider = sliders[5]
                    slider.eventson = False
                    slider.set_val(stdev_percent)
            mu, sigma = compute_mu_sigma(gen_params[3], stdev)
            trv_ = compute_trvec(gen_params[0], gen_params[1], gen_params[2], me, mp, mu, sigma, rgv)
            excl_line.set_xdata(trv_)
            mu, sigma = compute_mu_sigma(gen_params[3], stdev)
            cy_list, ty = compute_lnp_curves(x_, gen_params[0], gen_params[1], me, mp, mu, sigma, gen_params[2])
            for curve, y_ in zip(g_curves, cy_list+[ty]):
                curve.set_ydata(y_)
            if plot_mnp:
                poresize = compute_mode(mu, sigma)
                ty_mnp = compute_mnp_curves(x_, gen_params[0], gen_params[1], me, mp, poresize, gen_params[2])[1]
                mnp_curve.set_ydata(ty_mnp)
                psd_mode.set_xdata(poresize)
            disty = distr_func(r, mu, sigma)
            psd_curve.set_ydata(disty)
            psd_curve_label.set_text("Lognormal(%.3g, %.3g)" % (mu, sigma))
            new_ymax = np.max(disty)*1.2
            ax3.set_ylim(ymin, max(init_ymax, new_ymax))
            fig.canvas.draw_idle()
            if slider is not None:
                slider.eventson = True

        slider_axes = []
        sliders = []
        for k, (label, valmin, valmax, valinit) in enumerate(slider_specs, start=0):
            ax_ = fig.add_axes([0.85, 0.7 - 0.08*k, 0.11, 0.03])
            slider  = Slider(ax_, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
            slider.on_changed(lambda val, k_=k: slider_update(k_, val))
            slider_axes.append(ax_)
            sliders.append(slider)

        def show_curve_data(event):
            print("show_curve_data")
            for k, curve in enumerate(g_curves):
                x, y = curve.get_data()
                print([k], y[0:5])

        def reset(event):
            print("reset")
            for k, slider in enumerate(sliders):
                slider.reset()

        button_ax = plt.axes([0.85, 0.2, 0.12, 0.05])
        debug_btn = Button(button_ax, 'Reset', hovercolor='0.975')
        debug_btn.on_clicked(reset)

        fig.tight_layout()
        fig.subplots_adjust(right=0.8)
        ret = plt.show()
    
    return ret