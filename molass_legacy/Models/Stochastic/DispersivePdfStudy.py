"""
    Models.Stochastic.DispersivePdfStudy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.widgets import Slider, Button
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.Stochastic.ParamLimits import USE_K, MAX_PORESIZE
from molass_legacy.Models.Stochastic.DispersivePdf import dispersive_monopore_pdf

def study_pdf():
    N = 2000
    T = 0.5
    K = N*T
    me = 1.5
    mp = 1.5
    poresize = 75
    rgs = np.array([50, 40, 30, 20])
    rhov = rgs/poresize
    rhov[rhov > 1] = 1
    t0 = 500
    N0 = 14400
    t = np.arange(0, 1000)

    def compute_curves(x, N, T, x0, tI, N0, rhov):
        cy_list = []
        for rho in rhov:
            ni = N*(1 - rho)**me
            ti = T*(1 - rho)**mp
            cy = dispersive_monopore_pdf(x - tI, ni, ti, N0, x0 - tI)
            cy_list.append(cy)
        return cy_list

    rv = np.linspace(1, 70, 100)
    def compute_trv(N, T, poresize):
        rhov_ = rv/poresize
        rhov_[rhov_ > 1] = 1
        niv = N*(1 - rhov_)**me
        tiv = T*(1 - rhov_)**mp
        trv = t0 + niv*tiv
        return trv
    trv = compute_trv(N, T, poresize)

    def plot_curves(ax, x, x0, tI, title, with_arrow=False):
        ax.set_title(title + " with $t_{I}$=%g" % tI, fontsize=16)
        cy_list = compute_curves(x, N, T, x0, tI, N0, rhov)
        curves = []
        for k, (cy, rg) in enumerate(zip(cy_list, rgs)):
            curve, = ax.plot(x, cy, ':', label='$R_g$=%g' % rg)
            curves.append(curve)
        lines = []
        line = ax.axvline(x=x0, color='red', label='$t_0$')
        lines.append(line)
        line = ax.axvline(x=tI, color='gray', label='$t_I$')
        lines.append(line)
        if with_arrow:
            hy = np.average(ax.get_ylim())
            dx = x0 - tI
            ax.arrow(x=tI, y=hy, dx=dx, dy=0, width=0.005, head_width=0.015,
                    head_length=0.2*dx, length_includes_head=True, color='pink')
        ax.legend()

        axt = ax.twinx()
        axt.grid(False)
        excl_curve, = axt.plot(trv + tI, rv, color='yellow', label='exclusion curve')
        axt.legend(loc="center right")

        return curves, lines, excl_curve

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,5))
        fig.suptitle("SDM PDF Study in Experimental Time Axis with Poresize=%g, N*T=%g" % (poresize, K), fontsize=20)

        curves1, lines1, excl_curve1 = plot_curves(ax1, t, t0, 0, "Theoretical Time Axis")

        tI = -500
        x = t + tI
        x0 = t0 + tI
        curves2, lines2, excl_curve2 = plot_curves(ax2, x, x0, tI, "Experimental Time Axis")

        fig.tight_layout()
        fig.subplots_adjust(right=0.7, wspace=0.2)

        slider_params = [N, K, t0, poresize, N0, tI]
        bounds = [(100, 3000), (100, 3000), (0, 1000), (30, MAX_PORESIZE), (100, 30000), (-1000, 0)]

        slider_specs = [    (r"$N$", *bounds[0], slider_params[0]),
                            (r"$K$", *bounds[1], slider_params[1]),
                            (r"$t_0$", *bounds[2], slider_params[2]),
                            (r"$poresize$", *bounds[3], slider_params[3]),
                            (r"$N_0$", *bounds[4], slider_params[4]),
                            (r"$t_I$", *bounds[5], slider_params[5]),
                            ]

        def slider_update(k, val):
            # print([k], "slider_update", val)
            slider_params[k] = val
            N_, K_, t0_, poresize_, N0_, tI_ = slider_params
            T_ = K_/N_
            rhov_ = rgs/poresize_
            cy_list1 = compute_curves(t, N_, T_, t0_, 0, N0_, rhov_)
            for curve, cy in zip(curves1, cy_list1):
                curve.set_data(t, cy)
            for line, pos in zip(lines1, [t0_, 0]):
                line.set_xdata(pos)

            x = t + tI_
            x0 = t0_ + tI_
            cy_list2 = compute_curves(x, N_, T_, x0, tI_, N0_, rhov_)
            for curve, cy in zip(curves2, cy_list2):
                curve.set_data(x, cy)
            for line, pos in zip(lines2, [x0, tI_]):
                line.set_xdata(pos)

            trv_ = compute_trv(N_, T_, poresize_)
            excl_curve1.set_data(trv_, rv)
            excl_curve2.set_data(trv_ + tI_, rv)

            fig.canvas.draw_idle()

        left = 0.78
        width = 0.15
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