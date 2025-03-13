"""
    Models.Stochastic.MonoporeUtilsImpl.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from SecTheory.BasicModels import robust_single_pore_pdf

def compute_monopore_curves(x, mnp_colparams, rgs, scales, return_moments=False):
    N, T, x0, me, mp, poresize = mnp_colparams
    rhov = rgs/poresize
    rhov[rhov > 1] = 1
    cy_list = []
    moments = []
    for rho, scale in zip(rhov, scales):
        ni_ = N * (1 - rho)**me
        ti_ = T * (1 - rho)**mp
        cy = scale * robust_single_pore_pdf(x - x0, ni_, ti_)
        cy_list.append(cy)
        M1 = x0 + N * T * (1 - rho)**(me + mp)
        M2 = 2 * N * T**2 * (1 - rho)**(me + 2*mp)
        moments.append((M1, M2))
    ty = np.sum(cy_list, axis=0)
    if return_moments:
        return cy_list, ty, moments
    else:
        return cy_list, ty

def plot_monopore_component_with_sliders_impl(title, x, y, peak_rgs, mnp_params, x0_upper, cy_list, ty, moments, egh_moments_list):
    from matplotlib.widgets import Slider, Button
    from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

    print("moments=", moments)

    plot_params = np.asarray(mnp_params).copy()
    print("plot_params=", plot_params)

    slider_specs = [    ("N", 0, 6000, mnp_params[0]),
                        ("T", 0, 3, mnp_params[1]),
                        ("t0",  -1000, 1000, mnp_params[2]),
                        ("me", 0, 3, mnp_params[3]),
                        ("mp", 0, 3, mnp_params[4]),
                        ("poresize", 20, 400, mnp_params[5]),
                        ]

    num_peaks = len(peak_rgs)
    with plt.Dp():
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(18, 5))
        fig.suptitle(title)
        ax2.set_title("Moments Fitting for %s" % get_in_folder())
        ax2.set_xlabel("Time (frames)")
        ax2.set_ylabel("Intensity")
    
        mlines = []
        ecurves = []
        for i, ax in enumerate((ax1,ax2)):
            ax.plot(x, y, color='orange', label='data')
            curve, = ax.plot(x, ty, ':', color="red", label='model total')
            if i == 1:
                ecurves.append(curve)
            for k, (cy, M, eghM) in enumerate(zip(cy_list, moments, egh_moments_list)):
                curve, = ax.plot(x, cy, ':', label='component-%d' % k)
                if i == 1:
                    ecurves.append(curve)
                label = None if k < num_peaks - 1 else "Monopore Moments"
                line = ax.axvline(x=M[0], color="green", label=label)
                if i == 1:
                    mlines.append(line)
                label = None if k < num_peaks - 1 else "EGH Moments"
                ax.axvline(x=eghM[0], ls=":", lw=3, color="cyan", label=label)
            ax.axvline(x=x0_upper, color="pink")

        def slider_update(k, val):
            print([k], "slider_update", val)
            plot_params[k] = val
            cy_list_, ty_, moments_ = compute_monopore_curves(x, plot_params[0:6], peak_rgs, plot_params[6:], return_moments=True)
            for cy, ecurve in zip([ty_] + cy_list_, ecurves):
                ecurve.set_ydata(cy)
            print("moments_=", moments_)
            for line, M in zip(mlines, moments_):
                line.set_xdata(M[0])
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

        # xmin, xmax = ax1.get_xlim()
        # ymin, ymax = ax1.get_ylim()
        m1, m2 = moments[0]
        xmin = m1 - 10*np.sqrt(m2)
        m1, m2 = moments[-1]
        xmax = m1 + 10*np.sqrt(m2)                
        ax2.set_xlim(xmin, xmax)
        ax2.legend()
        fig.tight_layout()
        fig.subplots_adjust(right=0.7)
        ret = plt.show()

    return ret