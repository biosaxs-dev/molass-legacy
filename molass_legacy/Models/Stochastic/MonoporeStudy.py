"""
    Models.Stochastic.MonoporeStudy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from matplotlib.widgets import Slider, Button
from molass_legacy.Models.Stochastic.MomentUtils import compute_egh_moments
from molass_legacy.Models.Stochastic.MonoporeUtils import compute_monopore_curves

def study(x, y, baseline, model, peaks, peak_rgs, props):
    from importlib import reload
    import Models.Stochastic.RoughGuess
    reload(Models.Stochastic.RoughGuess)
    from molass_legacy.Models.Stochastic.RoughGuess import guess_monopore_params_roughtly
    import Models.Stochastic.MonoporeGuess
    reload(Models.Stochastic.MonoporeGuess)
    from molass_legacy.Models.Stochastic.MonoporeGuess import guess_monopore_params_using_moments

    moments_list = compute_egh_moments(peaks)
    monopore_params = guess_monopore_params_roughtly(x, y, model, peaks, peak_rgs, props, moments_list, debug=True)
    print("peak_rgs = ", peak_rgs)
    print("monopore_params = ", monopore_params)
    if monopore_params is None:
        return

    with plt.Dp(button_spec=["OK", "Cancel"]):
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(16,5))
        fig.suptitle("Stochastic Monopore Model Study", fontsize=20)
        ax1.set_title("EGH Decomposition", fontsize=16)
        ax1.plot(x, y, label='data')
        cy_list = []
        for k, params in enumerate(peaks):
            cy = model(x, params)
            cy_list.append(cy)
            ax1.plot(x, cy, ":", label='component-%d' % k)
        ty = np.sum(cy_list, axis=0)
        ax1.plot(x, ty, ":", color="red", label='model total')
        ax1.plot(x, baseline, color="red", label='baseline')
        ax1.legend()

        ax2.set_title("Monopore Decomposition", fontsize=16)
        ax2.plot(x, y, label='data')
        for M in moments_list:
            m = M[0]
            s = np.sqrt(M[1])
            ax2.axvline(m, color="green")
            ax2.axvspan(m-s, m+s, color="cyan", alpha=0.2)

        cy_list, ty = compute_monopore_curves(x, monopore_params[0:6], peak_rgs, monopore_params[6:])
        artists = []
        for k, cy in enumerate(cy_list):
            curve, = ax2.plot(x, cy, ":", label='component-%d' % k)
            artists.append(curve)
        curve, = ax2.plot(x, ty, ":", color="red", label='model total')
        artists.append(curve)

        seleccted_indeces = [0,1,2,5]
        study_params = monopore_params[seleccted_indeces]
        update_params = monopore_params.copy()
        slider_specs = [    ("N", 300, 4000, study_params[0]),
                            ("T", 0, 3, study_params[1]),
                            ("t0",  -500, 500, study_params[2]),
                            ("poresize", 10, 600, study_params[3]),
                            ]
        def slider_update(k, val):
            # print([k], "slider_update", val)
            study_params[k] = val
            update_params[seleccted_indeces] = study_params
            cy_list, ty = compute_monopore_curves(x, update_params[0:6], peak_rgs, update_params[6:])
            for curve, y_ in zip(artists, cy_list + [ty]):
                curve.set_ydata(y_)
            fig.canvas.draw_idle()

        slider_axes = []
        sliders = []
        for k, (label, valmin, valmax, valinit) in enumerate(slider_specs, start=0):
            ax_ = fig.add_axes([0.8, 0.7 - 0.08*k, 0.11, 0.03])
            slider  = Slider(ax_, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
            slider.on_changed(lambda val, k_=k: slider_update(k_, val))
            slider_axes.append(ax_)
            sliders.append(slider)

        fig.tight_layout()
        fig.subplots_adjust(right=0.7)
        ret = plt.show()

    if not ret:
        return
    
    egh_moments_list = compute_egh_moments(peaks)
    better_monopore_params = guess_monopore_params_using_moments(x, y, egh_moments_list,
                                                                 peak_rgs, props, monopore_params, debug=True)
    return ret