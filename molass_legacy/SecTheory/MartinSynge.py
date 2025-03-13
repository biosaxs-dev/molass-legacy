"""
    SecTheory.MartinSynge.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.widgets import Slider, Button
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Peaks.ElutionModels import gaussian

def num_plates_demo():
    tR = 500
    tI = 0
    sigma = 10
    x = np.arange(1000)

    def compute_num_plates(tR, sigma):
        return (tR/sigma)**2

    def compute_pdf(tR, sigma):
        y = gaussian(x, 1, tR, sigma)
        y /= np.sum(y)
        return y

    N = compute_num_plates(tR, sigma)
    y = compute_pdf(tR, sigma)

    init_params = [tR, sigma]
    slider_params = init_params.copy()

    def set_title(ax, N):
        ax.set_title(r"Theoretical Number of Plates: $ N = (\frac{t_R}{\sigma})^2 = %d $" % round(N), fontsize=20)

    with plt.Dp():
        fig, ax = plt.subplots(figsize=(12, 5))
        set_title(ax, N)
        ax.set_xlabel("Time (Frames)")
        ax.set_ylabel("Density")
        curve, = ax.plot(x, y)
        ax.axvline(x=0, label='$t_I$', color='gray')
        line = ax.axvline(x=tR, label='$t_R$', color='red')
        ax.legend()
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, 0.1)
        fig.tight_layout()
        fig.subplots_adjust(right=0.7)

        slider_specs = [    (r"$t_R$", 0, 1000, slider_params[0]),
                            (r"$\sigma$", 1, 50, slider_params[1]),
                            ]

        def slider_update(k, val):
            # print([k], "slider_update", val)
            slider_params[k] = val
            tR, sigma = slider_params
            line.set_xdata(tR)
            y = compute_pdf(tR, sigma)
            curve.set_data(x, y)
            N = compute_num_plates(tR, sigma)
            set_title(ax, N)
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

        plt.show()