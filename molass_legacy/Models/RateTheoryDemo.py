"""
    RateTheoryDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.gridspec import GridSpec
import molass_legacy.KekLib.DebugPlot as plt
from DataUtils import get_in_folder
from molass_legacy.Models.ElutionCurveModels import EGHA, EMGA

def compare_models(in_folder, sd):
    from importlib import reload
    import Models.RateTheory.EDM
    reload(Models.RateTheory.EDM)
    from .RateTheory.EDM import guess, edm_impl

    ecurve = sd.get_xray_curve()
    x = ecurve.x
    y = ecurve.y

    model = EGHA()
    g_params = model.guess(y, x=x)
    ret = model.fit(y, g_params, x=x)
    y_egh = model.eval(ret.params, x=x)

    model = EMGA()
    g_params = model.guess(y, x=x)
    ret = model.fit(y, g_params, x=x)
    y_emg = model.eval(ret.params, x=x)

    params = guess(x, y)
    print("params=", params)

    with plt.Dp():
        fig = plt.figure(figsize=(18,7))
        fig.suptitle("Model Comparison using %s" % get_in_folder(in_folder), fontsize=20)
        gs = GridSpec(4,3)
        ax11, ax12, ax13, ax21, ax22, ax23 = [ fig.add_subplot(gs_) for gs_ in (gs[0:3,0], gs[0:3,1], gs[0:3,2], gs[3,0], gs[3,1], gs[3,2])]
        ax11.set_title("EGH", fontsize=16)
        ax12.set_title("EMG", fontsize=16)
        ax13.set_title("EDM", fontsize=16)

        def draw_rmsd(ax1, rmsd):
            tx = np.mean(ax1.get_xlim())
            ty = np.mean(ax1.get_ylim())
            text = "rmsd=%.2g" % rmsd
            ax1.text(tx, ty, text, va="center", ha="center", fontsize=40, alpha=0.3)

        rmsds = []

        ax11.plot(x, y)
        ax11.plot(x, y_egh)
        div = y_egh - y
        ax21.plot(x, div)
        rmsd = np.sqrt(np.mean(div**2))
        rmsds.append(rmsd)
        draw_rmsd(ax11, rmsd)

        ax12.plot(x, y)
        ax12.plot(x, y_emg)
        div = y_emg - y
        ax22.plot(x, div)
        rmsd = np.sqrt(np.mean(div**2))
        rmsds.append(rmsd)
        draw_rmsd(ax12, rmsd)

        ax13.plot(x, y)
        y_ = edm_impl(x, *params)
        ax13.plot(x, y_)
        div = y_ - y
        ax23.plot(x, div)
        rmsd = np.sqrt(np.mean(div**2))
        rmsds.append(rmsd)
        draw_rmsd(ax13, rmsd)

        ylims = []
        for ax1 in ax21, ax22, ax23:
            ylims.append(ax1.get_ylim())

        ylims = np.array(ylims)
        ymin = np.min(ylims[:,0])
        ymax = np.max(ylims[:,1])

        for ax1 in ax21, ax22, ax23:
            ax1.set_ylim(ymin, ymax)

        m = np.argmin(rmsds)
        for k, axes in enumerate(((ax11, ax21), (ax12, ax22), (ax13, ax23))):
            if k == m:
                for ax1 in axes:
                    ax1.set_facecolor("peachpuff")

        fig.tight_layout()
        plt.show()

def edm_inspect(in_folder, sd):
    import warnings
    warnings.filterwarnings("ignore")
    from matplotlib.widgets import Slider, Button
    from importlib import reload
    import Models.RateTheory.EDM
    reload(Models.RateTheory.EDM)
    from .RateTheory.EDM import guess, edm_impl

    ecurve = sd.get_xray_curve()
    x = ecurve.x.copy()
    y = ecurve.y

    x += 0

    params = guess(x, y)
    print("params=", params)
    fullparams = np.concatenate([params, [0, 0.0001, 2.0, 30.0, 30.0]])

    slider_specs = [    ("t0", -50, 100, fullparams[0]),
                        ("u", 0, 2, fullparams[1]),
                        ("a", 0, 2, fullparams[2]),
                        ("b", -4, 1, fullparams[3]),
                        ("e", 0, 2, fullparams[4]),
                        ("Dz", 0, 1, fullparams[5]),
                        ("cinj", 0, 10, fullparams[6]),
                        ("cinit", 0, 1, fullparams[7]),
                        ("c0", 0, 3, fullparams[8]),
                        ("tinj", 0, 3, fullparams[9]),
                        ("L", 0, 35, fullparams[10]),
                        ("z", 0, 35, fullparams[11]),
                        ]

    in_folder = get_in_folder(in_folder)
    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(24,10))
        dp = plt.get_dp()

        fig.suptitle("EDM Params Inspection for %s" % in_folder, fontsize=32)
        ax2.set_axis_off()

        ax1.plot(x, y)
        edm_line, = ax1.plot(x, edm_impl(x, *params))

        def slider_update(k, val):
            print([k], "slider_update", val)
            params[k] = val
            y_ = edm_impl(x, *params)
            edm_line.set_data(x, y_)
            dp.draw()

        slider_axes = []
        sliders = []
        for k, (label, valmin, valmax, valinit) in enumerate(slider_specs, start=0):
            ax_ = fig.add_axes([0.6, 0.8 - 0.06*k, 0.3, 0.03])
            slider  = Slider(ax_, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
            slider.label.set_size(16)
            slider.on_changed(lambda val, k_=k: slider_update(k_, val))
            slider_axes.append(ax_)
            sliders.append(slider)

        def reset_slider_valules(event):
            for slider in sliders:
                slider.reset()
            dp.draw()

        axreset = fig.add_axes([0.9, 0.0, 0.1, 0.075])
        breset = Button(axreset, 'Reset')
        breset.label.set_fontsize(16)
        breset.on_clicked(reset_slider_valules)

        ax1.tick_params(axis='x', labelsize=16)
        ax1.tick_params(axis='y', labelsize=16)

        fig.tight_layout()
        plt.show()

