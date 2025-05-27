"""
    ProportionalDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from molass_legacy.Models.ElutionCurveModels import egha, emga
from DataUtils import get_in_folder

def demo(in_folder, sd):
    import molass_legacy.KekLib.DebugPlot as plt
    from importlib import reload
    import Models.CfsEvalPeaks
    reload(Models.CfsEvalPeaks)
    from .CfsEvalPeaks import get_cfs_evaluated_peaks
    import Models.Proportional
    reload(Models.Proportional)
    from .Proportional import get_proportional_curves

    print(in_folder)
    ecurve = sd.get_xray_curve()
    x = ecurve.x
    y = ecurve.sy

    num_peaks = len(ecurve.peak_info)
    egha_params, emga_params = get_cfs_evaluated_peaks(x, y, num_peaks, debug=True)
    egh_curve_list, emg_curve_list = get_proportional_curves(x, y, egha_params, emga_params)

    in_folder_ = get_in_folder(in_folder)
    with plt.Dp():
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
        fig.suptitle("Proportional Decomposition Demo using %s" % in_folder_, fontsize=20)

        ax1 = axes[0,0]
        ax2 = axes[0,1]
        ax3 = axes[1,0]
        ax4 = axes[1,1]

        for model, func, ax, params in [
                        ("EGHA", egha, ax1, egha_params),
                        ("EMGA", emga, ax2, emga_params),
                        ]:

            ax.set_title("%s Decomposition using CF Space" % model, fontsize=16)

            ax.plot(x, y)

            ty = np.zeros(len(x))
            for p in params:
                cy = func(x, *p)
                ax.plot(x, cy, ":")
                ty += cy

            ax.plot(x, ty, ":", color="red", lw=3)

        for model, ax, curve_list in [
                        ("EGHA", ax3, egh_curve_list),
                        ("EMGA", ax4, emg_curve_list),
                        ]:

            ax.set_title("%s Proportional Decomposition" % model, fontsize=16)

            ax.plot(x, y)

            ty = np.zeros(len(x))
            for cy in curve_list:
                ax.plot(x, cy, ":")
                ty += cy

            ax.plot(x, ty, ":", color="red", lw=3)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.show()
