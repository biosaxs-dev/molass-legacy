"""
    CfsEvalPeaksDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from matplotlib.gridspec import GridSpec
from molass_legacy.KekLib.SciPyCookbook import smooth
from DataUtils import get_in_folder

VERY_SMALL_VALUE = 1e-6

def demo(in_folder, sd, extra_peaks=0):
    import molass_legacy.KekLib.DebugPlot as plt
    from molass_legacy.Models.ElutionCurveModels import egha, emga
    from SecTheory.Edm import edm_func
    from importlib import reload
    import Models.CfsEvalPeaks
    reload(Models.CfsEvalPeaks)
    from .CfsEvalPeaks import get_cfs_evaluated_peaks

    print(in_folder)
    ecurve = sd.get_xray_curve()
    x = ecurve.x
    y = ecurve.y

    num_peaks = len(ecurve.peak_info) + extra_peaks
    egh_peak_list, emg_peak_list, edm_peak_list = get_cfs_evaluated_peaks(x, y, num_peaks, using_cfs=False, debug=True)
    print("egh_peak_list=", egh_peak_list)
    print("emg_peak_list=", emg_peak_list)

    in_folder_ = get_in_folder(in_folder)
    with plt.Dp():
        fig = plt.figure(figsize=(20,12))
        fig.suptitle("Fitting Difference Demo between PDF and CF using %s" % in_folder_, fontsize=24)

        gs = GridSpec(3,15)
        ax0 = fig.add_subplot(gs[0,0])

        ax1 = fig.add_subplot(gs[0,1:3])
        ax2 = fig.add_subplot(gs[0,3:7])
        ax3 = fig.add_subplot(gs[0,7:11])
        ax4 = fig.add_subplot(gs[0,11:15])

        ax5 = fig.add_subplot(gs[1,0])
        ax6 = fig.add_subplot(gs[1,1:3])
        ax7 = fig.add_subplot(gs[1,3:7])
        ax8 = fig.add_subplot(gs[1,7:11])
        ax9 = fig.add_subplot(gs[1,11:15])

        axa = fig.add_subplot(gs[2,0])
        axb = fig.add_subplot(gs[2,1:3])
        axc = fig.add_subplot(gs[2,3:7])
        axd = fig.add_subplot(gs[2,7:11])
        axe = fig.add_subplot(gs[2,11:15])

        ax0.set_title("No", fontsize=16)
        ax1.set_title("Space", fontsize=16)
        ax2.set_title("EGHA", fontsize=16)
        ax3.set_title("EMGA", fontsize=16)
        ax4.set_title("EDM", fontsize=16)

        def plot_params(ax0, ax1, ax2, ax3, ax4, text0, text1, egh_peak_list, emg_peak_list, edm_peak_list, proportional=False):

            for ax, text, fontsize in (ax0, text0, 50), (ax1, text1, 16):
                ax.grid(False)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_facecolor("white")
                ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=fontsize)

            for func, ax, peak_list in [
                            (egha, ax2, egh_peak_list),
                            (emga, ax3, emg_peak_list),
                            (edm_func, ax4, edm_peak_list),
                            ]:
                ax.plot(x, y)

                cy_list = []
                for p in peak_list:
                    cy = func(x, *p)
                    cy_list.append(cy)
                ty = np.sum(cy_list, axis=0)

                if proportional:
                    sy = smooth(y)
                    cy_list_ = []
                    ty[ty < VERY_SMALL_VALUE] = VERY_SMALL_VALUE
                    ty_ = np.zeros(len(x))
                    for cy in cy_list:
                        cy_ = sy * cy/ty
                        cy_list_.append(cy_)
                        ty_ += cy_
                else:
                    ty_ = ty
                    cy_list_ = cy_list

                for cy in cy_list_:
                    ax.plot(x, cy, ":")

                ax.plot(x, ty_, ":", color="red", lw=3)

        plot_params(ax0, ax1, ax2, ax3, ax4, "1", "Elution Space\nevaluated", egh_peak_list, emg_peak_list, edm_peak_list)

        egh_peak_list, emg_peak_list, edm_peak_list = get_cfs_evaluated_peaks(x, y, num_peaks, using_cfs=True, debug=True)
        plot_params(ax5, ax6, ax7, ax8, ax9, "2", "CF Space\nevaluated", egh_peak_list, emg_peak_list, edm_peak_list)
        plot_params(axa, axb, axc, axd, axe, "3", "CF Space\nevaluated\n\nwith\n\nProportional\nAllocation", egh_peak_list, emg_peak_list, edm_peak_list, proportional=True)

        fig.tight_layout()
        fig.subplots_adjust(top=0.88, left=0.05,  bottom=0.08, right=0.95)
        plt.show()
