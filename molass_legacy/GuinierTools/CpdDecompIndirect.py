"""
    GuinierTools.CpdDecompIndiect.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.ElutionCurveModels import EGHA
from RgProcess.RgCurveUtils import plot_rg_curve
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from GuinierTools.RgCurveUtils import get_connected_curve_info

def cpd_spike_impl(editor):
    from importlib import reload
    import Peaks.PeakEditorUtils
    reload(Peaks.PeakEditorUtils)
    from molass_legacy.Peaks.PeakEditorUtils import apply_new_peaks
    import GuinierTools.CpdDecompIndirectImpl
    reload(GuinierTools.CpdDecompIndirectImpl)
    from GuinierTools.CpdDecompIndirectImpl import compute_end_points, imporove_decomposition

    print("cpd_spike_impl")
    model = EGHA()
    x, y, peaks, baseline = editor.xr_draw_info
    rg_curve = editor.dsets[1]

    x_, y_, rgv, qualiteis, valid_bools = get_connected_curve_info(rg_curve)

    nc = 4
    end_points = compute_end_points(nc, x_, rgv)
    new_peaks = imporove_decomposition(x, y, model, peaks, end_points)

    in_folder = get_in_folder()

    def plot_decomposition(ax, peaks):
        ax.plot(x, y, color='orange')
        cy_list = []
        for i, peak in enumerate(peaks):
            cy = model(x, peak)
            cy_list.append(cy)
            ax.plot(x, cy, ":", label='component-%d' % (i+1))
        ty = np.sum(cy_list, axis=0)
        ax.plot(x, ty, ":", color='red')
        axt = ax.twinx()
        axt.grid(False)
        plot_rg_curve(axt, rg_curve, label='Rg Curve')
        for i in range(nc):
            x1 = end_points[i]
            x2 = end_points[i+1]
            axt.axvspan(x1, x2, color='C%d'%i, alpha=0.1)
        ymin, ymax = axt.get_ylim()
        axt.set_ylim(15, ymax)
        ax.legend()
        axt.legend(loc='center right')

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        fig.suptitle("Better Decomposition using CPD of Rg-curve on %s" % in_folder, fontsize=20)
        ax1.set_title("Initial Decomposition", fontsize=16)
        ax2.set_title("Improved Decomposition", fontsize=16)
        plot_decomposition(ax1, peaks)
        plot_decomposition(ax2, new_peaks)
        fig.tight_layout()
        ret = plt.show()
    
    if ret:
        print("apply")
        apply_new_peaks(editor, new_peaks, debug=False)