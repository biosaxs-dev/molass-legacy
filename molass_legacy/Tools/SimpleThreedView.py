"""
    SimpleThreedView.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import molass_legacy.KekLib.DebugPlot as plt
from MatrixData import simple_plot_3d
from DataUtils import get_in_folder

def show_simple_3d_view(sd):
    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6), subplot_kw=dict(projection="3d"))
        fig.suptitle("Simple 3D View of %s" % get_in_folder(), fontsize=20)

        ax1.set_title("UV Data", fontsize=16)
        D, _, wv, uv_curve = sd.get_uv_data_separate_ly()
        simple_plot_3d(ax1, D, x=wv)

        ax2.set_title("Xray Data", fontsize=16)
        D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
        simple_plot_3d(ax2, D, x=qv)

        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        plt.show()
