"""
    BaselineInpect.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt
from MatrixData import simple_plot_3d
from DataUtils import get_in_folder
from importlib import reload
import Baseline.IntegralBaseline
reload(Baseline.IntegralBaseline)            
from .IntegralBaseline import IntegralBaseline

def baseline_inspect(in_folder, sd):
    print("baseline_inspect", in_folder)

    U, _, wv, uv_curve = sd.get_uv_data_separate_ly()
    M, E, qv, xr_curve = sd.get_xr_data_separate_ly()

    print(U.shape, M.shape)

    uv_stop = bisect_right(wv, 300)
    xr_stop = bisect_right(qv, 0.06)
    U_ = U[0:uv_stop,:]
    M_ = M[0:xr_stop,:]
    wv_ = wv[0:uv_stop]
    qv_ = qv[0:xr_stop]

    uv_mp = sd.absorbance.index
    xr_mp = sd.xray_index

    uv_baseline = IntegralBaseline(uv_curve.x, uv_curve.y)
    xr_baseline = IntegralBaseline(xr_curve.x, xr_curve.y)

    Ub = uv_baseline.get_baseplane(U_, 0, -1)
    Mb = xr_baseline.get_baseplane(M_, 0, -1)

    with plt.Dp():
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20,9), subplot_kw=dict(projection="3d"))
        fig.suptitle("Integral Baseline Inspection for %s" % get_in_folder(in_folder),fontsize=20)
        ax1 = axes[0,0]
        ax2 = axes[0,1]
        ax3 = axes[0,2]
        ax4 = axes[0,3]
        ax5 = axes[1,0]
        ax6 = axes[1,1]
        ax7 = axes[1,2]
        ax8 = axes[1,3]

        ax1.set_title("UV whole",fontsize=16)
        ax2.set_title("UV head",fontsize=16)
        ax3.set_title("UV head baseline",fontsize=16)
        ax4.set_title("UV head corrected",fontsize=16)
        ax5.set_title("XR whole",fontsize=16)
        ax6.set_title("XR head",fontsize=16)
        ax7.set_title("XR head baseline",fontsize=16)
        ax8.set_title("XR head corrected",fontsize=16)
        simple_plot_3d(ax1, U, x=wv)
        simple_plot_3d(ax5, M, x=qv)
        simple_plot_3d(ax2, U_, x=wv_)
        simple_plot_3d(ax6, M_, x=qv_)

        uv_zlim = ax2.get_zlim()
        for ax in ax3, ax4:
            ax.set_zlim(uv_zlim)

        xr_zlim = ax6.get_zlim()
        for ax in ax7, ax8:
            ax.set_zlim(xr_zlim)

        simple_plot_3d(ax3, Ub, x=wv_)
        simple_plot_3d(ax4, U_ - Ub, x=wv_)
        simple_plot_3d(ax7, Mb, x=qv_)
        simple_plot_3d(ax8, M_ - Mb, x=qv_)

        uv_y = uv_curve.x
        uv_z = uv_curve.y
        uv_b = uv_baseline.yb
        uv_x = np.ones(len(uv_y))*wv[uv_mp]
        for ax in ax1, ax2:
            ax.plot(uv_x, uv_y, uv_z, color="blue")
            ax.plot(uv_x, uv_y, uv_b, color="red", alpha=0.5)

        xr_y = xr_curve.x
        xr_z = xr_curve.y
        xr_b = xr_baseline.yb
        xr_x = np.ones(len(xr_y))*qv[xr_mp]
        for ax in ax5, ax6:
            ax.plot(xr_x, xr_y, xr_z, color="green")
            ax.plot(xr_x, xr_y, xr_b, color="red", alpha=0.5)

        fig.tight_layout()
        plt.show()
