# coding: utf-8
"""
    BaseEstimater.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt 

"""
    
"""
class BaseEstimater:
    def __init__(self, D, E=None, base_type=None):
        self.base_plane = np.zeros(D.shape)

    def get_baseline(self, i):
        return self.base_plane[i,:]

def demo(in_folder):
    from molass_legacy._MOLASS.SerialSettings import get_setting
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from MatrixData import simple_plot_3d

    sp = StandardProcedure()
    sp.load(in_folder, debug=False)
    sd = sp.get_sd()
    D, E, qv, xr_curve = sd.get_xr_data_separate_ly()

    U, _, wv, uv_curve = sd.get_uv_data_separate_ly()
    # print(U.shape, wv.shape, uv_curve.x.shape)

    q1 = get_setting('intensity_picking')
    q2 = q1*2
    w1 = get_setting('absorbance_picking_sub')
    w2 = get_setting('absorbance_picking')

    for curve, D_, vector, v1, v2, vname, colors in [
                                (xr_curve, D, qv, q1, q2, 'q', ('orange', 'green')),
                                (uv_curve, U, wv, w1, w2, 'wavelength', ('cyan', 'blue')),
                                ]:

        x = curve.x
        y = curve.y

        i1 = bisect_right(vector, v1)
        i2 = bisect_right(vector, v2)

        be = BaseEstimater(D_)
        b1 = be.get_baseline(i1)
        b2 = be.get_baseline(i2)

        plt.push()
        fig = plt.figure(figsize=(21, 7))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        fig.suptitle("Baseline Estimation using Decomposition-Rg Consistency", fontsize=24)
        ax1.set_title("3D View", fontsize=20)
        ax2.set_title("Elution at %s=%.03g" % (vname, v1), fontsize=20)
        ax3.set_title("Elution at %s=%.03g" % (vname, v2), fontsize=20)

        simple_plot_3d(ax1, D_, x=vector)
        y_ = x

        y2 = np.average(D_[i2-5:i2+6,:], axis=0)

        for v_, z_, color in [(v1, y, colors[0]), (v2, y2, colors[1])]:
            x_ = np.ones(len(x))*v_
            ax1.plot(x_, y_, z_, color=color)

        ax2.plot(x, y, color=colors[0])
        ax2.plot(x, b1, ':', color='red')

        ax3.plot(x, y2, color=colors[1])
        ax3.plot(x, b2, ':', color='red')
        ymin2, ymax2 = ax2.get_ylim()
        ymin3, ymax3 = ax3.get_ylim()
        ymin = min(ymin2, ymin3)
        ymax = max(ymax2, ymax3)
        for ax in [ax2, ax3]:
            ax.set_ylim(ymin, ymax)

        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        plt.show()
        plt.pop()
