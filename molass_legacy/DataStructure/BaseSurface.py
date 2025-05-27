# coding: utf-8
"""
    BaseSurface.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import logging
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from MatrixData import simple_plot_3d
from molass_legacy.Baseline.ScatteringBaseline import ScatteringBaseline
from ScatteringBasecurve import ScatteringBasecurve
from ScatteringBasespline import ScatteringBasespline
from molass_legacy.KekLib.SciPyCookbook import smooth
from ThreeDimUtils import compute_plane
from molass_legacy.Baseline.LambertBeer import BasePlane

class BaseSurface:
    def __init__(self, xd):
        self.logger = logging.getLogger(__name__)
        data, vector = xd.get_sliced_data()
        self.data = data
        self.vector = vector
        self.e_curve = xd.e_curve  # TODO
        self.xd = xd

        ends_array, ends_array_smoothed = self.compute_lpm_baselines()
        base_plane = self.lines_to_surface(ends_array)
        base_plane_smoothed = self.lines_to_surface(ends_array_smoothed)

        corrected_data = data - base_plane_smoothed

        index = xd.e_index    # TODO
        j0 = xd.j_slice.start
        bp = BasePlane(corrected_data, index, self.e_curve, j0=0 if j0 is None else j0)
        a, b, c = bp.solve()
        x = bp.x
        y = bp.y
        base_plane_adjusted  = compute_plane(a, b, c, x, y)
        adjusted_data = corrected_data - base_plane_adjusted

        if True:
            from DataUtils import get_in_folder
            plt.push()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Non-Corrected Data", fontsize=20)
            simple_plot_3d(ax, xd.data, x=xd.vector, alpha=0.3)
            fig.tight_layout()
            plt.show()
            plt.pop()

            fig = plt.figure(figsize=(21,11))
            fig.suptitle("Correction using LPM+MF Base Surface for " + get_in_folder(), fontsize=20)
            ax1 = fig.add_subplot(231, projection='3d')
            ax2 = fig.add_subplot(232, projection='3d')
            ax3 = fig.add_subplot(233, projection='3d')
            ax4 = fig.add_subplot(234, projection='3d')
            ax5 = fig.add_subplot(235, projection='3d')
            ax6 = fig.add_subplot(236, projection='3d')

            # xd.plot(ax, color='yellow', alpha=0.1)
            ax1.set_title("Corrected with Base Lines", fontsize=20)
            ax2.set_title("Corrected with Base Surface", fontsize=20)
            ax3.set_title("Adjusted with Matrix Factorization", fontsize=20)
            simple_plot_3d(ax1, data - base_plane, x=vector, alpha=0.3)
            simple_plot_3d(ax2, corrected_data, x=vector, alpha=0.3)
            simple_plot_3d(ax3, adjusted_data, x=vector, alpha=0.3)
            ax4.set_title("LPM Base Lines", fontsize=20)
            ax5.set_title("Smoothed Base Surface", fontsize=20)
            ax6.set_title("Matrix Factorization-Adjust Plane", fontsize=20)
            simple_plot_3d(ax4, ends_array, x=vector, color='red', alpha=0.3)
            simple_plot_3d(ax5, base_plane_smoothed, x=vector, color='red', alpha=0.3)
            adjusted_plane = compute_plane(a, b, c, x[[0,-1]], y[[0,-1]])
            simple_plot_3d(ax6, base_plane_smoothed, x=vector, color='red', alpha=0.1)
            simple_plot_3d(ax6, adjusted_plane, x=vector[[0,-1]], y=y[[0,-1]], color='blue', alpha=0.3)
            fig.tight_layout()
            fig.subplots_adjust(top=0.92)
            plt.show()

    def compute_lpm_baselines(self):
        end_points_list = []
        y = np.array([0, self.data.shape[1]-1])
        for i in range(self.data.shape[0]):
            z = self.data[i,:]
            sbl = ScatteringBaseline( z, curve=self.e_curve, logger=self.logger )
            D, E  = sbl.solve()     # p_final=p_final
            end_points_list.append(D*y + E)
        work_array = np.array(end_points_list)
        work_array_smoothed = np.vstack([smooth(work_array[:,j], window_len=40) for j in range(2)]).T
        return work_array, work_array_smoothed

    def lines_to_surface(self, lines):
        w = np.linspace(0, 1, self.data.shape[1])
        W = np.vstack([1-w, w])
        return np.dot(lines, W)
