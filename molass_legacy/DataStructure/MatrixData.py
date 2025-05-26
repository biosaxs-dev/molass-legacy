# coding: utf-8
"""
    MatrixData.py

    Copyright (c) 2019-2025, SAXS Team, KEK-PF
"""

import numpy as np
import logging
from mpl_toolkits.mplot3d import Axes3D, proj3d
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from molass_legacy.KekLib.OurMatplotlib3D import Inset2Din3D
from molass.PlotUtils.MatrixPlot import compute_3d_xyz, simple_plot_3d, contour_plot

class MatrixData:
    def __init__(self, data):
        self.logger = logging.getLogger(__name__)
        self.data = data
        self.i = np.arange(data.shape[0])
        self.j = np.arange(data.shape[1])
        self.e_index = None
        self.e_curve = None
        self.i_slice = None
        self.j_slice = None
        self.inset = None

    def plot(self, ax=None, color=None, alpha=1, title=None, ec_color='orange', inset=True):
        import molass_legacy.KekLib.DebugPlot as plt
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            need_show = True
        else:
            need_show = False

        if title is not None:
            ax.set_title(title, fontsize=16, y=1.09)

        x = self.vector
        simple_plot_3d(ax, self.data, x=x, color=color, alpha=alpha)

        if self.e_curve is not None:
            pos = x[self.e_index]
            y_ = self.j
            x_ = np.ones(len(y_))*pos
            z_ = self.e_curve.sy
            ax.plot(x_, y_, z_, color=ec_color)
            for rec in self.e_curve.peak_info:
                top_j = rec[1]
                x_ = self.vector
                y_ = np.ones(len(x_))*top_j
                z_ = self.data[:,top_j]
                ax.plot(x_, y_, z_, color='green')

        fc_exists = self.j_slice is not None and self.j_slice.start is not None and self.j_slice.start > 0
        if inset and self.e_curve is not None and fc_exists:
            islice, jslice = self.get_inset_slice()
            try:
                self.inset = inset = Inset2Din3D(ax, [0.6, 0.7, 0.4, 0.3])
                axins = inset.get_axis()

                x = self.j[jslice]
                index = (islice.start + islice.stop)//2
                y = self.data[index, jslice]
                axins.plot(x, y)
                ylim = axins.get_ylim()
                axins.set_ylim(ylim)
                fx = self.j_slice.start
                axins.plot([fx, fx], ylim, ':', color='yellow', linewidth=3)

                x_ = self.vector[index]
                ymin = x[0]
                ymax = x[-1]
                zmin = np.min(y)
                zmax = np.max(y)
                xi_pos = [x_]*5
                yi_pos = [ymin, ymax, ymax, ymin, ymin]
                zi_pos = [zmin, zmin, zmax, zmax, zmin]
                ax.plot(xi_pos, yi_pos, zi_pos, color='cyan')

                p1 = (x_, ymin, zmax)
                p2 = (x_, ymax, zmin)
                inset.set_annotation_lines(p1, p2)
                inset.draw()
                inset.set_event_handler()   # holding inset as self.inset seems to avoid losing (disabling) handler
            except:
                from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
                etb = ExceptionTracebacker()
                print(etb)

        if need_show:
            fig.tight_layout()
            plt.show()

    def set_elution_curve(self, index, slice_, correct=False):
        self.e_index = index
        self.e_slice = slice_
        y = np.average(self.data[slice_,:], axis=0)
        if correct:
            from LPM import get_corrected
            y = get_corrected(y)
        self.e_y = y
        try:
            self.e_curve = ElutionCurve(y)
        except:
            # e.g., No peak!
            self.e_curve = None
            if True:
                from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
                etb = ExceptionTracebacker()
                print(etb)
        print('self.e_curve=', self.e_curve)

    def set_restriction(self, i_slice, j_slice):
        self.i_slice = i_slice
        self.j_slice = j_slice

    def get_sliced_data(self):
        # print(self.i_slice, self.j_slice)
        return self.data[self.i_slice, self.j_slice], self.vector[self.i_slice]

    def get_inset_slice(self):
        isize = self.data.shape[0]
        jsize = self.data.shape[1]
        iw = int(isize*0.1)
        jw = int(jsize*0.2)
        icenter = int(self.data.shape[0]*0.7)
        jcenter = self.j_slice.start
        istart = max(0, icenter-iw) 
        istop = min(isize, icenter+iw)
        jstart = max(0, jcenter-jw) 
        jstop = min(jsize, jcenter+jw)
        return slice(istart, istop), slice(jstart, jstop)
