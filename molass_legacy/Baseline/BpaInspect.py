# coding: utf-8
"""
    Baseline.BpaInspect.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import copy
from bisect import bisect_right
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.DENSS.DenssUtils import fit_data
import molass_legacy.KekLib.DebugPlot as dplt
from molass_legacy._MOLASS.SerialSettings import get_setting
from MatrixData import simple_plot_3d
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from LPM import LPM_3d
from DataUtils import cut_upper_folders
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import BlinkingFrame
from molass_legacy.Mapping.MapperConstructor import create_mapper
from DataUtils import get_in_folder
import molass_legacy.KekLib.DebugPlot as dplt

class DebugPlotter:
    def __init__(self, sd, indeces):
        self.sd = sd
        self.qv = sd.qvector
        self.indeces = indeces
        ncols = len(indeces)
        fig, axes = dplt.subplots(nrows=1, ncols=ncols, figsize=(4.4*ncols, 6))
        fig.suptitle("LPM Debug for " + get_in_folder(), fontsize=20)
        self.fig = fig
        self.axes = axes
        self.bases = []

    def plot(self, i, base):
        try:
            w = np.where(self.indeces == i)[0]
            if len(w) == 0:
                return
            k = w[0]
        except:
            return

        self.bases.append(base)
        ax = self.axes[k]
        ax.set_title("Q[%d]=%.4g" % (i, self.qv[i]), fontsize=16)
        x = self.sd.xray_curve.x
        y = self.sd.intensity_array[:,i,1]
        y_ = y + base
        ax.plot(x, y_, color='orange')
        ax.plot(x, base, color='red')

    def show(self):
        ax_lims = []
        for ax in self.axes:
            ax_lims.append(ax.get_ylim())
        ax_lims = np.array(ax_lims)
        ymin = np.min(ax_lims[:,0])
        ymax = np.max(ax_lims[:,1])
        for ax in self.axes:
            ax.set_ylim(ymin, ymax)
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.85)
        dplt.show()

class BpaInspector(Dialog):
    def __init__(self, parent, sd):
        self.parent = parent
        self.sd1 = sd.get_copy()
        self.sd2 = sd
        self.qv = sd.qvector
        self.i1 = bisect_right(sd.qvector, 0.02)
        self.i2 = bisect_right(sd.qvector, 0.1)

        self.debug_obj = DebugPlotter(sd, np.arange(self.i2-2,self.i2+3))

        self.mapper = create_mapper(None, sd, full_width=True)  # full_width=True is required
        self.bp = self.sd2.apply_baseline_correction(self.mapper.get_mapped_info(), return_base=True, debug_obj=self.debug_obj)

        self.debug_obj.show()

        D1, E1, _, _ = self.sd1.get_xr_data_separate_ly()
        D2, E1, qv, ecurve = self.sd2.get_xr_data_separate_ly()
        self.D1 = D1
        self.D2 = D2

        self.x = np.arange(D1.shape[1])
        self.zeros = np.zeros(len(self.x ))
        self.axis_ylims = None

        self.in_folder = get_setting('in_folder')
        Dialog.__init__(self, parent, "BPA Inspector", visible=False)

    def show(self):
        self._show()

    def body(self, frame):
        cframe = Tk.Frame(frame)
        cframe.pack()
        bframe = Tk.Frame(frame)
        bframe.pack(fill=Tk.X)
        tframe = Tk.Frame(bframe)
        tframe.pack(side=Tk.LEFT)
        pframe = Tk.Frame(bframe)
        pframe.pack(side=Tk.RIGHT)

        self.fig = fig = plt.figure(figsize=(21, 11))
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        gs = GridSpec(2,3)
        ax00 = fig.add_subplot(gs[0,0], projection='3d')
        ax01 = fig.add_subplot(gs[0,1])
        ax02 = fig.add_subplot(gs[0,2])
        ax10 = fig.add_subplot(gs[1,0], projection='3d')
        ax11 = fig.add_subplot(gs[1,1])
        ax12 = fig.add_subplot(gs[1,2])
        self.axes = np.array([[ax00, ax01, ax02], [ax10, ax11, ax12]])

        fig.suptitle("Base Plane Inspection for " + get_in_folder(), fontsize=20)
        ax01.set_title("Q[%d]=%.4g" % (self.i1, self.qv[self.i1]), fontsize=16)
        ax02.set_title("Q[%d]=%.4g" % (self.i2, self.qv[self.i2]), fontsize=16)
        self.draw_3d(ax00, self.D1)
        self.draw_3d(ax10, self.D2)
        self.draw_ecurve_at(ax01, self.D1, self.i1, baseline=True)
        self.draw_ecurve_at(ax02, self.D1, self.i2, baseline=True, base=self.debug_obj.bases[2])
        self.draw_ecurve_at(ax11, self.D2, self.i1, zeroline=True)
        self.draw_ecurve_at(ax12, self.D2, self.i2, zeroline=True )

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        self.mpl_canvas.draw()

    def buttonbox( self ):
        bottom_frame = Tk.Frame(self)
        bottom_frame.pack(fill=Tk.BOTH, expand=1)

        width = int(self.mpl_canvas_widget.cget('width'))
        padx = width*0.05

        tframe = Tk.Frame(bottom_frame)
        tframe.pack(side=Tk.LEFT, padx=padx)
        self.toolbar = NavigationToolbar( self.mpl_canvas, tframe )
        self.toolbar.update()

        space = Tk.Frame(bottom_frame, width=width*0.25)
        space.pack(side=Tk.RIGHT)

        box = Tk.Frame(bottom_frame)
        box.pack(side=Tk.RIGHT)

        self.same_scale = Tk.IntVar()
        self.same_scale.set(0)
        w = Tk.Checkbutton(box, text="same scale", variable=self.same_scale)
        w.pack(side=Tk.LEFT, padx=20, pady=5)
        self.same_scale.trace("w", self.same_scale_tracer)

        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=20, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def draw_3d(self, ax, D):
        simple_plot_3d(ax, D, x=self.qv)

    def draw_ecurve_at(self, ax, D, i, baseline=False, zeroline=False, base=None):
        y = D[i,:]
        ax.plot(self.x, y, color='orange')

        if base is not None:
            ax.plot(self.x, base, color='pink', label='LPM base')

        if baseline:
            by = self.bp[i,:]
            ax.plot(self.x, by, color='red', label='BPA base')

        if zeroline:
            ax.plot(self.x, self.zeros, ':', color='red')

        if base is not None:
            ax.legend()

    def same_scale_tracer(self, *args):
        if self.axis_ylims is None:
            ylims = []
            for i in range(2):
                row = []
                for j in range(3):
                    ax = self.axes[i,j]
                    row.append(ax.get_ylim())
                ylims.append(row)
            self.axis_ylims = np.array(ylims)

        same_scale = self.same_scale.get()
        for i in range(2):
            ax = self.axes[i,2]
            j = 1 if same_scale else 2
            ylim = self.axis_ylims[i,j]
            ax.set_ylim(ylim)

        self.mpl_canvas.draw()
