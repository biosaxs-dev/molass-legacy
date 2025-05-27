# coding: utf-8
"""
    MdViewer.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from MatrixData import compute_3d_xyz

class MdViewer(Dialog):
    def __init__(self, parent, D, xvector=None):
        self.D = D
        if xvector is None:
            xvector = np.arange(D.shape[0])
        self.xvector = xvector
        self.yvector = np.arange(D.shape[1])
        self.xones = np.ones(D.shape[1])
        self.i = bisect_right(xvector, 0.02)
        self.artists = None
        Dialog.__init__(self, parent, "MdViewer", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tframe = Tk.Frame(body_frame)
        tframe.pack(fill=Tk.X)

        self.fig = fig = plt.figure(figsize=(14,7))
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        gs = GridSpec(1,2)
        ax1 = fig.add_subplot(gs[0,0], projection='3d')
        ax2 = fig.add_subplot(gs[0,1])
        axs = fig.add_axes([0.05, 0.01, 0.4, 0.03])
        self.axes = [ax1, ax2, axs]

        xx, yy, zz = compute_3d_xyz(self.D, x=self.xvector)
        ax1.plot_surface(xx, yy, zz, alpha=0.5)

        self.draw_elution()
        self.slider = Slider(axs, 'X', self.xvector[0], self.xvector[-1], valinit=self.xvector[self.i])

        self.slider.on_changed(self.on_x_change)

        fig.tight_layout()
        self.mpl_canvas.draw()

    def on_x_change(self, val):
        # print(val)
        self.i = bisect_right(self.xvector, val)
        self.draw_elution()
        self.mpl_canvas.draw()

    def draw_elution(self):
        ax1, ax2 = self.axes[0:2]
        q = self.xvector[self.i]
        x = self.xones*q
        y = self.yvector
        z = self.D[self.i,:]
        if self.artists is None:
            eline3d, = ax1.plot(x, y, z, color='orange')
            eline2d, = ax2.plot(y, z, color='orange')
            self.artists = [eline3d, eline2d]
        else:
            eline3d, eline2d = self.artists
            eline3d.set_data(x, y)
            eline3d.set_3d_properties(z)
            eline2d.set_data(y, z)

        zmin = np.min(z)
        zmax = np.max(z)
        ax2.set_ylim(zmin, zmax)
