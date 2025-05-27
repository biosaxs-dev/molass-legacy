# coding: utf-8
"""
    DenoiseFigures.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from bisect import bisect_right
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from MatrixData import simple_plot_3d
from SvdDenoise import get_denoised_data
from .Qmm3dFigure import find_where_to_zoom_impl

NROWS = 2
NCOLS = 3

class DenoiseFigures(Dialog):
    def __init__(self, parent, dataset, frame):
        self.dataset = dataset
        self.vector = dataset.vector
        self.data = dataset.data
        self.frame = frame
        Dialog.__init__(self, parent, title="Denoise Figures", visible=False)

    def show(self):
        self._show()

    def body(self, bframe):

        fig = plt.figure(figsize=(24,12))
        self.mpl_canvas = FigureCanvasTkAgg(fig, bframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        gs = GridSpec(NROWS,NCOLS)
        axes_list = []
        for i in range(NROWS):
            ax_row = []
            for j in range(NCOLS):
                ax = fig.add_subplot(gs[i,j], projection='3d')
                ax_row.append(ax)
            axes_list.append(ax_row)
        self.axes = np.array(axes_list)

        self.draw_figures()

        fig.tight_layout()

    def draw_figures(self):
        aslice, eslice, w = find_where_to_zoom_impl(self.dataset, self.frame)

        data = self.data[aslice, eslice]
        x = self.vector[aslice]
        y = np.arange(eslice.start, eslice.stop)
        px = x**2

        frame = self.frame

        gy = frame.gy_list[0]
        mu = frame.mu_list[0]
        C = frame.C_list[0]
        print("w=", w, "C.shape=", C.shape)

        Cinv = np.linalg.pinv(C)
        rank = 3

        for i in range(NROWS):
            for j in range(NCOLS):
                ax = self.axes[i,j]
                rank += 3
                M = get_denoised_data(self.data, rank=rank)
                P = np.dot(M, Cinv)
                ax.set_title("Rank=%d" % rank, fontsize=20)
                simple_plot_3d(ax, np.log(M[aslice, eslice]), x=px, y=y, color='green', alpha=0.1)
                for k in w:
                    m = int(round(mu[k]))
                    py = np.ones(len(px))*m
                    z = P[aslice,k]
                    pz = np.log(z*gy[m])
                    ax.plot(px, py, pz, label='c-%d' % k, color='C%d' % (k+2))
                ax.legend()
