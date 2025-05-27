# coding: utf-8
"""
    SecTools.CorMap.PairwiseMap.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from MatrixData import simple_plot_3d

class PairwiseMap(Dialog):
    def __init__(self, parent, M, j_pair, qv, datcmp_data):
        self.M = M
        self.j_pair = j_pair
        self.j_pair_ = tuple([j+1 for j in j_pair])
        self.R = np.corrcoef(M[:,j_pair])
        self.qv = qv
        self.datcmp_data = datcmp_data
        self.pairwise_point = None
        self.pairwise_lines = None
        Dialog.__init__(self, parent, "Pairwise Map", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()

        fig = plt.figure(figsize=(18,6))
        self.fig = fig
        ax1 = fig.add_subplot(131, projection="3d")
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.set_title("Pair Frames in 3D", fontsize=16)
        ax2.set_title("Pair Correlation between Frames(%d, %d)" % self.j_pair, fontsize=16)
        ax3.set_title("P-value in Schilling Distribution (n=%d)" % len(self.qv), fontsize=16)
        self.axes = ax1, ax2, ax3

        self.draw_pairframes_3d(ax1)
        self.draw_pairwise_correlation(ax2)
        self.draw_pvalue_in_schilling(ax3)

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.1)

        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas.mpl_connect('button_press_event', self.on_figure_click)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=50, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def draw_pairframes_3d(self, ax):
        ax.set_xlabel("$q(\AA^{-1})$")
        ax.set_ylabel("Frame No.")
        ax.set_zlabel("Intensity")
        simple_plot_3d(ax, self.M, x=self.qv)
        x = self.qv
        for j in self.j_pair:
            y = np.ones(len(x)) * j
            z = self.M[:,j]
            ax.plot(x, y, z, label="Frame No. %d" % j)
        ax.legend()

    def draw_pairwise_correlation(self, ax):
        ax.set_xlabel(r"$q(\AA^{-1})$")
        ax.set_ylabel(r"$q(\AA^{-1})$")

        f, t = self.qv[[0, -1]]
        ax.imshow(self.R, extent=(f, t, t, f))

    def get_datcmp_rec(self):
        datcmp_key = "%d,%d" % self.j_pair_
        return self.datcmp_data.get(datcmp_key)

    def draw_pvalue_in_schilling(self, ax):
        from .SchillingDist import schilling_pdf
        ax.set_xlabel("Longest Run")
        ax.set_ylabel("Density")
        n = len(self.qv)
        x = np.linspace(2, 30, 100)
        y = schilling_pdf(n, 0.5, x)
        ax.plot(x, y)
        length, p_val, p_adj = self.get_datcmp_rec()
        x_ge = x >= length
        x_ = x[x_ge]
        y_ = y[x_ge]
        ax.fill_between(x_, y_, color="red", alpha=0.3)
        ymin, ymax = ax.get_ylim()
        def draw_text(tx, tyw, text):
            ty = ymin * (1 - tyw) + ymax * tyw
            ax.text(tx, ty, text, fontsize=16)

        draw_text(15, 0.8, "Longest Run = %.3g" % length)
        draw_text(15, 0.7, "P-value     = %.3g" % p_val)
        draw_text(15, 0.6, "P-adjusted  = %.3g" % p_adj)

    def on_figure_click(self, event):
        if event.button != 1:
            return

        ax2 = self.axes[1]
        if event.inaxes != ax2:
            return

        self.draw_pointed_data(event)
        self.mpl_canvas.draw()

    def draw_pointed_data(self, event):
        x = event.xdata
        y = event.ydata
        min_j = 0
        max_j = len(self.qv) - 1
        def get_index(x):
            j = bisect_right(self.qv, x)
            return max(min_j, min(max_j, j))

        i, j = tuple(sorted([get_index(v) for v in [x, y]]))
        xyz_list = []
        for k in (i, j):
            x_ = self.qv[[k,k]]
            y_ = self.j_pair
            z_ = self.M[k,y_]
            xyz_list.append((x_, y_, z_))
        if self.pairwise_point is None:
            ax1, ax2 = self.axes[0:2]
            self.pairwise_point, = ax2.plot(x, y, 'o', color="cyan")
            lines = []
            for k, (x_, y_, z_), c in zip((i, j), xyz_list, ["red", "yellow"]):
                line, = ax1.plot(x_, y_, z_, "-o", color=c, label="points at %.3g" % self.qv[k])
                lines.append(line)
            self.ax1_legend = ax1.legend()
            self.pairwise_lines = lines
        else:
            self.pairwise_point.set_data(x, y)
            for line, (x_, y_, z_) in zip(self.pairwise_lines, xyz_list):
                line.set_xdata(x_)
                line.set_ydata(y_)
                line.set_3d_properties(z_)
            for text, q in zip(self.ax1_legend.get_texts()[2:4], self.qv[[i,j]]):
                text.set_text("points at %.3g" % q)

        self.mpl_canvas.draw()
