# coding: utf-8
"""
    SvdTutorial2D.py

    Copyright (c) 2019,2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon, Circle
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
import molass_legacy.KekLib.DebugPlot as dplt
from molass_legacy.KekLib.OurMatplotlib import get_color
from ModeledData import ModeledData
from MatrixData import simple_plot_3d
from .SvdPrecision import plot_text

FONTSIZE = 200

class SvdTutorial2D(Dialog):
    def __init__(self, parent):
        self.parent = parent
        Dialog.__init__(self, parent, "SVD Tutorial", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        self.fig = fig = plt.figure(figsize=(15,10))
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas.mpl_connect('button_press_event', self.on_press)
        self.create_axes()
        self.draw()
        fig.tight_layout()
        self.mpl_canvas.draw()

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Animate", width=10, command=self.animate, state=Tk.DISABLED)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

    def create_axes(self):
        fig = self.fig
        gs = GridSpec(4,3)
        ax1 = fig.add_subplot(gs[1:3, 0])
        ax2 = fig.add_subplot(gs[0:2, 1])
        ax3 = fig.add_subplot(gs[1:3, 2])
        ax4 = fig.add_subplot(gs[2:4, 1])
        self.axes = [ax1, ax2, ax3, ax4]
        self.ax5 = fig.add_subplot(gs[3, 2])
        self.ax5.set_axis_off()
        plot_text(self.ax5, "Click anywhere to play", alpha=0.2, fontsize=30)

    def draw(self):
        self.generate_random_points()
        ax1, ax2, ax3, ax4 = self.axes
        self.draw1(ax1)
        self.draw2(ax2)
        self.draw3(ax3)
        self.draw4(ax4)

    def on_press(self, event):
        for ax in self.axes:
            ax.cla()
        self.draw()
        self.mpl_canvas.draw()

    def generate_random_points(self):
        a = np.random.rand()*2 - 1
        b = np.random.rand()*2 - 1
        c = np.random.rand()*2 - 1
        d = np.random.rand()*2 - 1
        self.points = [(a,b), (c,d)]
        self.M = np.array(self.points).T
        U, s, VT = np.linalg.svd(self.M)
        self.U = U
        self.S = np.diag(s)
        self.V = VT.T

    def draw_points(self, ax, W):
        px = W[0,:]
        py = W[1, :]
        vx = np.sum(px)
        vy = np.sum(py)
        p = W[:,0]
        q = W[:,1]
        ax.plot(*p, 'o')
        ax.plot(*q, 'o')
        poly_points = [(0,0), p, (vx,vy), q]
        polygon = Polygon(poly_points, alpha=0.5, color='cyan')
        ax.add_patch(polygon)
        poly_points = [(0,0), (1,0), (1,1), (0,1)]
        polygon = Polygon(poly_points, alpha=0.1, color='red')
        ax.add_patch(polygon)

    def draw_disc(self, ax):
        circle = Circle((0,0), 1, alpha=0.1)
        ax.add_patch(circle)

    def draw1(self, ax):
        self.draw_points(ax, self.M)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        plot_text(ax, "$M$", fontsize=FONTSIZE)

    def draw2(self, ax):
        self.draw_points(ax, self.U)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        self.draw_disc(ax)
        plot_text(ax, "$V^t$", fontsize=FONTSIZE)

    def draw3(self, ax):
        self.draw_points(ax, self.S)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        plot_text(ax, "$\Sigma$", fontsize=FONTSIZE)

    def draw4(self, ax):
        self.draw_points(ax, self.V)
        self.draw_disc(ax)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        plot_text(ax, "$U$", fontsize=FONTSIZE)

    def animate(self):
        self.show_matrix_mapping()
        self.make_anim_VT()
        self.make_anim_S()
        self.make_anim_U()
        self.show_anim()

    def show_matrix_mapping(self):
        dplt.push()
        fig, axes = dplt.subplots(nrows=1, ncols=2, figsize=(12,6))
        ax1, ax2 = axes
        ax1.plot(0,0,'o')
        ax2.set_axis_off()
        fig.tight_layout()
        dplt.show()
        dplt.pop()

    def make_anim_VT(self):
        pass

    def make_anim_S(self):
        pass

    def make_anim_U(self):
        pass

    def show_anim(self):
        pass
