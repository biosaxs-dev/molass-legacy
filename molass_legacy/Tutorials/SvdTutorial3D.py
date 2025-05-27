# coding: utf-8
"""
    SvdTutorial3D.py

    Copyright (c) 2019, SAXS Team, KEK-PF
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
from OurMatplotlib3D import plot_parallelepiped, plot_sphere
from ModeledData import ModeledData
from MatrixData import simple_plot_3d
from .SvdPrecision import plot_text

FONTSIZE = 200

class SvdTutorial3D(Dialog):
    def __init__(self, parent):
        self.parent = parent
        self.B = np.array([[1,0,0], [0,1,0], [0,0,1]]).T
        Dialog.__init__(self, parent, "SVD Tutorial (3D)", visible=False)

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
        ax1 = fig.add_subplot(gs[1:3, 0], projection='3d')
        ax2 = fig.add_subplot(gs[0:2, 1], projection='3d')
        ax3 = fig.add_subplot(gs[1:3, 2], projection='3d')
        ax4 = fig.add_subplot(gs[2:4, 1], projection='3d')
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
        e = np.random.rand()*2 - 1
        f = np.random.rand()*2 - 1
        g = np.random.rand()*2 - 1
        h = np.random.rand()*2 - 1
        i = np.random.rand()*2 - 1
        j = np.random.rand()*2 - 1

        self.points = [(a,b,c), (d,e,f), (h,i,j)]
        self.M = np.array(self.points).T
        U, s, VT = np.linalg.svd(self.M)
        self.U = U
        self.S = np.diag(s)
        self.V = VT.T

    def draw_points(self, ax, W):
        ax.plot([0], [0], [0], 'o', color='red')
        px = W[0,:]
        py = W[1,:]
        pz = W[2,:]
        ax.plot(px, py, pz, 'o')

    def draw_parallelepiped(self, ax, W, color=None, edgecolor='w'):
        p1 = W[:,0]
        p2 = W[:,1]
        p3 = W[:,2]
        plot_parallelepiped(ax, (0,0,0), [p1, p2, p3], color=color, edgecolor=edgecolor, alpha=0.5)

    def draw_sphere(self, ax):
        plot_sphere(ax, (0,0,0), 1, alpha=0.05, n=40, color='gray', edgecolor='gray')

    def draw1(self, ax):
        self.draw_points(ax, self.M)
        self.draw_parallelepiped(ax, self.B, color='pink', edgecolor='red')
        self.draw_parallelepiped(ax, self.M, color='cyan', edgecolor='blue')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

        # plot_text(ax, "$M$", fontsize=FONTSIZE)

    def draw2(self, ax):
        self.draw_sphere(ax)
        self.draw_points(ax, self.V)
        self.draw_parallelepiped(ax, self.B, color='pink', edgecolor='red')
        self.draw_parallelepiped(ax, self.V, color='cyan', edgecolor='blue')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

        # plot_text(ax, "$V^t$", fontsize=FONTSIZE)

    def draw3(self, ax):
        self.draw_points(ax, self.S)
        self.draw_parallelepiped(ax, self.B, color='pink', edgecolor='red')
        self.draw_parallelepiped(ax, self.S, color='cyan', edgecolor='blue')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

        # plot_text(ax, "$\Sigma$", fontsize=FONTSIZE)

    def draw4(self, ax):
        self.draw_sphere(ax)
        self.draw_points(ax, self.U)
        self.draw_parallelepiped(ax, self.B, color='pink', edgecolor='red')
        self.draw_parallelepiped(ax, self.U, color='cyan', edgecolor='blue')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

        # plot_text(ax, "$U$", fontsize=FONTSIZE)

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
