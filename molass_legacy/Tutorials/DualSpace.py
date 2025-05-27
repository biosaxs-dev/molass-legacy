# coding: utf-8
"""
    DualSpace.py

    Copyright (c) 2019,2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
import mpl_toolkits.mplot3d.art3d as art3d
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
import molass_legacy.KekLib.DebugPlot as dplt
from molass_legacy.KekLib.OurMatplotlib import get_color
from ModeledData import ModeledData
from MatrixData import simple_plot_3d
from OurManim import manim_init
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar

class DualSpace(Dialog):
    def __init__(self, parent):
        self.parent = parent
        self.currently_dragging = False
        self.current_n = None
        self.polygon = None
        manim_init()
        Dialog.__init__(self, parent, "Dual Space", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        bframe = Tk.Frame(body_frame)
        bframe.pack(fill=Tk.BOTH)
        tframe = Tk.Frame(bframe)
        tframe.pack(side=Tk.LEFT)

        fig = plt.figure(figsize=(21,7))
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        self.axes = ax1, ax2, ax3
        self.draw()
        fig.tight_layout()

        self.mpl_canvas.draw()

    def draw(self):
        q = np.linspace(0.01, 0.6, 600)
        rg_list = [50, 24]
        d_list = [1, 3]
        md = ModeledData(q, 300, rg_list=rg_list, d_list=d_list)

        ax1, ax2, ax3 = self.axes

        ax1.set_xlabel('Q')
        ax1.set_ylabel('Eno')
        ax1.set_zlabel('Intensity')

        M = md.get_data()
        simple_plot_3d(ax1, M, x=q)
        ax1.view_init(elev=20, azim=-20)

        i = [20, 130]
        j = [120, 170]

        n = M.shape[1]
        ey = np.arange(n)
        for i_ in i:
            ex = np.ones(n)*q[i_]
            ez = M[i_,:]
            # ax1.plot(ex, ey, ez, color='orange')
            poly_points = [(0,0)] + list(zip(ey,ez)) + [(n-1,0)]
            poly =  Polygon(poly_points, alpha=0.3, color='orange')
            ax1.add_patch(poly)
            art3d.pathpatch_2d_to_3d(poly, z=q[i_], zdir='x')

        n = M.shape[0]
        ex = q
        for j_ in j:
            ey = np.ones(n)*j_
            ez = M[:,j_]
            ax1.plot(ex, ey, ez, color='green')
            poly_points = [(0,0)] + list(zip(ex,ez)) + [(q[-1],0)]
            poly =  Polygon(poly_points, alpha=0.3, color='green')
            ax1.add_patch(poly)
            art3d.pathpatch_2d_to_3d(poly, z=j_, zdir='y')

        ax2.set_title('2D Scattering Curve Space', fontsize=20)
        ax2.set_xlabel('Intensity on Q=%.2g' % q[i[0]], color='blue', fontsize=16)
        ax2.set_ylabel('Intensity on Q=%.2g' % q[i[1]], color='red', fontsize=16)

        labels = ['a', 'b', 'c', 'd']
        x = [q[i[0]], q[i[1]], q[i[0]], q[i[1]]]
        y = [j[0], j[0], j[1], j[1]]
        z = [M[i[0],j[0]], M[i[1],j[0]], M[i[0],j[1]], M[i[1],j[1]]]
        k = 0
        for label, x_, y_, z_ in zip(labels, x, y, z):
            color = 'blue' if k%2==0 else 'red'
            ax1.plot([x_, x_], [y_, y_], [0, z_], color=color, linewidth=3)
            ax1.text(x_, y_, z_, label, fontsize=20, color=color)
            k += 1

        a, b, c, d = z
        s_points = [(0,0), (1,0), (1,1), (0,1)]
        p_points = [(0,0), (a,b), (a+c, b+d), (c,d)]
        for points, color in [(s_points, 'pink'), (p_points, 'cyan')]:
            poly =  Polygon(points, alpha=0.3, color=color)
            ax2.add_patch(poly)

        def plot_point_s(x, y, label):
            ax2.plot([0,x], [y,y], color='blue')
            ax2.plot([x,x], [0,y], color='red')
            ax2.plot(x, y, 'o', color='orange')
            ax2.text(x, y+0.04, label, fontsize=20)

        plot_point_s(a, b, r"$ \begin{bmatrix} a \\ b \end{bmatrix} $")
        plot_point_s(c, d, r"$ \begin{bmatrix} c \\ d \end{bmatrix} $")
        ax2.set_xlim(-0.4, 1.4)
        ax2.set_ylim(-0.4, 1.4)

        ax3.set_title('2D Elution Curve Space', fontsize=20)
        ax3.set_xlabel('Intensity on Eno=%d' % j[0], color='purple', fontsize=16)
        ax3.set_ylabel('Intensity on Eno=%d' % j[1], color='purple', fontsize=16)

        s_points = [(0,0), (1,0), (1,1), (0,1)]
        p_points = [(0,0), (a,c), (a+b, c+d), (b,d)]
        for points, color in [(s_points, 'pink'), (p_points, 'cyan')]:
            poly =  Polygon(points, alpha=0.3, color=color)
            ax3.add_patch(poly)

        def plot_point_e(x, y, label, color):
            ax3.plot([0,x], [y,y], color=color)
            ax3.plot([x,x], [0,y], color=color)
            ax3.plot(x, y, 'o', color='orange')
            ax3.text(x, y+0.04, label, fontsize=20)

        plot_point_e(a, c, r"$ \begin{bmatrix} a \\ c \end{bmatrix} $", "blue")
        plot_point_e(b, d, r"$ \begin{bmatrix} b \\ d \end{bmatrix} $", "red")

        ax3.set_xlim(-0.4, 1.4)
        ax3.set_ylim(-0.4, 1.4)
