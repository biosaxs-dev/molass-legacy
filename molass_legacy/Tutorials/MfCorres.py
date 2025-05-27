# coding: utf-8
"""
    MfCorres.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
import molass_legacy.KekLib.DebugPlot as dplt
from molass_legacy.KekLib.OurMatplotlib import get_color
from ModeledData import ModeledData
from MatrixData import simple_plot_3d
from DraggablePoints import DraggablePoints
from .SvdPrecision import plot_text

def demo_func_impl(fig):
    q = np.linspace(0.01, 0.6, 600)
    rg_list = [50, 24]
    d_list = [1, 3]
    md = ModeledData(q, 300, rg_list=rg_list, d_list=d_list)
    if False:
        dplt.push()
        md.plot_components()
        dplt.pop()
    M = md.get_data()

    rank = len(rg_list)

class MfCorresDemo(Dialog):
    def __init__(self, parent):
        self.parent = parent
        self.currently_dragging = False
        self.current_n = None
        self.polygon = None
        Dialog.__init__(self, parent, "MF Tutorial Demo", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        fig = plt.figure(figsize=(20,10))
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas.mpl_connect('button_press_event', self.on_press)
        self.mpl_canvas.mpl_connect('button_release_event', self.on_release)
        self.mpl_canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.q, self.U, self.s, self.VT, self.axes, points, self.objects = demo_func_impl(fig)
        self.sqrt_s1s2 = np.sqrt(self.s[0]*self.s[1])
        self.ax6 = ax6 = self.axes[1][2]
        plot_text(ax6, "$X$", fontsize=FONTSIZE)
        xmin, xmax = ax6.get_xlim()
        ymin, ymax = ax6.get_ylim()
        tx = (xmin + xmax)/2
        ty = ymin*0.8 + ymax*0.2
        ax6.text(tx, ty, "Left  Click: any P, C", alpha=0.1, ha='center', fontsize=20)
        ty = ymin*0.9 + ymax*0.1
        ax6.text(tx, ty, "Right Click: povitive P, C only", alpha=0.1, ha='center', fontsize=20)

        self.points = points
        self.draw_points()
        self.mpl_canvas.draw()

    def draw_points(self):
        points = self.points
        points_ = np.array(points)
        #  print(points)
        # print(points_)
        for point, pobj in zip(points, self.objects):
            p, = pobj
            p.set_xdata([point[0]])
            p.set_ydata([point[1]])

        self.px = points_[:,0]
        self.py = points_[:,1]
        vx = np.sum(self.px)
        vy = np.sum(self.py)
        poly_points = [(0, 0), points[0], (vx,vy), points[1]]
        if self.polygon is None:
            self.polygon = Polygon(poly_points, alpha=0.1)
        else:
            self.polygon.set_xy(poly_points)

        self.ax6.add_patch(self.polygon)

    def on_press(self, event):
        if event.dblclick or True:
            self.draw_next_rand_point(event.button)
            return

        if event.inaxes != self.ax6:
            return

        print('on_press')
        self.currently_dragging = True
        dist = (self.px - event.xdata)**2 + (self.py - event.ydata)**2
        n = np.argmin(dist)
        print('n=', n)
        self.current_n = n

    def on_release(self, event):
        if not self.currently_dragging:
            return

        print('on_release')
        self.update_point(event)
        self.currently_dragging = False
        self.current_n = None
        self.redraw_decomp()
        self.mpl_canvas.draw()

    def on_motion(self, event):
        if not self.currently_dragging:
            return

        print('on_motion')
        n = self.current_n
        p, = self.objects[n]
        p.set_xdata([event.xdata])
        p.set_ydata([event.ydata])
        self.mpl_canvas.draw()

    def update_point(self, event):
        n = self.current_n
        self.points[n] = (event.xdata, event.ydata)
        self.draw_points()

    def redraw_decomp(self):
        ax8, ax5, _, ax7 = self.axes[1]
        for ax in [ax5, ax7, ax8]:
            ax.cla()
        draw_PQ([ax5, ax7, ax8], self.q, self.points, self.U, self.s, self.VT)

    def draw_next_rand_point(self, button):
        print('draw_next_rand_point')
        n = 0
        while True:
            # print([n])
            points = generate_random_points(self.s)
            # points = generate_random_points()
            if button == 1 or self.positive_PC(points):
                break
            n += 1
            if n > 1000000:
                break
        print([n])
        self.points = points
        self.draw_points()
        self.redraw_decomp()
        self.mpl_canvas.draw()

    def positive_PC(self, points):
        rank = len(points)
        a, b = points[0]
        c, d = points[1]
        s1, s2 = self.s[0:2]
        e, f, g, h = solve_efgh(a, b, c, d, s1, s2)
        X = np.array([[a, c],[b, d]])
        YT = np.array([[e, f],[g, h]])
        # print(np.dot(X, YT))
        # print(s1, s2)
        P = np.dot(self.U[:, 0:rank], X)
        C = np.dot(YT, self.VT[0:rank,:])
        return np.all(P >= 0) and np.all(C >= 0)
