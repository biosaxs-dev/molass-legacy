# coding: utf-8
"""
    MatrixFactorization.py

    Copyright (c) 2019, SAXS Team, KEK-PF
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

FONTSIZE = 200

def solve_efgh_equation():
    """
        Ｐ・Ｃ ＝ Ｍ ＝ Ｕ・Ｓ・Ｖt ＝ (Ｕ・Ｘ)・(Ｙt・Ｖt)

         Ｘ  ・ Ｙt  ＝   Ｓ

        [a c]  [e f] = [s1  0]
        [b d]  [g h]   [0  s2]

        (1) a*e + c*g = s1
        (2) a*f + c*h =  0
        (3) b*e + d*g =  0
        (4) b*f + d*h = s2
        (5) (a*d - b*c)**2 = s1*s2
    """
    from sympy import var, solve
    a, b, c, d, e, f, g, h, s1, s2 = var('a b c d e f g h s1 s2')
    E1 = a*e + c*g - s1
    E2 = a*f + c*h
    E3 = b*e + d*g
    E4 = b*f + d*h - s2
    sols = solve([E1, E2, E3, E4], [e, f, g, h])
    print(sols)
    """
       {e: d*s1/(a*d - b*c), f: -c*s2/(a*d - b*c), g: -b*s1/(a*d - b*c), h: a*s2/(a*d - b*c)}

        e:  d*s1/(a*d - b*c)
        f: -c*s2/(a*d - b*c)
        g: -b*s1/(a*d - b*c)
        h:  a*s2/(a*d - b*c)
    """
    E5 = (a*d - b*c) - s1*s2
    sols = solve([E5], [d])
    print(sols)
    """
        {d: (b*c + s1*s2)/a}
    """

def solve_efgh(a, b, c, d, s1, s2):
    det = a*d - b*c
    e = d*s1/det
    f = -c*s2/det
    g = -b*s1/det
    h = a*s2/det
    return e, f, g, h

def demo_func():
    fig = dplt.figure(figsize=(20,10))
    demo_func_impl(fig)
    dplt.show()

def generate_random_points(s=None):
    a = np.random.rand()*2 - 1
    b = np.random.rand()*2 - 1
    c = np.random.rand()*2 - 1
    r = np.random.rand()*2 - 1
    if s is None:
        d = r
    else:
        scale = np.sqrt(s[0]*s[1])
        a *= scale
        b *= scale
        c *= scale
        # d = (b*c + s[0]*s[1])/a
        d = r*scale
    points = [(a,b), (c,d)]
    return points

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

    U, s, VT = np.linalg.svd(M)

    Us_ = np.dot( U[:,0:rank], np.diag( s[0:rank] ) )
    M_  = np.dot( Us_, VT[0:rank,:] )

    gs = GridSpec(2,4)
    axes = []
    for i in range(2):
        arow = []
        for k in range(4):
            proj = '3d' if k == 0 else None
            arow.append(fig.add_subplot(gs[i,k], projection=proj))
        axes.append(arow)
    ax4, ax1, ax2, ax3  = axes[0]

    for i in range(rank):
        ax1.plot(q, U[:,i])
        ax3.plot(VT[i,:])

    simple_plot_3d(ax4, M_, x=q)
    ns = 5
    ax2.plot(s[0:ns], ':', color='gray')
    for k in range(ns):
        ax2.plot(k, s[k], 'o', color=get_color(k))

    plot_text(ax1, "$U$", fontsize=FONTSIZE)
    plot_text(ax2, "$\Sigma$", fontsize=FONTSIZE)
    plot_text(ax3, "$V^T$", fontsize=FONTSIZE)

    ax8, ax5, ax6, ax7  = axes[1]
    points = generate_random_points(s)
    # points = generate_random_points(s)
    draw_PQ([ax5, ax7, ax8], q, points, U, s, VT)

    # r = 2
    # r = np.sqrt(s[0]*s[1])
    r = s[0]
    ax6.set_xlim(-r, r)
    ax6.set_ylim(-r, r)
    colors = ['red', 'yellow']

    objects = []
    for p, c_ in zip(points, colors):
        objects.append(ax6.plot(*p, 'o', color=c_))

    fig.tight_layout()
    return q, U, s, VT, axes, points, objects

def draw_PQ(axes, q, points, U, s, VT):
    ax5, ax7, ax8 = axes
    s1 = s[0]
    s2 = s[1]
    rank = len(points)
    a, b = points[0]
    c, d = points[1]
    e, f, g, h = solve_efgh(a, b, c, d, s1, s2)
    X = np.array([[a, c],[b, d]])
    YT = np.array([[e, f],[g, h]])
    print(np.dot(X, YT))
    print(s1, s2)
    P = np.dot(U[:, 0:rank], X)
    C = np.dot(YT, VT[0:rank,:])
    Q = np.dot(P, C)

    for i in range(rank):
        ax5.plot(q, P[:,i])
        ax7.plot(C[i,:])
    simple_plot_3d(ax8, Q, x=q)
    plot_text(ax5, "$P$", fontsize=FONTSIZE)
    plot_text(ax7, "$C$", fontsize=FONTSIZE)

class MatrixFactorization(Dialog):
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
