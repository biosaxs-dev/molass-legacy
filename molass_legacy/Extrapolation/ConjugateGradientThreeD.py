# coding: utf-8
"""
    ConjugateGradientThreeD.py

    Copyright (c) 2019-2020, SAXS Team, KEK-PF
"""
import numpy as np
try:
    # for numba 1.49 or later
    from numba.core.decorators import jit
except:
    from numba.decorators import jit
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpl_patches
import matplotlib.animation as animation
from molass_legacy.KekLib.OurTkinter import Tk, Dialog

TITLE_FONTSIZE  = 16
USE_3D_WORKAROUND   = False

def  dot_plot_wa(ax, x, y, z, marker=None, color=None, markersize=None):
    ax.plot([x[0], x[0]], [y[0], y[0]], [z[0], z[0]], marker=marker, color=color, markersize=markersize)

@jit(nopython=True)
def compute_norm_1d(x, y, elution_z, mx, my):
    assert x.shape == y.shape

    z = []
    for i in range(x.shape[0]):
        x_ = x[i]
        y_ = y[i]
        z_ = np.linalg.norm( elution_z * x_ - mx )**2 + np.linalg.norm( elution_z * y_ - my )**2 
        z.append(z_)

        """
        if False and i == x.shape[0]//2:
            fig = dplt.figure()
            ax = fig.gca()
            ax.plot(elution_z * x_, label='elution_z * x_')
            ax.plot(mx, label='mx')
            ax.plot(elution_z * y_, label='elution_z * y_')
            ax.plot(my, label='my')
            ax.legend()
            fig.tight_layout()
            dplt.show()
        """

    return np.array(z)

@jit(nopython=True)
def compute_norm_2d(aa, bb, elution_z, ma, mb):
    assert aa.shape == bb.shape

    ret_array = np.zeros(aa.shape)
    # row_array = []
    for i in range(aa.shape[0]):
        # row = []
        for j in range(aa.shape[1]):
            a = aa[i,j]
            b = bb[i,j]
            c = np.linalg.norm( elution_z * a - ma )**2 + np.linalg.norm( elution_z * b - mb )**2 
            """
            if False and i == aa.shape[0]//2 and j == aa.shape[1]//2:
                fig = dplt.figure()
                ax = fig.gca()
                ax.plot(elution_z * a, label='elution_z * a')
                ax.plot(ma, label='ma')
                ax.plot(elution_z * b, label='elution_z * b')
                ax.plot(mb, label='mb')
                ax.legend()
                fig.tight_layout()
                dplt.show()
            """
            # row.append(c)
            ret_array[i,j] = c
        # row_array.append(row)

    # return np.array(row_array)
    return ret_array

dp = 0.03

class GdTreedPlot:
    def __init__(self, dialog, draw_trace):
        self.q  = dialog.q
        self.qf = dialog.qf
        self.qt = dialog.qt
        self.p1 = dialog.p1
        self.p2 = dialog.p2
        self.eval_range = dialog.eval_range
        self.data = dialog.data
        self.elution_z = dialog.elution_z
        self.p2m_scale = dialog.p2m_scale
        self.draw_trace = draw_trace

        N = 21
        p = np.linspace(self.p1 - dp, self.p1 + dp, N)
        q = np.linspace(self.p2 - dp, self.p2 + dp, N)
        pp, qq = np.meshgrid(p, q)
        f, _, t = self.eval_range
        f_data = self.data[self.qf, f:t]
        t_data = self.data[self.qt, f:t]
        rr = compute_norm_2d(pp, qq, self.elution_z, f_data, t_data)
        self.surface_xyz = (pp, qq, rr)

        if draw_trace:
            points = np.array(dialog.solver.demo_data)
            print('points.shape=', points.shape)
            x = points[:,0]*self.p2m_scale
            y = points[:,1]*self.p2m_scale
            z = compute_norm_1d(x, y, self.elution_z, f_data, t_data)
            self.trace_xyz = (x, y, z)

    def draw(self, ax, title_fontsize, animate_fig=None, title_add=''):
        qf_ = self.q[self.qf]
        qt_ = self.q[self.qt]

        ax.set_title("Objective function restricted to A(%.2g) and A(%.2g)%s" % (qf_, qt_, title_add), fontsize=title_fontsize)
        ax.set_xlabel('$A(%.2g)$' % qf_)
        ax.set_ylabel('$A(%.2g)$' % qt_)
        ax.set_zlabel('$norm^2$')

        ax.plot([self.p1 - dp, self.p1 + dp], [self.p2 - dp, self.p2 - dp], [0, 0], ':', color='red', alpha=0.3, linewidth=5)
        if USE_3D_WORKAROUND:
            dot_plot_wa(ax, [self.p1], [self.p2 - dp], [0], 'o', color='red', markersize=10)
        else:
            ax.plot([self.p1], [self.p2 - dp], [0], 'o', color='red', markersize=10)

        ax.plot([self.p1 + dp, self.p1 + dp], [self.p2 - dp, self.p2 + dp], [0, 0], ':', color='green', alpha=0.3, linewidth=5)
        if USE_3D_WORKAROUND:
            dot_plot_wa(ax, [self.p1 + dp], [self.p2], [0], 'o', color='green', markersize=10)
        else:
            ax.plot([self.p1 + dp], [self.p2], [0], 'o', color='green', markersize=10)

        ax.plot_surface(*self.surface_xyz, color='pink', alpha=0.3)

        if self.draw_trace:
            if animate_fig is None:
                ax.plot(*self.trace_xyz, marker='o', color='red', markersize=3)
            else:
                self.animate_trace_3d(animate_fig, ax)

    def animate_trace_3d(self, fig, ax, add_inset=True):
        x, y, z = self.trace_xyz
        ax.plot(x, y, z, marker='o', color='yellow', markersize=3)
        line, = ax.plot([x[0], x[0]],[y[0], y[0]], [z[0], z[0]], marker='o', color='red', markersize=3)

        data = np.array([x, y, z])

        def update_line(n, data, line):
            line.set_data(data[0:2, :n])
            line.set_3d_properties(data[2, :n])
            return (line,)

        self.anim3d = animation.FuncAnimation(fig, update_line, len(x), fargs=(data, line),
                    interval=200, blit=True)

        if add_inset:
            self.animate_trace_2d(fig, ax)

    def animate_trace_2d(self, fig, ax3d):
        ax = ax3d.inset_axes([0.04, 0.65, 0.25, 0.25])
        ax.set_xticklabels('')
        ax.set_yticklabels('')

        x, y, z = self.trace_xyz

        last_x = x[-1]
        last_y = y[-1]
        w, h = 0.0002, 0.0002
        ax.set_xlim(last_x - w, last_x + w)
        ax.set_ylim(last_y - h, last_y + h)

        ax.plot(x, y, marker='o', color='yellow', markersize=3)
        line, = ax.plot([x[0], x[0]], [y[0], y[0]], marker='o', color='red', markersize=5)

        data = np.array([x, y])

        def update_line(n, data, line):
            line.set_data(data[:, :n])
            return (line,)

        self.anim2d = animation.FuncAnimation(fig, update_line, len(x), fargs=(data, line),
                    interval=200, blit=True)

class GdAnimationDialog(Dialog):
    def __init__(self, parent, gd3d):
        self.parent = parent
        self.gd3d   = gd3d

        Dialog.__init__( self, parent, "Gradient Descent Animation", visible=False )

    def __del__(self):
        """
        TODO:
            seems to be a bug in tkinter.__init__.py
            fix tkinter instead of fixing app

        Exception in Tkinter callback
        Traceback (most recent call last):
          File "C:\Program Files\Python37\lib\tkinter\__init__.py", line 1705, in __call__
            return self.func(*args)
          File "C:\Program Files\Python37\lib\tkinter\__init__.py", line 752, in callit
            self.deletecommand(name)
          File "C:\Program Files\Python37\lib\tkinter\__init__.py", line 601, in deletecommand
            self._tclCommands.remove(name)
        AttributeError: 'NoneType' object has no attribute 'remove'
        """
        pass

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()

        self.fig = fig = plt.figure(figsize=(12, 10))

        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.mpl_canvas.draw()
        ax = fig.add_subplot(111, projection='3d')
        self.gd3d.draw(ax, TITLE_FONTSIZE, animate_fig=fig, title_add=' with a zoomed 2D inset')

        ax.disable_mouse_rotation()     # to avoid a rotation bug

        fig.tight_layout()

    def buttonbox( self ):
        box = Tk.Frame(self)
        box.pack()
        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.bind("<Return>", self.ok)
