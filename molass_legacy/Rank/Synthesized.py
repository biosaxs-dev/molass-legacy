# coding: utf-8
"""
    Synthesized.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from SvdDenoise import get_denoised_data
# import molass_legacy.KekLib.DebugPlot as plt
from DataUtils import get_in_folder
# from .Boundary import get_boundary
from Theory.SolidSphere import get_boundary_params

def solve(M, E, c, qv, b, k, debug=False):
    M2 = get_denoised_data(M, rank=2)
    M1 = get_denoised_data(M, rank=1)

    C2 = np.array([c, c**2])
    P2 = M2 @ np.linalg.pinv(C2)
    C1 = np.array([c])
    P1 = M2 @ np.linalg.pinv(C1)

    w = 1/(1+np.exp(-k*(qv-b)))

    M_ = (M2.T*(1-w) + M1.T*w).T

    P_ = M_ @ np.linalg.pinv(C2)
    py_list = []
    py_list.append(P_[:,0].copy())

    for i in range(0):
        P_[b:,1] = 0
        C_ = np.linalg.pinv(P_) @ M_
        P_ = M_ @ np.linalg.pinv(C_)
        py_list.append(P_[:,0].copy())

    return P1, P2, w, py_list

class Demo(Dialog):
    def __init__(self, parent, *args):
        self.args = args
        self.fig = None
        self.almerge_result = None
        Dialog.__init__(self, parent, "Synthesized LRF Demo", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        fig, axes = plt.subplots(ncols=3, figsize=(21,7))
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        self.fig = fig
        self.axes = axes
        self.ax2t = None
        self.show_bq = Tk.IntVar()
        self.show_bq.set(0)
        self.show_almerge = Tk.IntVar()
        self.show_almerge.set(0)

        self.draw()

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
        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=10, pady=5)

        panel = Tk.Frame(bottom_frame)
        panel.pack(side=Tk.RIGHT, padx=padx)

        w = Tk.Checkbutton(panel, text="show ALMERGE info", variable=self.show_almerge)
        w.pack(side=Tk.LEFT, padx=20)

        w = Tk.Checkbutton(panel, text="show B(q)", variable=self.show_bq)
        w.pack(side=Tk.LEFT, padx=20)

        w = Tk.Button(panel, text="Redraw", width=10, command=self.draw)
        w.pack(side=Tk.LEFT, padx=20)

    def draw(self):
        qv, P1, P2, b, w, py_list = self.args[3:]

        fig = self.fig
        for ax in self.axes:
            ax.cla()
        ax1, ax2, ax3 = self.axes
        if self.show_bq.get():
            if self.ax2t is None:
                ax2t = ax2.twinx()
                self.ax2t = ax2t
            else:
                ax2t = self.ax2t
            ax2t.cla()
            ax2t.grid(False)
        else:
            if self.ax2t is not None:
                self.ax2t.remove()
                self.ax2t = None

        fig.suptitle("Logistically Synthesized LRF Demo for " + get_in_folder(), fontsize=20)
        ax1.set_title("Rank(1,1) Weights (Logistic Function)", fontsize=16)
        ax2.set_title("Separate LRF", fontsize=16)
        ax3.set_title("Logistically Synthesized LRF", fontsize=16)

        ax1.set_ylabel('Weight')
        ax1.set_xlabel('Q')

        for ax in [ax2, ax3]:
            ax.set_yscale('log')
            ax.set_ylabel('$Log_{10}(I)$')
            ax.set_xlabel('Q')

        c0 = np.ones((len(w),3))*to_rgb('C0')
        c1 = np.ones((len(w),3))*to_rgb('C1')
        c_ = (c0.T*(1-w) + c1.T*w).T

        for i in range(len(w)):
            ax1.plot(qv[i:i+2], w[i:i+2], color=c_[i])

        for rank, y in [(2, P2[:,0]), (1, P1[:,0])]:
            ax2.plot(qv, y, label='rank({0},{0})'.format(rank), alpha=0.5)

        if self.show_bq.get():
            ax2t.plot(qv, P2[:,1], color='pink', label='B(q) in Rank(2,2)')

        y = py_list[-1]
        for i in range(len(w)):
            ax3.plot(qv[i:i+2], y[i:i+2], color=c_[i])

        ax3.set_ylim(ax2.get_ylim())

        bq = b
        for ax in [ax1, ax2, ax3]:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            ax.plot([bq, bq], [ymin, ymax], ':', color='red', label='rank boundary')

        if self.show_almerge.get():
            abq, ay = self.get_almerge_results()
            ax2.plot(qv, ay, label='almerge extrapolated', alpha=0.5)
            ymin, ymax = ax2.get_ylim()
            ax2.plot([abq, abq], [ymin, ymax], color='red', label='almerge boundary')

        ax1.legend()
        ax2.legend()
        if self.show_bq.get():
            ax2t.legend(bbox_to_anchor=(1, 0.85), loc='upper right')
        ax3.legend()
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        # plt.show()
        self.mpl_canvas.draw()

    def get_almerge_results(self):
        abq, ay = self.compute_almerge_results()
        return abq, ay

    def compute_almerge_results(self):
        from SerialAtsasTools import AlmergeExecutor

        if self.almerge_result is None:
            almerge = AlmergeExecutor()
            M, E, c, qv = self.args[0:4]
            result = almerge.execute_matrix(qv, M, E, c)
            self.almerge_result = result
        else:
            qv = self.args[3]
            result = self.almerge_result
        i = result.overlap_from_max
        return qv[i], result.exz_array[:,1]

def demo(parent, sd, debug=True):
    M, E, qv, ecurve = sd.get_xr_data_separate_ly()

    range_ = ecurve.get_ranges_by_ratio(0.5)[0]
    f = range_[0]
    p = range_[1]
    t = range_[2]+1
    eslice = slice(f,t)
    M_ = M[:,eslice]
    E_ = E[:,eslice]

    Rg = sd.pre_recog.get_rg()
    b1, b2, k = get_boundary_params(Rg)

    x = ecurve.x
    y = ecurve.y
    c = y[f:t].copy()
    c /= np.max(c)

    P1, P2, w, py_list = solve(M_, E_, c, qv, b1, k, debug=debug)

    if debug:
        dialog = Demo(parent, M_, E_, c, qv, P1, P2, b1, w, py_list)
        dialog.show()

def get_reduced_rank(rank, cd):
    if cd == 1:
        ret_rank = rank
    else:
        ret_rank = rank - 1 if rank < 4 else rank - 2
    return ret_rank

def synthesized_data(qv, M, Rg, rank=2, cd=1, boundary=None, k=None, logger=None):
    M2 = get_denoised_data(M, rank=rank)
    rank_ = get_reduced_rank(rank, cd)
    M1 = get_denoised_data(M, rank=rank_)
    if boundary is None:
        assert k is None
        boundary, b2, k = get_boundary_params(Rg)
    else:
        assert k is not None

    w = 1/(1+np.exp(-k*(qv-boundary)))
    M_ = (M2.T*(1-w) + M1.T*w).T

    return M_, boundary
