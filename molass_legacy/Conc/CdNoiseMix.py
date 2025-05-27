# coding: utf-8
"""
    CdNoiseMix.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Button
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkUtils import split_geometry
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.Models.ElutionCurveModels import emg_x_from_height_ratio
from SvdDenoise import get_denoised_data
from .SimData import SimData

class CdNoiseMixDialog(Dialog):
    def __init__( self, parent, seed=None, show_rank=False, for_paper=False):
        in_folder = 'D:/PyTools/reports/analysis-erh/preview_results'
        a_file = in_folder + '/A1.dat'
        b_file = in_folder + '/B1.dat'
        self.simd = simd = SimData(a_file, b_file)
        self.q = simd.q
        h, mu, sigma, tau = simd.e_params[0]
        f, t = [int(p+0.5) for p in emg_x_from_height_ratio(0.5, mu, sigma, tau)]
        print("(f, t)=", (f, t))
        self.eslice = slice(f,t+1)
        if seed is None:
            seed = np.random.randint(1000, 9999)
        self.seed = seed
        self.show_rank = show_rank
        self.for_paper = for_paper
        np.random.seed(seed)
        Dialog.__init__( self, parent, "CdNoiseMixDialog", visible=False )

    def show(self):
        self._show()

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        fig = plt.figure(figsize=(23, 12))
        self.fig = fig
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        fig.suptitle("Apparent B(q)'s at various levels of Noises and SCD's; random seed=%d" % self.seed, fontsize=30)

        noise_levels = [0.01, 0.02, 0.05]
        cd_levels = [0.05, 0.1, 1, 2, 5]

        nrows = len(noise_levels)
        ncols = len(cd_levels)
        gs = GridSpec(nrows, ncols)
        axes = []
        for i, noise in enumerate(noise_levels):
            axes_row = []
            for j, cd in enumerate(cd_levels):
                ax = fig.add_subplot(gs[i,j])
                ax.set_title("Noise=%.g, SCD=%.g" % (noise, cd), fontsize=16)
                self.draw_aqbq(ax, noise, cd)
                axes_row.append(ax)
            axes.append(axes_row)
        axes = np.array(axes)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9, bottom=0.05)
        self.mpl_canvas.draw()

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

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def draw_aqbq(self, ax, noise, cd):
        sd = self.simd
        i = sd.i
        M, _ = sd.get_data(noise, cd)
        eslice = self.eslice
        c = M[i,eslice]
        c = c/np.max(c)
        C = np.array([c, c**2])
        M_ = get_denoised_data(M[:,eslice], rank=2)
        P_ = M_ @ np.linalg.pinv(C)
        a = P_[:,0]
        b = P_[:,1]

        if self.for_paper:
            aq_color = 'red'
            bq_color = 'blue'
        else:
            aq_color = 'C1'
            bq_color = 'cyan'

        ax.plot(self.q, a, color=aq_color)
        ax.plot(self.q, b, color=bq_color, alpha=0.5)

        if not self.for_paper:
            axt = ax.twinx()
            axt.grid(False)
            axt.plot(self.q, b, color='pink', alpha=0.5)

        if self.show_rank:
            self.draw_exact_rank(ax, self.simd.a, a, M, c)

    def draw_exact_rank(self, ax, truth, a, M, c):
        solution = a
        def obj_func(pv):
            return np.sum((solution*pv[0] - truth)**2)

        res = minimize(obj_func, (1,))
        rank2_score = res.fun

        eslice = self.eslice
        M_ = get_denoised_data(M[:,eslice], rank=1)
        C = np.array([c])
        P_ = M_ @ np.linalg.pinv(C)
        solution = P_[:,0]
        res = minimize(obj_func, (1,))
        rank1_score = res.fun

        print('scores=(%.3g, %.3g)' % (rank1_score, rank2_score))
        rank = 1 if rank1_score < rank2_score else 2

        tx = np.average(ax.get_xlim())
        ty = np.average(ax.get_ylim())

        ax.text(tx, ty, str(rank), fontsize=200, alpha=0.1, color='blue', ha='center', va='center')
