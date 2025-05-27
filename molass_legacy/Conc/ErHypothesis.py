# coding: utf-8
"""
    ErHypothesis.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
from bisect import bisect_right
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkUtils import split_geometry
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from ModeledData import ModeledData
from MatrixData import simple_plot_3d
from molass_legacy.Models.ElutionCurveModels import emg_x_from_height_ratio
from SvdDenoise import get_denoised_data

class ErHypothesisDialog(Dialog):
    def __init__( self, parent, seed=None, noise=0.01):
        self.init_seed = seed
        self.noise = noise
        Dialog.__init__( self, parent, "ErHypothesis", visible=False )

    def show(self):
        self._show()

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        fig = plt.figure(figsize=(21, 10))
        self.fig = fig
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        gs = GridSpec(2,3)
        ax00 = fig.add_subplot(gs[0,0], projection='3d')
        ax01 = fig.add_subplot(gs[0,1])
        ax02 = fig.add_subplot(gs[0,2])
        ax10 = fig.add_subplot(gs[1,0])
        ax11 = fig.add_subplot(gs[1,1])
        ax12 = fig.add_subplot(gs[1,2])
        self.axes = np.array([[ax00, ax01, ax02], [ax10, ax11, ax12]])

        self.generate_data(modeled=False, seed=self.init_seed)

        fig.suptitle("Verification Simulation for Exact Rank Hypothesis; random seed=%d" % self.seed, fontsize=30)

        self.draw_3d(ax00)
        self.draw_sigmas(ax10)
        self.draw_input_curves(ax01, ax11)
        self.draw_p_curves(ax02, ax12)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9, bottom=0.1)
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

        self.almerge = Tk.IntVar()
        self.almerge.set(0)
        w = Tk.Checkbutton(box, text="ALMERGE", variable=self.almerge)
        w.pack(side=Tk.LEFT, padx=10, pady=5)

        w = Tk.Button(box, text="Solve", width=10, command=self.solve_several_ways, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=10, pady=5)

        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=10, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def generate_data(self, modeled=True, seed=None, noise=None):
        if seed is None:
            seed = np.random.randint(1000, 9999)

        if noise is None:
            noise = self.noise

        if modeled:
            num_elutions = 300
            self.qvector = qvector = np.linspace(0.007, 0.5, 600)
            self.md = md = ModeledData(qvector, num_elutions)
            self.M = M = md.get_data(noise=noise)
            h, mu, sigma, tau = self.md.e_params[0]
            self.conc = 0
        else:
            from .SimData import SimData
            in_folder = 'D:/PyTools/reports/analysis-erh/preview_results'
            a_file = in_folder + '/A1.dat'
            b_file = in_folder + '/B1.dat'
            simd = SimData(a_file, b_file)
            self.qvector = qvector = simd.q
            conc = 1
            M, E = simd.get_data(noise=noise, conc=conc, seed=seed)
            self.M = M
            self.E = E
            h, mu, sigma, tau = simd.e_params[0]
            self.conc = conc
            self.truth = simd.a

        self.seed = seed
        self.i = i = bisect_right(qvector, 0.02)
        self.ez = z =  M[i,:]
        c = z/np.max(z)
        self.C = np.array([c, c**2])

        f, t = [int(p+0.5) for p in emg_x_from_height_ratio(0.5, mu, sigma, tau)]
        print("(f, t)=", (f, t))
        self.eslice = eslice = slice(f,t+1)

        U, s, VT = np.linalg.svd(self.M[:,eslice])
        self.sigmas = s
        rank = 2
        M_ = U[:,0:rank] @ np.diag(s[0:rank]) @ VT[0:rank,:]
        C_ = self.C[:,eslice]
        self.P = M_ @ np.linalg.pinv(C_)

    def draw_3d(self, ax):
        qvector = self.qvector
        M = self.M
        num_elutions = M.shape[1]

        simple_plot_3d(ax, M, x=qvector)

        x = np.ones(num_elutions)*qvector[self.i]
        y = np.arange(num_elutions)
        z = self.ez
        ax.plot(x, y, z, color='orange')
        self.elution = z

        self.j = j = np.argmax(z)
        x = qvector
        y = np.ones(len(x))*j
        z = M[:,j]
        ax.plot(x, y, z, color='green')
        self.ridge = z

    def draw_sigmas(self, ax):
        s = self.sigmas
        n = min(5, len(s))
        ax.plot(np.arange(n), s[0:n], ':', marker='o')
        tx = np.average(ax.get_xlim())
        ty = np.average(ax.get_ylim())
        ax.text(tx, ty, r'$\frac{\sigma_1}{\sigma_0}=%.3g$' % (s[1]/s[0]), alpha=0.5, fontsize=40, ha='center', va='center')

    def draw_input_curves(self, axe, axs):
        axe.plot(self.elution, color='orange')
        f = self.eslice.start
        t = self.eslice.stop-1
        ymin, ymax = axe.get_ylim()
        p = Rectangle(
                (f, ymin),      # (x,y)
                t - f,          # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        axe.add_patch(p)

        axs.set_yscale('log')
        axs.plot(self.qvector, self.ridge)

    def draw_p_curves(self, ax, axt):
        P = self.P
        a = P[:,0]
        b = P[:,1]

        ax.plot(self.qvector, a, color='C1', label='$A_2$')
        for ax_ in [ax, axt]:
            ax_.plot(self.qvector, b, color='pink', label='$B_2$')

    def solve_several_ways(self):
        import molass_legacy.KekLib.DebugPlot as dplt

        if self.conc < 0.5:
            rank_pairs = [(1, 1), (2, 1), (2, 2), (3, 2), (0, 1), (0, 2)]
        else:
            rank_pairs = [(2, 2), (2, 1), (3, 2), (4, 2), (0, 1), (0, 2)]

        scores = []
        for m, n in rank_pairs:
            score = self.evaluate_rank_mn_solution(m, n)
            scores.append(score)

        if self.almerge.get():
            score = self.evaluate_almerge_solution()
            scores.append(score)

        dplt.set_global_opts(kill_button=False)     # reset later if necessary
        dplt.push()
        fig, ax = dplt.subplots()

        ax.set_title("Evaluation of Various Rank (m,n) Solutions", fontsize=20)
        ax.set_ylabel("Norm of Deviation from true Curve", fontsize=16)
        ax.set_yscale('log')

        for k, score in enumerate(scores):
            label = 'Rank %s' % str(rank_pairs[k]) if k < len(rank_pairs) else 'ALMERGE'
            ax.plot(k, score, 'o', label=label)
            if k == 0:
                dx = 0.1
                dy = 2
                ax.annotate("Exact Rank", xy=(k, score), xytext=(k+dx, score*dy),
                                arrowprops=dict(arrowstyle='->', color='k'))

        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax)
        score = scores[0]
        ax.plot([xmin, xmax], [score, score], ':', color='red', alpha=0.5)

        ax.legend()
        fig.tight_layout()
        dplt.show()
        dplt.pop()

    def evaluate_rank_mn_solution(self, m, n):
        assert n in [1, 2]
        eslice = self.eslice
        M_ = get_denoised_data(self.M[:,eslice], rank=m)
        if n == 1:
            C_ = self.C[0,eslice][np.newaxis,:]
        else:
            C_ = self.C[:,eslice]
        P = M_ @ np.linalg.pinv(C_)
        solution = P[:,0]

        def obj_func(pv):
            return np.sum((solution*pv[0] - self.truth)**2)

        res = minimize(obj_func, (1,))
        return res.fun

    def evaluate_almerge_solution(self):
        from ReAtsas.Almerge import Almerge
        from molass_legacy._MOLASS.SerialSettings import get_setting

        eslice = self.eslice
        c = self.C[0,:]
        datafiles = self.save_data()
        almerge = Almerge()
        indeces = np.arange(eslice.start, eslice.stop)
        out_file = get_setting('temp_folder') + '/extrapolated.dat'
        ret = almerge.execute(c, datafiles, indeces, out_file)
        solution = ret.exz_array[:,1]

        def obj_func(pv):
            return np.sum((solution*pv[0] - self.truth)**2)

        res = minimize(obj_func, (1,))
        return res.fun

    def save_data(self, out_folder=None):
        from DataUtils import get_pytools_folder
        from molass_legacy.KekLib.BasicUtils import clear_dirs_with_retry
        from molass_legacy.KekLib.NumpyUtils import np_savetxt

        if out_folder is None:
            out_folder = get_pytools_folder() + '/temp'

        clear_dirs_with_retry([out_folder])
        datafiles = []
        qv = self.qvector
        for j in range(self.M.shape[1]):
            file = 'simdata_%05d.dat' % j
            path = '/'.join([out_folder, file])
            np_savetxt(path, np.array([qv, self.M[:,j], self.E[:,j]]).T)
            datafiles.append(path)
        return datafiles
