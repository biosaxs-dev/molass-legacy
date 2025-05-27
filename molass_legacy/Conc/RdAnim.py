# coding: utf-8
"""
    RdAnim.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.SciPyCookbook import smooth
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from tkinter import ttk
from SvdDenoise import get_denoised_data
from DataUtils import get_in_folder
from SerialAtsasTools import AlmergeExecutor
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.KekLib.BasicUtils import clear_dirs_with_retry

WIDTH = 10

class RdAnim(Dialog):
    def __init__( self, parent, sd, rank=2, sythesized=False):
        self.sd = sd
        self.j0 = self.sd.xr_j0
        assert rank in [1, 2]
        self.rank = rank
        self.sythesized = sythesized
        self.almerge = AlmergeExecutor()
        self.prepare_curves()
        self.mode = Tk.IntVar()
        self.mode.set(1)
        self.hw = WIDTH//2
        Dialog.__init__( self, parent, "RD Animation", visible=False )

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(21, 11))
        self.fig = fig
        self.axes = axes
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95, wspace=0.2)
        self.draw()
        self.mpl_canvas.draw()

    def buttonbox(self):
        frame = Tk.Frame(self)
        frame.pack(fill=Tk.X)

        box = Tk.Frame(frame)
        box.pack(side=Tk.LEFT, padx=200)
        panel = Tk.Frame(frame)
        panel.pack(side=Tk.RIGHT, padx=200)

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.build_panel(panel)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def build_panel(self, frame):
        w = Tk.Checkbutton(frame, text="animate", variable=self.mode)
        w.pack(side=Tk.LEFT, padx=20)

        f, t = self.get_range()
        self.pos = Tk.IntVar()
        self.pos.set(f+self.hw)
        scale = ttk.Scale(
            frame,
            variable=self.pos,
            orient=Tk.HORIZONTAL, 
            length=200,
            from_=f,
            to=t-WIDTH-1)
        scale.pack(side=Tk.LEFT, padx=20)

        spinbox = ttk.Spinbox(frame, textvariable=self.pos, format="%3.0f",
                                  from_=f, to=t-WIDTH-1, increment=1,
                                  justify=Tk.CENTER, width=6)
        spinbox.pack(side=Tk.LEFT, padx=20)
        self.mode_dependents = [scale, spinbox]
        self.mode_tracer()

        self.mode.trace("w", self.mode_tracer)
        self.pos.trace("w", self.position_changed)

    def mode_tracer(self, *args):
        mode = self.mode.get()
        state = Tk.DISABLED if mode else Tk.NORMAL
        for w in self.mode_dependents:
            w.config(state=state)

    def position_changed(self, *args):
        mode = self.mode.get()
        if mode:
            return

        f, t = [p + self.hw for p in self.get_range()]
        try:
            value = self.pos.get()
            i = int(round(value))
            if f <= i and i < t-WIDTH:
                self.pos.set(i)
        except:
            pass

    def prepare_curves(self):
        if self.sythesized:
            from Rank.Synthesized import synthesized_data
            self.Rg = sd.pre_recog.get_rg()

        M, E, qv, ecurve = self.sd.get_xr_data_separate_ly()
        self.qv = qv
        self.ecurve = ecurve
        self.anim_range = self.ecurve.get_ranges_by_ratio(0.1)[0]
        print('anim_range=', self.anim_range)
        print('j0=', self.j0)
        f, t = self.get_range()

        y = ecurve.y
        c = y/np.max(y)

        self.scurve_ys_list = [[], []]
        self.switch_points = []
        if self.rank == 2:
            self.bq_ys_list = [[], []]
            self.bq_lims = []
            bq_slice = slice(0, int(len(qv)*0.1))
        for k in range(f, t-WIDTH):
            j_ = slice(k, k+WIDTH)
            Mj = M[:,j_]
            c_ = c[j_]
            sp = self.get_almerge_switch_point(Mj, E[:,j_], c_)
            print([k], 'sp=', sp)
            self.switch_points.append(sp)

            if self.rank == 1:
                C_ = np.array([c_])
                Mj_ = get_denoised_data(Mj, rank=self.rank)
            else:
                C_ = np.array([c_, c_**2])
                if self.sythesized:
                    Mj_, _ = synthesized_data(self.qv, Mj, self.Rg)
                else:
                    Mj_ = get_denoised_data(Mj, rank=self.rank)

            for m, M_ in enumerate([Mj, Mj_]):
                scurve_ys = self.scurve_ys_list[m]
                P_ = M_ @ np.linalg.pinv(C_)
                scurve_ys.append(P_[:,0])
                if self.rank == 2:
                    bq_ys = self.bq_ys_list[m]
                    bq = P_[:,1]
                    bq_ys.append(bq)
                    self.bq_lims.append((np.min(bq[bq_slice]), np.max(bq[bq_slice])))

        self.noises_list = []
        for scurve_ys in self.scurve_ys_list:
            noises = []
            for k, y in enumerate(scurve_ys):
                j = f+k
                y_ = y/E[:,j]
                sy = smooth(y_)
                n = np.linalg.norm(y_ - sy)
                noises.append(n)
            self.noises_list.append(np.array(noises))

    def get_almerge_switch_point(self, Mj, Ej, c):
        temp_folder = get_setting('temp_folder')
        qv = self.qv
        files = []
        for k in range(Mj.shape[1]):
            data = np.array([qv, Mj[:,k], Ej[:,k]]).T
            file = os.path.join(temp_folder, 'data-%03d.dat' % k)
            np.savetxt(file, data)
            files.append(file)
        indeces = np.arange(Mj.shape[1], dtype=int)
        out_file = os.path.join(temp_folder, 'extrapolated.dat')
        result = self.almerge.execute(c, files, indeces, out_file)
        clear_dirs_with_retry([temp_folder])
        return result.overlap_from_max

    def get_range(self):
        f = self.anim_range[0]
        t = self.anim_range[2] + 1
        return f, t

    def draw(self):
        fig = self.fig
        ax1, ax2, ax3 = self.axes[0,:]
        ax4, ax5, ax6 = self.axes[1,:]

        fig.suptitle("Range Dependency Animation for %s with Rank(:, %d)" % (get_in_folder(), self.rank), fontsize=30)
        ax1.set_title("Elution Curve", fontsize=24)
        ax2.set_title("LRF'ed Scattering Curve", fontsize=24)
        ax3.set_title("Noise Levels", fontsize=24)

        for ax in [ax2, ax5]:
            ax.set_yscale('log')
        ecurve = self.ecurve

        x = ecurve.x + self.j0
        y = ecurve.y
        switch_points = self.switch_points

        f, t = self.get_range()
        slice_ = slice(f, f+WIDTH)
        dcurves = []
        for ax in [ax1, ax4]:
            ax.plot(x, y, color='orange')
            dcurve, = ax.plot(x[slice_], y[slice_], color='red', linewidth=5, alpha=0.5)
            dcurves.append(dcurve)

        qv = self.qv
        if self.rank == 2:
            axt_bqs = []
            bcurves = []
        scurves = []
        m = 0
        for ax, ys in zip([ax2, ax5], self.scurve_ys_list):
            if self.rank == 2:
                axt = ax.twinx()
                axt.grid(False)
                axt_bqs.append(axt)
            scurve, = ax.plot(qv, ys[0], label="Aq")
            scurves.append(scurve)
            if self.rank == 2:
                bq_ys = self.bq_ys_list[m]
                bcurve, = axt.plot(qv, bq_ys[0], color='pink', label='Bq')
                bcurves.append(bcurve)
            m += 1

        axts = []
        points = []
        for ax, noises in zip([ax3, ax6], self.noises_list):
            ax.plot(x, y, ':', color='orange')
            axt = ax.twinx()
            ax.grid(False)
            axts.append(axt)
            nj_ = slice(f, t-WIDTH)
            axt.plot(x[nj_]+self.hw, noises)
            point, = axt.plot(x[f]+self.hw, noises[0], 'o', color='red')
            points.append(point)

        def set_wider_ylim(axes, ylim=None):
            if ylim is None:
                ylims = np.array([ax.get_ylim() for ax in axes])
                ymin = np.min(ylims[:,0])
                ymax = np.max(ylims[:,1])
            else:
                ymin, ymax = ylim
            for ax in axes:
                ax.set_ylim(ymin, ymax)

        set_wider_ylim([ax2, ax5], ylim=[1e-5, 1])
        if self.rank == 2:
            bq_lims = np.array(self.bq_lims)
            ymin = np.percentile(bq_lims[:,0], 20)
            ymax = np.percentile(bq_lims[:,1], 80)
            set_wider_ylim(axt_bqs, ylim=[ymin, ymax])
        set_wider_ylim(axts)

        sp = qv[switch_points[0]]
        ymin2, ymax2 = ax2.get_ylim()
        switch_line, = ax2.plot([sp, sp], [ymin2, ymax2], color='red', label='almerge boundary')

        for ax in [ax2, ax5]:
            ax.legend()
        if self.rank == 2:
            for ax in axt_bqs:
                ax.legend(bbox_to_anchor=(1, 0.85), loc='upper right')

        artists = dcurves + scurves + points + [switch_line]
        if self.rank == 2:
            artists += bcurves

        def get_dcurve_data(i):
            f_ = f+i
            slice_ = slice(f_, f_+WIDTH)
            return x[slice_], y[slice_]

        def init():
            return artists

        def animate(i):
            mode = self.mode.get()
            if mode:
                self.pos.set(f+i+self.hw)
            else:
                i = self.pos.get() - self.hw - f
            xy = get_dcurve_data(i)
            for dcurve in dcurves:
                dcurve.set_data(*xy)
            for scurve, ys in zip(scurves, self.scurve_ys_list):
                scurve.set_data(qv, ys[i])
            if self.rank == 2:
                for bcurve, bq_ys in zip(bcurves, self.bq_ys_list):
                    bcurve.set_data(qv, bq_ys[i])
            for point, noises in zip(points, self.noises_list):
                point.set_data(x[f+i]+self.hw, noises[i])

            sp = qv[switch_points[i]]
            switch_line.set_data([sp, sp], [ymin2, ymax2])
            return artists

        num_frames = t - WIDTH - f
        self.anim = animation.FuncAnimation(self.fig, animate, init_func=init,
                                       frames=num_frames, interval=500, blit=True)
