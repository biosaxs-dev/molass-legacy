# coding: utf-8
"""
    GuinierRegion.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from bisect import bisect_right
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from MatrixData import simple_plot_3d
from DataUtils import get_in_folder

MARGIN_WIDTH = 0.13     # angstrom⁻¹

class GuinierRegionInspector(Dialog):
    def __init__(self, parent, dialog, data, i_smp, eslice):
        self.logger = logging.getLogger(__name__)
        self.dialog = dialog
        sd = dialog.serial_data
        pre_rg = dialog.pre_rg
        self.sd = sd
        self.sg = pre_rg.sg
        qv = sd.qvector
        margin  = int(round(MARGIN_WIDTH/(qv[1] - qv[0])))
        self.i_smp = i_smp
        self.i_stop = max(i_smp, self.sg.guinier_stop) + margin
        self.eslice = eslice
        self.aslice = aslice = slice(0, self.i_stop)
        self.gqv = qv[aslice]
        self.j0 = eslice.start
        self.gdata = data[aslice,eslice].copy()
        canvas2, canvas3 = self.dialog.frames[0].canvases[1:3]
        self.ptx = canvas2.opos
        self.ecurve = canvas2.ecurve
        self.asx = canvas3.x[canvas3.restrict_info.start]
        self.x = canvas3.x[aslice]      # same as self.gqv
        self.xx = self.x**2
        self.y = canvas3.y[aslice]
        self.compute_corrected_info()

        Dialog.__init__(self, parent, "Guinier Region Inspector", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()

        fig = plt.figure(figsize=(18,7))
        self.fig = fig

        in_folder = get_in_folder()
        fig.suptitle("Guinier Region Inspection on %s" % in_folder, fontsize=20)
        ax1 = fig.add_subplot(131, projection="3d")
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        self.draw_3d(ax1)
        self.determin_2d_fig_limit()
        self.draw_default_start(ax2)
        self.draw_corrected_start(ax3)

        fig.tight_layout()
        fig.subplots_adjust(top=0.85, bottom=0.1)

        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

    def draw_3d(self, ax):
        ax.set_title("Guinier Region 3D View", fontsize=16)

        framenums = self.j0 + np.arange(self.gdata.shape[1])
        simple_plot_3d(ax, self.gdata, x=self.gqv, y=framenums)

        length = self.gdata.shape[1]
        y = np.arange(length) + self.j0
        i_smp = self.i_smp
        for i, label in [(self.sg.guinier_start, "Guinier Start"), (self.sg.guinier_stop, "Guinier Stop"), (i_smp, "SMP")]:
            x = np.ones(length)*self.gqv[i]
            z = self.gdata[i,:]
            ax.plot(x, y, z, color="orange" if i == i_smp else "cyan", label=label)

        x = self.gqv
        y = np.ones(len(x)) * self.ptx
        z = self.y
        ax.plot(x, y, z, color="green", label="Peak Top Ridge")

        ax.legend()

    def determin_2d_fig_limit(self):
        qmax = 2.0/self.sg.Rg
        self.q2max = qmax**2
        print("q2max=", self.q2max)
        i = bisect_right(self.gqv, qmax)
        self.ymin = np.log(self.y[i])

    def draw_default_start(self, ax):
        self.draw_start_impl(ax, "Default Range Start")

    def draw_corrected_start(self, ax):
        self.draw_start_impl(ax, "Corrected Range Start", info=self.corrected_info)

    def draw_start_impl(self, ax, title, info=None):

        if info is None:
            x = self.x
            y = self.y
            sg = self.sg
            asx = self.asx
            y_orig = None
        else:
            x, y, sg, asx = info
            y_orig = self.y

        ax.set_title(title, fontsize=16)

        ax.set_xlabel("$Q^2$")
        ax.set_ylabel("$Ln(I)$")

        xx = x**2
        if y_orig is not None:
            ax.plot(xx, np.log(y_orig), ":", color="gray")
        ax.plot(xx, np.log(y), color="green")

        _, ymax = ax.get_ylim()
        ymin = self.ymin
        ax.set_ylim(ymin, ymax)

        k = 0
        gx = x[[sg.guinier_start, sg.guinier_stop]]
        for x_, label in [(asx, "Range Start"), (gx[0], "Guinier Start"), (gx[1], "Guinier Stop")]:
            color = "yellow" if k == 0 else "cyan"
            x2 = x_**2
            ax.plot([x2, x2], [ymin, ymax], color=color, label=label)
            k += 1

        self.plot_guinier_line(ax, sg)

        xmin, xmax = ax.get_xlim()
        w = 0.3
        tx = xmin*(1-w) + xmax*w
        w = 0.6
        ty = ymin*(1-w) + ymax*w
        ax.text(tx, ty, "Range Start = $%.3g^2$" % asx, fontsize=20, alpha=0.5)

        ax.legend()

    def compute_corrected_info(self):
        from LPM import LPM_3d
        from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
        from molass_legacy._MOLASS.SerialSettings import get_setting
        from .PreliminaryRg import PreliminaryRg
        from .GuinierLimit import GuinierLimit

        lpm = LPM_3d(self.gdata)
        corrected_data = lpm.data
        j = self.ptx - self.j0
        y = corrected_data[:,j]
        ey = np.average(corrected_data[self.i_smp-5:self.i_smp+6,:], axis=0)
        error_data = self.sd.intensity_array[self.eslice, self.aslice, 2].T
        e_curve = ElutionCurve(ey)
        pre_rg = PreliminaryRg(corrected_data, error_data, e_curve, self.gqv, self.i_stop)
        gl = GuinierLimit(corrected_data, e_curve, pre_rg, self.i_stop)
        rg_consistency = get_setting('acceptable_rg_consist')
        try:
            angle_start = gl.get_limit(rg_consistency)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(self.logger, "GuinierLimit.get_limit failed.", n=10)
            angle_start = 0

        asx = self.gqv[angle_start]

        if False:
            import molass_legacy.KekLib.DebugPlot as dplt
            from molass_legacy.Elution.CurveUtils import simple_plot
            dplt.push()
            fig, ax = dplt.subplots()
            ax.set_title("compute_corrected_info debug")
            simple_plot(ax, e_curve)
            dplt.show()
            dplt.pop()

        self.corrected_info = (self.x, y, pre_rg.sg, asx)

    def plot_guinier_line(self, ax, sg):
        # see ProofPlot.plot_guineier_inverval
        Rg, I0, f, t = sg.Rg, sg.Iz, sg.guinier_start, sg.guinier_stop

        x = self.xx
        b_ = np.log(I0)
        a_ = -(Rg**2 / 3)
        y0 = b_ + a_ * x[f]
        y1 = b_ + a_ * x[t]
        ax.plot( [ x[f], x[t] ], [ y0, y1 ], marker='o', color="red", label="Guinier Line")

        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin/4 if xmin < 0 else xmin, min(xmax, self.q2max))

    def save_the_figure(self, file):
        self.fig.savefig(file)
