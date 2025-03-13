"""
    RgVisibleFigure.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from GuinierTools.RgCurveUtils import compute_rg_curves
from .FvScoreConverter import convert_score
from .LrfExporter import LrfExporter

DX_RATIO = 0.05
DY_RATIO = 0.04
SQ_DIST = DX_RATIO**2 + DY_RATIO**2     # squared distance in normalized space

class RgVisibleFigure(Dialog):
    def __init__(self, parent, js_canvas, with_range=False):
        self.parent = parent
        self.js_canvas = js_canvas
        self.with_range = with_range
        self.elution_model = js_canvas.elution_model
        self.fullopt = js_canvas.fullopt
        self.class_code = js_canvas.dialog.class_code
        self.params = js_canvas.get_current_params()
        self.separate_params = self.fullopt.split_params_simple(self.params)

        self.fv = convert_score(js_canvas.get_current_fv())
        # exporter = self.fullopt.params_type.get_exporter(js_canvas.optinit_info.sd, js_canvas.dsets, self.fullopt, self.params)
        exporter = LrfExporter(self.fullopt, self.params, js_canvas.dsets)
        # exporter.compute_LRF_xr()
        self.peaktops = exporter.xr_result.peaktops
        Dialog.__init__(self, parent, "Rg-visible Figure", visible=False)

    def get_secparams(self):
        secparams = self.separate_params[7]
        if self.elution_model == 1:
            t0, rp, N, me, T, mp = secparams
            P = N*T
            m = me+mp
        else:
            t0, P, rp, m = secparams[0:4]
        return t0, P, rp, m

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tframe = Tk.Frame(body_frame)
        tframe.pack(fill=Tk.X)

        fig_width = 6
        if self.with_range:
            t0, P, rp, m = self.get_secparams()
            self.sec_range = t0, t0+P
            x = self.fullopt.xr_curve.x
            width = max(t0+P, x[-1]) - min(t0, x[0])
            ratio = width/len(x)
            fig_width = min(18, fig_width*ratio)

        fig, ax = plt.subplots(figsize=(fig_width,5))
        self.fig = fig
        axt = ax.twinx()
        axt.grid(False)
        self.axes = (ax, axt)
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        self.draw_figure(fig, ax, axt)

    def draw_figure(self, fig, ax, axt):
        in_folder = get_in_folder()
        nc = self.fullopt.n_components - 1
        job_name = self.js_canvas.dialog.get_job_info()[0]
        ax.set_title("%s, NC=%d, F=%s, SV=%.3g, J=%s" % (in_folder, nc, self.class_code, self.fv, job_name), fontsize=14, y=1.05)
        ax.set_xlabel("Time (Eno)")
        ax.set_ylabel("Intensity")
        axt.set_ylabel("$R_g$")

        fullopt = self.fullopt

        x = fullopt.xr_curve.x
        y = fullopt.xr_curve.y
        lrf_info = fullopt.objective_func(self.params, return_lrf_info=True)
        xr_cy_list = lrf_info.get_xr_cy_list()
        xr_ty = lrf_info.xr_ty

        ax.plot(x, y, color="orange")

        stop = -2 if fullopt.separate_eoii else -1
        for k, cy in enumerate(xr_cy_list[0:stop]):
            ax.plot(x, cy, ":")

        if fullopt.separate_eoii:
            cy = xr_cy_list[-1]
            ax.plot(x, cy, ":", color="pink")

        cy = xr_cy_list[stop]
        ax.plot(x, cy, color="red")

        ax.plot(x, xr_ty, ":", color="red")

        xr_params, _, rg_params = self.separate_params[0:3]

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)

        if self.with_range:
            xmin, xmax = ax.get_xlim()
            dx = (xmax - xmin)*0.02
            k = 0
            for t_, color in zip(self.sec_range, ["red", "green"]):
                ax.plot([t_, t_], [ymin, ymax], ":", color=color, lw=3)
                if k == 0:
                    ax.text(t_+dx, ymin, "$t_0$", fontsize=20, color=color, ha="left", va="bottom")
                else:
                    ax.text(t_-dx, ymin, "$t_0 + P$", fontsize=20, color=color, ha="right", va="bottom")
                k += 1

        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax)

        xsize = (xmax - xmin)
        ysize = (ymax - ymin)
        dx = xsize*0.02
        dy = ysize*0.02

        n = len(self.peaktops)
        m, r = divmod(n, 2)

        text_pos = []
        def find_better_pos(tx, ty):
            for x, y in text_pos:
                dist2 = ((tx - x)/xsize)**2 + ((ty - y)/ysize)**2
                if dist2 < SQ_DIST:
                    tx += np.sign(tx - x) * xsize * DX_RATIO
                    ty += np.sign(ty - y) * ysize * DY_RATIO
                    break
            return tx, ty

        for k, (px, py) in enumerate(self.peaktops):
            if px < 0:
                continue

            ax.plot(px, py, "o", color="yellow")
            dy_ = dy
            if k <= m:
                if r == 1 and k == m:
                    ha = "center"
                    dx_ = 0
                else:
                    ha = "right"
                    dx_ = -dx
            else:
                ha = "left"
                dx_ = dx
            tx = px+dx_
            ty = py+dy_
            tx, ty = find_better_pos(tx, ty)
            ax.text(tx, ty, "%.3g" % rg_params[k], ha=ha, fontsize=16, alpha=0.5)
            text_pos.append((px, py))
            text_pos.append((tx, ty))

        # Rg curves
        if self.with_range:
            from SecTheory.ConformanceDemo import SecConfCurve
            t0, P, rp, m = self.get_secparams()
            secconf_curve = SecConfCurve(t0, P, rp, m, size=100)
            x_ = x[x > t0]
            y_ = secconf_curve(x_)
            axt.plot(x_, y_, ":", color="purple", lw=3, label="conformance curve")

        xr_ty_ = xr_ty - xr_cy_list[-1]      # remove baseline part
        if len(xr_params.shape) == 2:
            xr_heights = xr_params[:,0]
        else:
            xr_heights = xr_params
        rg_curves1, rg_curves2 = compute_rg_curves(x, xr_heights/np.max(xr_heights), rg_params, xr_cy_list[:-1], xr_ty_, self.fullopt.rg_curve)

        k = 0
        for (x1, rg1), (x2, rg2) in zip(rg_curves1, rg_curves2):
            axt.plot(x1, rg1, color='gray', alpha=0.5)
            axt.plot(x2, rg2, ':', color='black')
            k += 1

        ymin, ymax = axt.get_ylim()
        axt.set_ylim(min(10, ymin), max(50, ymax))
        # axt.legend(loc='upper right', fontsize=16)

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        self.mpl_canvas.draw()
