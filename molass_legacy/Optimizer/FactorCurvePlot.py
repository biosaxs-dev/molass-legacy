"""
    Optimizer.FactorCurvePlot.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.special import iv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from SecTheory.BasicModels import robust_single_pore_pdf as monopore_pdf

class FactorCurvePlot(Dialog):
    def __init__(self, parent, inspect):
        self.inspect = inspect
        Dialog.__init__(self, parent, "Factor Curve Plot", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tframe = Tk.Frame(body_frame)
        tframe.pack(fill=Tk.X)

        fig, axes = plt.subplots(ncols=3, figsize=(18,5))
        self.fig = fig
        self.axes = axes

        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        self.draw_curves(fig, axes)

    def draw_curves(self, fig, axes):
        inspect = self.inspect
        t0, d, N, me, T, mp = inspect.get_values()
        xr_params = inspect.get_xr_params()
        rg_params = inspect.get_rg_params()
        rho = rg_params/d
        rho[rho > 1] = 1

        ax1, ax2, ax3 = axes

        x = inspect.x
        y = inspect.y
        axts = []
        for ax in axes:
            ax.plot(x, y, color="orange")
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            axt = ax.twinx()
            axt.grid(False)
            axts.append(axt)

        ones = np.ones(len(x))
        t = x - t0
        nc = len(rg_params)

        for k, r in enumerate(rho):
            np_ = N * (1 - r)**me
            tp_ = T * (1 - r)**mp
            w = xr_params[k]

            w = (k+1)/(nc+1)
            gy = ones*(ymin*w + ymax*(1-w))

            # iv(1, np.sqrt(4*np_*t/tp_))
            v = iv(1, np.sqrt(4*np_*t/tp_))
            finite = np.isfinite(v)
            ax1.plot(x[finite], gy[finite], "o", markersize=1)
            axts[0].plot(x, w * v, ":")

            # np.sqrt(np_/(t*tp_))
            v = np.sqrt(np_/(t*tp_))
            finite = np.isfinite(v)
            ax2.plot(x[finite], gy[finite], "o", markersize=1)
            axts[1].plot(x, w * v, ":")

            # np.exp(-t/tp_-np_)
            v = np.exp(-t/tp_-np_)
            finite = np.isfinite(v)
            ax3.plot(x[finite], gy[finite], "o", markersize=1)
            axts[2].plot(x, w * v, ":")

        fig.tight_layout()
        self.mpl_canvas.draw()
