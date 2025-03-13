"""
    Kratky.GuinierKratkyPlots.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar

def guinier_kratky_plots_impl(ax1, ax2, qv, y, rg, I0, color, interval=None, markersize=3):
    
    # task: consider unifying with Optimizer.GuinierKratkyView.py

    if ax1 is not None:
        from molass.PlotUtils.V1GuinierPlot import guinier_plot
        ax1.set_title("Guinier Plot", fontsize=16)
        guinier_plot(ax1, qv, y, color, interval, markersize)

    if ax2 is not None:
        from molass.PlotUtils.V1KratkyPlot import kratky_plot
        ax2.set_title("Normalized Kratky Plot", fontsize=16)
        kratky_plot(ax2, qv, y, rg, I0, color, markersize)

class GuinierKratkyPlots(Dialog):
    def __init__(self, parent, qv=None, y=None, rg=None, I0=None, range_=None, interval=None, color=None):
        self.qv = qv
        self.y = y
        self.rg = rg
        self.I0 = I0
        self.range_ = range_
        self.interval = interval
        self.color = color
        super().__init__(parent, "GuinierPlot", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tframe = Tk.Frame(body_frame)
        tframe.pack(fill=Tk.X)

        self.fig, self.axes = plt.subplots(ncols=2, figsize=(14,7))
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()
        if self.qv is not None:
            self.draw(self.fig, self.axes, self.qv, self.y)
            self.mpl_canvas.draw()

    def draw(self, fig, axes, qv, y):
        range_ = "" if self.range_ is None else "Range [%d, %d]" % self.range_
        title = "Guinier/Kratky Plots for %s" % range_
        interval = self.interval
        if interval is None:
            if self.rg is not None:
                logy = np.log(y)
                interval = self.guess_interval_from_rg(ax1, qv, self.rg, logy)

        fig.suptitle(title, fontsize=20)
        ax1, ax2 = axes

        guinier_kratky_plots_impl(ax1, ax2, qv, y, self.rg, self.I0, self.color, interval)

        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

    def guess_interval_from_rg(self, ax, qq, rg, logy):
        qrg = qq*rg
        i = bisect_right(qrg, 1.3)
        return 0, i
