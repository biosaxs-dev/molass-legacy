"""
    LogGuinierKratkyPlotter.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier

class LogGuinierKratkyPlotter(Dialog):
    def __init__(self, parent, data, path):
        self.data = data
        self.path = path
        Dialog.__init__( self, parent, "Log Guinier Kratky Plotter", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        from importlib import reload
        import GuinierAnalyzer.GuinierKratkyPlots
        reload(GuinierAnalyzer.GuinierKratkyPlots)
        from molass_legacy.GuinierAnalyzer.GuinierKratkyPlots import guinier_kratky_plots_impl

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        sg = SimpleGuinier(self.data)
        qv, y, e = self.data.T

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
        fig.suptitle("LGK Plot for %s" % self.path, fontsize=20)

        ax1.set_title("Log Plot", fontsize=16)
        ax1.set_yscale('log')
        ax1.plot(qv, y)

        interval = sg.guinier_start, sg.guinier_stop
        color = None
        guinier_kratky_plots_impl(ax2, ax3, qv, y, sg.Rg, sg.Iz, color, interval=interval, markersize=1)

        fig.tight_layout()
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
