"""
    SecTheory.MwRgFigure.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from .MwRgInfo import get_mwrg_info
import molass_legacy.KekLib.CustomMessageBox as MessageBox

class MwRgFigure(Dialog):
    def __init__(self, parent, excl_limit, **kwargs):
        self.excl_limit = excl_limit
        self.pickable = kwargs.pop('pickable', True)
        Dialog.__init__(self, parent, "Mw-Rg Figure", visible=False, **kwargs)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()

        self.fig, self.ax = plt.subplots()
        self.draw_mwrg_figure(self.ax)
        self.fig.tight_layout()

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas.draw()

        if self.pickable:
            self.mpl_canvas.mpl_connect('button_press_event', self.on_click)

    def buttonbox(self):
        lower_frame = Tk.Frame(self)
        lower_frame.pack(fill=Tk.BOTH, expand=1)

        width = int(self.mpl_canvas_widget.cget('width'))
        padx = width*0.05

        tframe = Tk.Frame(lower_frame)
        tframe.pack(side=Tk.LEFT, padx=padx)
        self.toolbar = NavigationToolbar( self.mpl_canvas, tframe )
        self.toolbar.update()

        space = Tk.Frame(lower_frame, width=padx)
        space.pack(side=Tk.RIGHT)

        box = Tk.Frame(lower_frame)
        box.pack(side=Tk.RIGHT)

        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=padx, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def draw_mwrg_figure(self, ax):
        mwrg_info = get_mwrg_info(devel=True)
        self.mwrg_info = mwrg_info

        excl_limit =  self.excl_limit
        Rg = mwrg_info.compute_rg(excl_limit)

        ax.set_title(r"Rg=%.3g ($\AA$) from the Exclusion Limit %g (kDa)" % (Rg, excl_limit), fontsize=20)
        ax.set_xlabel("Molecular Weight (kDa)", fontsize=16)
        ax.set_ylabel(r"Rg ($\AA$)", fontsize=16)

        ax.plot(mwrg_info.mws, mwrg_info.rgs, "o")

        ax.plot(excl_limit, Rg, "o", color="red", alpha=0.2, markersize=20)

        # xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xmax = excl_limit * 1.3
        ymax = Rg * 1.3
        xmin = -xmax*0.1
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        x = np.linspace(xmin, xmax, 100)
        lower = mwrg_info.compute_rg(x, n_sigma=-1)
        upper = mwrg_info.compute_rg(x, n_sigma=+1)
        ax.fill_between(x, lower, upper, color="cyan", alpha=0.2, label=r"1$\sigma$ uncertainty")

        y = mwrg_info.compute_rg(x)
        ax.plot(x, y, ":")

        lower_rg = mwrg_info.compute_rg(excl_limit, n_sigma=-1)
        upper_rg = mwrg_info.compute_rg(excl_limit, n_sigma=+1)
        ax.plot([xmin, excl_limit], [Rg, Rg], ":", color="gray")
        ax.plot([excl_limit, excl_limit], [ymin, upper_rg], ":", color="gray")

        for rg in [lower_rg, upper_rg]:
            ax.plot([xmin, excl_limit], [rg, rg], ":", color="gray")

        dx = (xmax - xmin)*0.05
        dy = (ymax - ymin)*0.05
        for rg in [lower_rg, Rg, upper_rg]:
            ax.annotate("%.3g" % rg, xy=(xmin, rg), xytext=(xmin + dx, rg - dy),
                        ha='left', va="top", arrowprops=dict(arrowstyle="->", color='gray'),
                        fontsize=16)

        ax.legend(loc="lower right", fontsize=16)

    def on_click(self, event):
        if event.button != 1:
            return

        if event.xdata is None or event.ydata is None:
            return

        mws = self.mwrg_info.mws
        rgs = self.mwrg_info.rgs
        dist2 = (mws - event.xdata)**2 + (rgs - event.ydata)**2
        m = np.argmin(dist2)
        # print("m=", m, self.mwrg_info.names[m])
        MessageBox.showinfo("Mw-Rg Data Info",
            "Mw=%.3g, Rg=%.3g for %s" % (mws[m], rgs[m], self.mwrg_info.names[m]),
            parent=self
            )
