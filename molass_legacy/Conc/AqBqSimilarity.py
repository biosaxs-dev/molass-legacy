# coding: utf-8
"""
    AqBqSimilarity.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
from bisect import bisect_right
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker
from matplotlib.widgets import Button
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkUtils import split_geometry
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar

@ticker.FuncFormatter
def major_formatter(x, pos):
    apparant = False if pos is None else (pos - 1)%2 == 0
    return "%.0f" % x if apparant else ""

class AqBqSimilarityDialog(Dialog):
    def __init__( self, parent, dialog):
        self.parent = parent
        self.dialog = dialog
        self.q = dialog.q
        self.A2 = A2 = dialog.A2
        B2 = dialog.B2
        q_slice = slice(dialog.q_index, dialog.q_index2)
        scale = np.average(A2[q_slice])/np.average(B2[q_slice])
        self.B2 = B2*scale

        Dialog.__init__( self, parent, "A(a)-B(q) Similarity", visible=False )

    def show(self):
        self._show()

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        self.fig = fig = plt.figure(figsize=(14,7))
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        gs = GridSpec(3,2)
        ax11 = fig.add_subplot(gs[0:2,0])
        ax21 = fig.add_subplot(gs[2,0])
        ax12 = fig.add_subplot(gs[0:2,1])
        ax22 = fig.add_subplot(gs[2,1])

        fig.suptitle("A(q)-B(q) Similarity Inspection", fontsize=20)
        self.draw_curves(ax11, ax21, ax12, ax22)

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
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

        box = Tk.Frame(bottom_frame)
        box.pack(side=Tk.RIGHT)

        w = Tk.Button(box, text="Close", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.RIGHT, padx=padx, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def draw_curves(self, ax11, ax21, ax12, ax22):
        ax11.set_title("Scaled Log Plot", fontsize=16)
        ax11.set_yscale('log')
        ax11.plot(self.q, self.A2, label='A(q)', color='C1')
        ax11.plot(self.q, self.B2, label='scaled B(q)', color='pink', alpha=0.5)
        ax11.axes.get_xaxis().set_ticks([])
        ax11.legend()

        ax21.plot(self.q, np.log10(self.A2) - np.log10(self.B2), label='log(A(q)) - log(scaled B(q))')
        ax21.legend()

        ax12.set_title("Scaled Linear Plot", fontsize=16)
        ax12.plot(self.q, self.A2, label='A(q)', color='C1')
        ax12.plot(self.q, self.B2, label='scaled B(q)', color='pink', alpha=0.5)
        ax12.axes.get_xaxis().set_ticks([])
        ax12.legend()

        y = self.dialog.compute_min_norm_diff_curve(self.A2, self.B2)
        ax22.plot(self.q, y, label='A(q) - B(q)')
        ax22.legend()

        y_ = y[0:self.dialog.q_limit]
        cd_score = np.sqrt(np.average(y_**2))/self.dialog.scale
        self.dialog.draw_score_text(ax22, y[-1], cd_score*100)
