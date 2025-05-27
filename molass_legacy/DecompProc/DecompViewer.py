# coding: utf-8
"""
    DecompProcess.DecompViewer.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import sleep
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import molass_legacy.KekLib.DebugPlot as dplt
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar

class DecompViewer(Dialog):
    def __init__(self, parent):
        self.draw_counter = 0
        Dialog.__init__(self, parent, "Decomposition Viewer", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):   # overrides parent class method

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        bframe = Tk.Frame(body_frame)
        bframe.pack(fill=Tk.X, expand=1)
        self.bframe = bframe
        self.fig = plt.figure( figsize=(21,11) )
        self.mpl_canvas = FigureCanvasTkAgg( self.fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.draw()

    def buttonbox(self):
        tframe = Tk.Frame(self.bframe)
        tframe.pack(side=Tk.LEFT)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        box = Tk.Frame(self.bframe)
        box.pack(side=Tk.RIGHT)

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def draw(self):
        fig = self.fig

        gs = GridSpec(3, 4)

        ax1 = fig.add_subplot(gs[0,0])
        axt = ax1.twinx()
        axt.grid(False)

        ax2 = fig.add_subplot(gs[0,1], projection='3d')
        ax3 = fig.add_subplot(gs[0,2])
        ax4 = fig.add_subplot(gs[0,3])

        axes = np.empty((2,4), dtype=object)
        for i in range(2):
            for j in range(4):
                axes[i,j] = fig.add_subplot(gs[i+1,j])

        fig.tight_layout()

        self.after(100, lambda: self.draw_subplots(fig, axes))
        self.mpl_canvas.draw()

    def draw_subplots(self, fig, axes):
        print("draw_counter=", self.draw_counter)

        i, j = divmod(self.draw_counter, 4)

        ax = axes[i,j]
        ax.plot(0, 0, 'o', markersize=50)
        fig.canvas.draw()

        self.draw_counter += 1
        if self.draw_counter < 8:
            self.after(1000, lambda: self.draw_subplots(fig, axes))
