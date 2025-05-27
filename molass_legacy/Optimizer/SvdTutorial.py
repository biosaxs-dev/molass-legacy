"""
    Optimizer.SvdTutorial.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar

class SvdTutorial(Dialog):
    def __init__(self, parent, dialog):
        self.parent = parent
        self.dialog = dialog

        Dialog.__init__(self, parent, "SVD Tutorial", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()

        fig = plt.figure(figsize=(18,11))

        self.fig = fig
        self.draw()
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas.draw()

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

    def draw(self):
        fig = self.fig
        gs = GridSpec(2,4)

        axes = []
        for i in range(2):
            axes_row = []
            for j in range(4):
                ax = fig.add_subplot(gs[i,j], projection="3d")
                axes_row.append(ax)
            axes.append(axes_row)

        fig.tight_layout()
