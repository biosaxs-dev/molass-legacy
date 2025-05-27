"""
    Models/RateTheory/EdmParamInspect.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar


class EdmParamInspect(Dialog):
    def __init__(self, parent, dialog):
        self.parent = parent
        self.dialog = dialog
        Dialog.__init__(self, parent, "Edm Parameter Inspection", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        bframe = Tk.Frame(body_frame)
        bframe.pack(fill=Tk.X, padx=20)

        tframe = Tk.Frame(bframe)
        tframe.pack(side=Tk.LEFT)
        pframe = Tk.Frame(bframe, width=100)
        pframe.pack(side=Tk.RIGHT)

        fig, ax = plt.subplots()
        self.fig = fig
        fig.tight_layout()
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

def demo(caller):
    dialog = caller.dialog
    inspect = EdmParamInspect(dialog, dialog)
    inspect.show()
