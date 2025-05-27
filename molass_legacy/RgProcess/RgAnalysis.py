# coding: utf-8
"""
    RgProcess.RgAnalysis.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
try:
    import molass_legacy.KekLib.CustomMessageBox as MessageBox
except:
    import OurMessageBox as MessageBox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Button
from .RgCurve import RgCurve

class RgAnalysisDialog(Dialog):
    def __init__(self, parent, si):
        parent.report_callback_exception = self.report_callback_exception
        self.parent = parent
        self.si = si
        print('X=', si.X.shape)
        print('curve_xy=', si.curve_xy.shape)
        self.xy = si.curve_xy
        self.q = si.X[0,:,0]
        self.q2 = self.q**2
        self.D = si.X[:,:,1].T
        self.E = si.X[:,:,2].T

        self.in_folder = str(si.in_folder)
        Dialog.__init__(self, parent, "Guinier Analysis", visible=False)

    def report_callback_exception(self, exc, val, tb):
        # This method is to override the Tk method to be able
        # to report to the spawned console in windows application mode.
        from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker

        etb = ExceptionTracebacker()
        msg = ( 'Overridden report_callback_exception: ' + str( etb ) )
        MessageBox.showerror( "Tkinter Error", msg, parent=self.parent )
        self.parent,quit()

    def show(self):
        self._show()

    def body(self, body_frame):
        dummy_label = Tk.Label(body_frame, text=self.in_folder)
        dummy_label.pack()

        self.cframe = Tk.Frame(body_frame)
        self.cframe.pack()

        fig, axes = plt.subplots(ncols=3, figsize=(21,7))

        self.fig = fig
        self.axes = axes

        self.mpl_canvas = FigureCanvasTkAgg(fig, self.cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        self.j = 150

        self.L_button = Button(plt.axes([0.04, 0.5, 0.04, 0.04]), '$ < $')
        self.L_button.on_clicked(lambda *args: self.move_step(-1))
        self.R_button = Button(plt.axes([0.27, 0.5, 0.04, 0.04]), '$ > $')
        self.R_button.on_clicked(lambda *args: self.move_step(+1))

        fig.tight_layout()
        self.reraw()

    def reraw(self):
        self.draw_ecurve(self.axes[0])
        self.draw_scurve(self.axes[1:])
        self.mpl_canvas.draw()

    def draw_ecurve(self, ax):
        x, y = self.xy
        ax.plot(x, y)
        j = self.j
        ax.plot(x[j], y[j], 'o', color='yellow')

    def draw_scurve(self, axes):
        j = self.j
        x = self.q2
        y = np.log(self.D[:,j])
        for ax in axes:
            ax.plot(x, y)

    def move_step(self, i):
        self.j = max(0, min(self.D.shape[1], self.j+i))
        print("move_step to", self.j)
        for ax in self.axes:
            ax.cla()
        self.reraw()
