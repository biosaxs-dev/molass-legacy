"""
    Optimizer.ResidualView.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.BasicUtils import ordinal_str
from DataUtils import get_in_folder
from MatrixData import simple_plot_3d

class ResidualView(Dialog):
    def __init__(self, parent, parent_dialog, state_canvas):
        self.parent = parent
        self.parent_dialog = parent_dialog
        self.state_canvas = state_canvas
        self.sd = state_canvas.optinit_info.sd
        self.x_array = state_canvas.demo_info[1]
        self.best_index = state_canvas.get_best_index()
        self.fullopt = state_canvas.fullopt
        Dialog.__init__(self, parent, "Residual View", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tframe = Tk.Frame(body_frame)
        tframe.pack(fill=Tk.X, padx=20)
        tframe_left = Tk.Frame(tframe)
        tframe_left.pack(side=Tk.LEFT)

        self.fig = fig = plt.figure(figsize=(12, 6))
        self.draw_residual_view(fig)

        fig.tight_layout()
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

    def buttonbox( self ):
        box = Tk.Frame(self)
        box.pack()
        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def draw_residual_view(self, fig):

        job_info = self.parent_dialog.get_job_info()
        job_name = job_info[0]
        in_folder = get_in_folder()

        best_params = self.x_array[self.best_index]
        fv, score_list, Pxr, Cxr, Puv, Cuv, mapped_UvD = self.fullopt.objective_func(best_params, return_full=True)

        Ruv = Puv @ Cuv - mapped_UvD
        Rxr = Pxr @ Cxr - self.fullopt.xrD_

        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")

        # fig.tight_layout()
        fig.subplots_adjust(top=0.8)

        fig.suptitle("Residual Views of Job %s at %s local minimium on %s" % (job_name, ordinal_str(self.best_index), in_folder), fontsize=20)
        ax1.set_title("UV Residual 3D View", fontsize=16)
        ax2.set_title("XR Residual 3D View", fontsize=16)

        simple_plot_3d(ax1, Ruv, x=self.sd.lvector)
        simple_plot_3d(ax2, Rxr, x=self.sd.qvector)
