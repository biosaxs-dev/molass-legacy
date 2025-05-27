"""
    Optimizer.ParameterTransitionPlot.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.DebugPlot import push, pop, get_parent
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from molass_legacy.KekLib.ScrolledFrame import ScrolledFrame
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Optimizer.FvScoreConverter import convert_scores
from molass_legacy.Optimizer.ParamsIterator import create_iterator

class ParameterTransitionPlot(Dialog):
    def __init__(self, parent, optimizer, params_array, bounds_array, fv_list):
        self.optimizer = optimizer
        self.params_array = params_array
        self.bounds_array = bounds_array
        self.fv_list = fv_list
        Dialog.__init__(self, parent, "Parameter Transition Plot", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        title_frame = Tk.Frame(body_frame)
        title_frame.pack(side=Tk.TOP, fill=Tk.X, expand=1)
        in_folder = get_in_folder()
        result_folder = get_setting('optimizer_folder').replace('/optimized', '')
        func_name = self.optimizer.get_name()
        title = Tk.Label(title_frame,
                         text="Parameter Transition on %s from %s with %s" % (in_folder, result_folder, func_name),
                         bg="white", font=(None,20))
        title.pack(fill=Tk.X, expand=1)
        sv_frame = ScrolledFrame(body_frame)
        sv_frame.pack(side=Tk.TOP)

        self.btn_frame = Tk.Frame(body_frame)
        self.btn_frame.pack(side=Tk.BOTTOM)
        tool_frame = Tk.Frame(body_frame)
        tool_frame.pack(side=Tk.BOTTOM)
        pv_frame = ScrolledFrame(body_frame)
        pv_frame.pack(side=Tk.BOTTOM)

        sv_frame.set_sync_info(pv_frame.canvas_info)
        pv_frame.set_sync_info(sv_frame.canvas_info)

        params_array = self.params_array
        nrows, ncols = params_array.shape

        width = 40
        sv_fig, sv_ax = plt.subplots(figsize=(width, 3))
        pv_fig, pv_axes = plt.subplots(nrows=ncols, figsize=(width,ncols*3))
        self.draw_transition(sv_fig, sv_ax, pv_fig, pv_axes)
        self.sv_canvas = FigureCanvasTkAgg(sv_fig, sv_frame.interior)
        self.sv_canvas_widget = self.sv_canvas.get_tk_widget()
        self.sv_canvas_widget.pack(fill=Tk.X, expand=1)
        self.pv_canvas = FigureCanvasTkAgg(pv_fig, pv_frame.interior)
        self.pv_canvas_widget = self.pv_canvas.get_tk_widget()
        self.pv_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.pv_canvas, tool_frame)
        self.toolbar.update()
        self.after(500, lambda: self.geometry("2000x800"))

    def buttonbox(self):
        box = self.btn_frame
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

    def draw_transition(self, sv_fig, sv_ax, pv_fig, pv_axes):
        sv_list = []
        minima_points_list = []
        start = 0
        for fv_array in self.fv_list:
            sv_list.append(convert_scores(fv_array))
            for i in create_iterator(fv_array):
                minima_points_list.append(start + i)
            start += len(fv_array)
        niter_vector = np.cumsum([len(v) for v in sv_list])
        minima_points = np.array(minima_points_list)
        def plot_job_sections_with_minima(ax):
            for niter in niter_vector[:-1]:
                ax.axvline(niter, color="gray", alpha=0.3)
            for i in minima_points:
                ax.axvline(i, color="red", ls=':')
        sv_ax.set_ylabel("SV", fontsize=16)
        sv_vector = np.concatenate(sv_list)
        sv_ax.plot(sv_vector)
        plot_job_sections_with_minima(sv_ax)
        sv_ax.set_ylim(0, 100)
        names = self.optimizer.get_parameter_names()
        for ax, name, params in zip(pv_axes, names, self.params_array.T):
            ax.set_ylabel(name, fontsize=16)
            ax.plot(params)
            plot_job_sections_with_minima(ax)

        for fig in [sv_fig, pv_fig]:
            fig.tight_layout()
            fig.subplots_adjust(left=0.03, right=0.97)

def plot_transition_impl(optimizer, params_array, bounds_array, fv_list):
    push()
    parent = get_parent()
    dialog = ParameterTransitionPlot(parent, optimizer, params_array, bounds_array, fv_list)
    dialog.show()
    pop()