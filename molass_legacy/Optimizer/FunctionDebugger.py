"""
    Optimizer.FunctionDebugger.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from SecTheory.BasicModels import robust_single_pore_pdf as monopore_pdf
from molass_legacy.Baseline.BaselineUtils import create_xr_baseline_object
from molass_legacy._MOLASS.SerialSettings import get_setting

class FunctionDebugger(Dialog):
    def __init__(self, parent, dialog, js_canvas, composite=None, debug=True):
        if debug:
            from importlib import reload
            import Optimizer.FunctionDebuggerUtils
            reload(Optimizer.FunctionDebuggerUtils)
        from .FunctionDebuggerUtils import create_optimizer_for_debug

        self.parent = parent
        self.dialog = dialog
        self.js_canvas = js_canvas
        self.dsets = dsets = js_canvas.dsets
        ecurve = dsets[0][0]
        self.x = ecurve.x
        self.y = ecurve.y
        self.elution_model = js_canvas.elution_model

        self.fullopt = create_optimizer_for_debug(
                            js_canvas.dsets, js_canvas.fullopt.n_components,
                            js_canvas.optinit_info, js_canvas.optinit_info.init_params,
                            composite=composite
                            )

        self.params = js_canvas.get_current_params()
        self.separate_params = self.fullopt.split_params_simple(self.params)
        Dialog.__init__(self, parent, "Function Debugger", visible=False)

    def get_xr_params(self):
        return self.separate_params[0]

    def get_rg_params(self):
        return self.separate_params[2]

    def get_sec_params(self):
        return self.separate_params[-1]

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tframe = Tk.Frame(body_frame)
        tframe.pack(fill=Tk.X)

        fig, axes = plt.subplots(ncols=3, figsize=(18,5))
        self.fig = fig

        axt = axes[1].twinx()
        axt.grid(False)
        self.axes = list(axes)
        self.axes.append(axt)

        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        self.popup_menu = None
        self.start_debugger()

    def start_debugger(self):
        from importlib import reload

        import Optimizer.TheDebugUtils
        reload(Optimizer.TheDebugUtils)
        import Optimizer.FvScoreConverter
        reload(Optimizer.FvScoreConverter)

        from .FvScoreConverter import convert_score

        if self.elution_model in [0, 2]:
            import ModelParams.EghPlotUtils as plot_utils
        else:
            import ModelParams.StcPlotUtils as plot_utils
        reload(plot_utils)

        if False:
            import Optimizer.CompositeInfo
            reload(Optimizer.CompositeInfo)
            from .CompositeInfo import CompositeInfo

            composites = [[0], [1,2], [3], [4]]
            composite = CompositeInfo(composites=composites)
            optworking_folder = get_setting('optworking_folder')
            path = os.path.join(optworking_folder, 'composite_info.txt')
            composite.save(path)
            composite.load(path)
            assert str(composites) == str(composite.composites)

        # should be unified with JobStateCanvas.draw_indexed_state()
        fig = self.fig
        axes = self.axes
        ax1, ax2, ax3, axt = self.axes
        ax1.set_title("UV Decomposition", fontsize=16)
        ax2.set_title("Xray Decomposition", fontsize=16)
        fv = self.fullopt.objective_func(self.params)
        ax3.set_title("Objective Function Scores in SV=%.3g" % convert_score(fv), fontsize=16)

        self.temp_params = self.params.copy()
        self.temp_params[4*3+2] = 35.9
        self.fullopt.objective_func(self.temp_params, plot=True, axis_info=(self.fig, self.axes))

        fig.tight_layout()
        self.mpl_canvas.draw()

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack(fill=Tk.X, padx=20)

        num_buttons = 4

        for j in range(num_buttons):
            box.columnconfigure(j, weight=1)

        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        col = 0
        w.grid(row=0, column=col, pady=10)
        self.cancel_btn = w

        col += 1
        w = Tk.Button(box, text="Show Bounds Inspection", width=24, command=self.show_bounds_indpection)
        w.grid(row=0, column=col, pady=10)

        col += 1
        w = Tk.Button(box, text="Show Normalized Parameters", width=24, command=self.show_norm_pameters)
        w.grid(row=0, column=col, pady=10)

        col += 1
        w = Tk.Button(box, text="Show Parameters", width=24, command=self.show_parameters)
        w.grid(row=0, column=col, pady=10)

        col += 1
        w = Tk.Button(box, text="Test", width=10, command=self.test)
        w.grid(row=0, column=col, pady=10)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def show_bounds_indpection(self, debug=True):
        if debug:
            from importlib import reload
            import Optimizer.BoundsInspection
            reload(Optimizer.BoundsInspection)
        from molass_legacy.Optimizer.BoundsInspection import BoundsInspection
        parent = self.js_canvas.dialog.parent
        dialog = BoundsInspection(parent, self.fullopt, self.temp_params)
        dialog.show()

    def show_norm_pameters(self, debug=True):
        if debug:
            from importlib import reload
            import Optimizer.ParamsInspection
            reload(Optimizer.ParamsInspection)
        from .ParamsInspection import ParamsInspection
        # self.dialog.grab_set()  # temporary fix to the grab_release problem
        norm_params = self.fullopt.to_norm_params(self.temp_params)
        parent = self.js_canvas.dialog.parent
        dialog = ParamsInspection(parent, norm_params, self.dsets, self.fullopt)     # consider better positioning (or positioning bug?)
        dialog.show()
        # self.dialog.grab_set()  # temporary fix to the grab_release problem

    def show_parameters(self, debug=True):
        if debug:
            from importlib import reload
            import Optimizer.ParamsInspection
            reload(Optimizer.ParamsInspection)
        from .ParamsInspection import ParamsInspection
        # self.dialog.grab_set()  # temporary fix to the grab_release problem
        parent = self.js_canvas.dialog.parent
        dialog = ParamsInspection(parent, self.temp_params, self.dsets, self.fullopt)     # consider better positioning (or positioning bug?)
        dialog.show()
        # self.dialog.grab_set()  # temporary fix to the grab_release problem

    def test(self):
        from importlib import reload
        import Optimizer.StructureFactorBoundsDemo
        reload(Optimizer.StructureFactorBoundsDemo)
        from .StructureFactorBoundsDemo import demo, demo_for_debugger

        self.dialog.grab_set()  # temporary fix to the grab_release problem
        demo_for_debugger(self.dialog, self.js_canvas)
        self.dialog.grab_set()  # temporary fix to the grab_release problem
