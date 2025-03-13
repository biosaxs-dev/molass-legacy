"""
    Optimizer.ModifyStateDialog.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, root
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar

class ModifyStateDialog(Dialog):
    def __init__(self, parent, dialog):
        self.dialog = dialog
        self.current_optimizer = dialog.fullopt
        self.orig_params = dialog.canvas.get_current_params()
        self.current_params = self.orig_params
        self.dsets = dialog.canvas.dsets
        self.applied = False
        Dialog.__init__(self, parent, "Modify State", visible=False)

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

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 5))
        axt = ax2.twinx()
        self.fig = fig
        self.axes = (ax1, ax2, ax3, axt)

        axt.grid(False)
        axis_info = fig, self.axes

        fv = 0
        ax1.set_title("UV Decomposition", fontsize=16)
        ax2.set_title("Xray Decomposition", fontsize=16)
        ax3.set_title("Objective Function Scores in SV=%.3g" % fv, fontsize=16)

        self.current_optimizer.objective_func(self.orig_params, plot=True, axis_info=axis_info)

        fig.tight_layout()
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        button = Tk.Button(pframe, text="Show Parameters", command=self.show_parameters)
        button.pack(side=Tk.RIGHT, padx=20)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=20, pady=5)
        w = Tk.Button(box, text="Evaluate a new suggestion", width=30, command=self.evaluate)
        w.pack(side=Tk.LEFT, padx=20, pady=5)
        w = Tk.Button(box, text="Start a new optimization process", width=30, command=self.start)
        w.pack(side=Tk.LEFT, padx=20, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def apply(self):
        from .InitialInfo import PeProxy

        print("apply")
        self.applied = True
        pdialog = self.dialog
        optinit_info = pdialog.optinit_info
        self.pe_proxy = PeProxy(optinit_info, self.current_optimizer, self.current_params)

    def get_pe_proxy(self):
        return self.pe_proxy

    def show_parameters(self, debug=True):
        print("show_parameters")
        if debug:
            from importlib import reload
            import Optimizer.ParamsInspection
            reload(Optimizer.ParamsInspection)
        from .ParamsInspection import ParamsInspection
        # self.dialog.grab_set()  # temporary fix to the grab_release problem
        parent = self.dialog.parent
        dialog = ParamsInspection(parent, self.current_params, self.dsets, self.current_optimizer)
        dialog.show()
        # self.dialog.grab_set()  # temporary fix to the grab_release problem

    def evaluate(self):
        print("evaluate")
        temp_optimizer, new_params = self.make_a_new_suggestion()
        self.current_optimizer = temp_optimizer
        self.current_params = new_params

        fv = 0

        for ax in self.axes:
            ax.cla()
        ax1, ax2, ax3, axt = self.axes
        axt.grid(False)
        ax1.set_title("Evaluated UV Decomposition", fontsize=16)
        ax2.set_title("Evaluated Xray Decomposition", fontsize=16)
        ax3.set_title("Evaluated Objective Function Scores in SV=%.3g" % fv, fontsize=16)
        axis_info = self.fig, (ax1, ax2, ax3, axt)
        temp_optimizer.objective_func(self.current_params, debug=True, axis_info=axis_info)
        self.mpl_canvas.draw()

    def make_a_new_suggestion(self, debug=True):
        if debug:
            from importlib import reload
            import Optimizer.FunctionDebuggerUtils
            reload(Optimizer.FunctionDebuggerUtils)
        from .FunctionDebuggerUtils import create_optimizer_for_debug

        js_canvas = self.dialog.canvas
        n_components = js_canvas.fullopt.n_components
        init_params = js_canvas.optinit_info.init_params

        params = init_params

        separate_params = js_canvas.fullopt.split_params_simple(params)
        n, new_params, new_composite = self.get_enhanced_params(n_components, separate_params)

        temp_optimizer = create_optimizer_for_debug(
                    self.dsets, n, js_canvas.optinit_info,
                    None,                                   # not used when prepare=False
                    composite=new_composite, prepare=False)

        temp_optimizer.prepare_for_optimization(new_params)

        return temp_optimizer, new_params

    def get_enhanced_params(self, n, separate_params):
        from .CompositeInfo import CompositeInfo

        xr_params, xr_baseparams, rg_params, mapping, uv_params = separate_params[0:5]

        ne = xr_params.shape[0]

        if n == 6:
            new_xr_params = []
            new_rg_params = []
            new_uv_params = []
            for i in range(ne):
                xr = xr_params[i].copy()
                rg = rg_params[i]
                uv = uv_params[i]

                if i == 1:
                    xr[0] *= 0.5
                    uv *= 0.5
                    new_xr_params.append(xr)
                    new_rg_params.append(rg)
                    new_uv_params.append(uv)

                new_xr_params.append(xr)
                new_rg_params.append(rg)
                new_uv_params.append(uv)

            new_xr_params = np.array(new_xr_params)
            new_rg_params = np.array(new_rg_params)
            new_uv_params = np.array(new_uv_params)

            new_params = np.concatenate([new_xr_params.flatten(), xr_baseparams, new_rg_params, mapping, new_uv_params] + separate_params[5:])
            return n + 1, new_params, CompositeInfo(composites=[[0], [1,2], [3,4], [5], [6]])

        elif n == 7:
            new_params = np.concatenate([xr_params.flatten()] + separate_params[1:])
            return n, new_params, CompositeInfo(composites=[[0,1,2], [3,4], [5], [6]])

        else:
            # i.e., no change
            new_params = np.concatenate([xr_params.flatten()] + separate_params[1:])
            return n, new_params, self.current_optimizer.composite

    def start(self):
        import molass_legacy.KekLib.CustomMessageBox as MessageBox

        reply = MessageBox.askokcancel("Restart Confirmation",
                    "This action will create a new analysis folder\n"
                    "and start a new optimization process with\n"
                    "the modified parameters.\n"
                    "Are you sure to continue?",
                    parent=self,
                    )
        if reply:
            self.ok()
        else:
            self.cancel()
