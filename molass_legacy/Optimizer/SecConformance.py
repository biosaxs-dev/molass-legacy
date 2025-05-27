"""
    Optimizer.SecConformance.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tksheet import Sheet
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from DataUtils import get_in_folder
from molass_legacy.KekLib.BasicUtils import ordinal_str, Struct
from SecTheory.RetensionTime import compute_retention_time

class SecConformance(Dialog):
    def __init__(self, parent, parent_dialog, state_canvas):
        self.parent = parent
        self.parent_dialog = parent_dialog
        self.state_canvas = state_canvas
        self.fullopt = state_canvas.fullopt
        self.x_array = state_canvas.demo_info[1]
        self.curr_index = state_canvas.get_curr_index()
        self.params = self.x_array[self.curr_index]
        self.n = self.fullopt.n_components
        self.separate_params = self.fullopt.split_params_simple(self.params)
        self.seccol_params = self.separate_params[-1]
        self.separate_optimization = Tk.IntVar()
        self.separate_optimization.set(0)
        self.sheet_frame = None
        self.sheet = None
        Dialog.__init__(self, parent, "SEC Conformance", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):

        figure_frame = Tk.Frame(body_frame)
        figure_frame.pack(side=Tk.LEFT)
        self.right_frame = Tk.Frame(body_frame)
        self.right_frame.pack(side=Tk.LEFT, padx=20)

        cframe = Tk.Frame(figure_frame)
        cframe.pack()
        tframe = Tk.Frame(figure_frame)
        tframe.pack(fill=Tk.X, padx=20)
        tframe_left = Tk.Frame(tframe)
        tframe_left.pack(side=Tk.LEFT)

        self.fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
        axt = ax1.twinx()
        self.axes = (ax1, ax2, axt)
        self.show_sec_conformance()

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack(fill=Tk.X)
        for j in range(2):
            box.columnconfigure(j, weight=1)

        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.grid(row=0, column=0, pady=5)

        option_frame = Tk.Frame(box)
        option_frame.grid(row=0, column=1)
        cb = Tk.Checkbutton(option_frame, text="separate optimization", variable=self.separate_optimization)
        cb.pack(side=Tk.LEFT)
        w = Tk.Button(option_frame, text="Refresh", width=10, command=self.show_sec_conformance)
        w.pack(side=Tk.LEFT)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def show_sec_conformance(self):
        rgs = self.separate_params[2]
        trs = self.separate_params[0][:,1]

        if self.separate_optimization.get():
            result = self.fullopt.params_type.estimate_conformance_params(rgs, trs)
            model_trs = compute_retention_time(result.x, rgs)
        else:
            seccol_params = self.separate_params[-1]
            model_trs = compute_retention_time(seccol_params, rgs)
            result = self.make_pseudo_result(seccol_params, trs, model_trs)

        fig = self.fig
        for ax in self.axes:
            ax.cla()
        ax1, ax2, axt = self.axes
        axt.grid(False)

        in_folder = get_in_folder()
        job_info = self.state_canvas.dialog.get_job_info()
        job_name = job_info[0]

        fig.suptitle("Conformance with Stochastic Theory of SEC on %s job %s at %s state" % (in_folder, job_name, ordinal_str(self.curr_index)), fontsize=20)
        ax1.set_title("Xray Decomposition", fontsize=16)

        axis_info = (self.fig, (None, ax1, None, axt))
        self.state_canvas.fullopt.objective_func(self.params, debug=True, axis_info=axis_info)

        ax2.set_title("Single Pore Model Conformance: Score=%.3g" % result.fun, fontsize=16)
        ax2.set_xlabel("Retention Time (Eno)")
        ax2.set_ylabel("Rg")
        ax2.plot(model_trs, rgs, "o-", label="stochastic model")
        ax2.plot(trs, rgs, "o-",label="currently optimized")
        ax2.legend()

        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        self.make_data_sheet(rgs, trs, result, model_trs)

    def make_data_sheet(self, rgs, trs, result, model_trs):
        if self.sheet_frame is not None:
            self.sheet_frame.destroy()
        if self.sheet is not None:
            self.sheet.destroy()

        self.sheet_frame = Tk.Frame(self.right_frame)
        self.sheet_frame.pack()
        sheet_frame = self.sheet_frame
        sheet_title = Tk.Label(sheet_frame, text="Conformance Data Sheet", font=(None, 20))
        sheet_title.pack(pady=10)
        sheet_body = Tk.Frame(sheet_frame)
        sheet_body.pack(pady=10)

        data_list = self.get_sheet_data_list(rgs, trs, result, model_trs)

        num_rows = len(rgs) +12
        num_columns = 3
        column_width = 90
        width = column_width*num_columns + 60
        height = int(22*num_rows) + 60
        self.sheet = Sheet(sheet_body, width=width, height=height, data=data_list, show_selected_cells_border=False, column_width=column_width)
        self.sheet.pack()
        self.sheet.enable_bindings()

    def get_sheet_data_list(self, rgs, trs, result, model_trs):
        colnames = ["Rg", "Model tR", "Solution tR"]
        data_list = [colnames]
        for rg, model_tr, tr in zip(rgs, model_trs, trs):
            data_list.append(["%.3g" % v for v in [rg, model_tr, tr]])

        data_list.append([])
        t0, K, rp, m = result.x
        data_list.append(["rp (pore size)", "%.3g" % rp])
        data_list.append(["t0", "%.3g" % t0])
        data_list.append(["K", "%.3g" % K])
        data_list.append(["m", "%.3g" % m])

        data_list.append([])
        data_list.append(["fit success", str(result.success)])
        data_list.append(["fit status", str(result.status)])
        data_list.append(["fit message", result.message])

        data_list.append([])
        data_list.append(["score", "%.3g" % result.fun])

        return data_list

    def make_pseudo_result(self, seccol_params, trs, model_trs):
        fun = np.sqrt(np.average((model_trs - trs)**2))
        return Struct(x=seccol_params, success=True, status=0, message="Global", fun=fun)
