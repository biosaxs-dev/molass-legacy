"""
    Optimizer.ParamsInspection.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
from molass_legacy.KekLib.OurTkinter import Tk, Dialog

ENABLE_OPTIONAL_BUTTONS = False 

class ParamsInspection(Dialog):
    def __init__(self, parent, params, dsets, optimizer, state_info=None):
        self.parent = parent
        self.params = params
        self.dsets = dsets
        self.optimizer = optimizer
        self.state_info = state_info
        Dialog.__init__(self, parent, "Parameter Inspection", visible=False, location="lower right")

    def show(self):
        self._show()

    def body(self, body_frame):
        self.psheet = self.optimizer.params_type.get_params_sheet(body_frame, self.params, self.dsets, self.optimizer)
        self.psheet.enable_copy()
        self.psheet.pack()

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack(fill=Tk.X, padx=50)

        num_buttons = 2 if self.state_info is None else 3
        if ENABLE_OPTIONAL_BUTTONS:
            num_buttons += 2
        for k in range(num_buttons):
            box.columnconfigure(k, weight=1)

        col = 0
        w = Tk.Button(box, text="◀ Close", width=10, command=self.cancel)
        w.grid(row=0, column=col, pady=10)

        if self.state_info is not None:
            w = Tk.Button(box, text="Highlight Active Parameters", width=24, command=self.highlight)
            col += 1
            w.grid(row=0, column=col, pady=10)

        w = Tk.Button(box, text="Save", width=10, command=self.save)
        col += 1
        w.grid(row=0, column=col, pady=10)

        if ENABLE_OPTIONAL_BUTTONS:
            w = Tk.Button(box, text="SEC Conformance", width=14, command=self.show_conformance)
            col += 1
            w.grid(row=0, column=col, pady=10)

            w = Tk.Button(box, text="SEC Simulation", width=14, command=self.show_simulator)
            col += 1
            w.grid(row=0, column=col, pady=10)

        self.bind("<Escape>", self.cancel)

    def highlight(self):
        try:
            self.psheet.highlight_active_params(self.state_info)
        except:
            import molass_legacy.KekLib.CustomMessageBox as MessageBox
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(None, "highlight failed: ", n=10)
            MessageBox.showerror("Error", "Not supported yet.", parent=self.parent)

    def save(self):
        import os
        from tkinter import filedialog
        from molass_legacy._MOLASS.SerialSettings import get_setting

        initialdir = get_setting("optjob_folder")
        f = filedialog.asksaveasfilename(
            title="名前を付けて保存",
            # filetypes=[("Excel Books", "*.xlsx"), ("CSV Text Files", "*.csv")],
            filetypes=[("CSV Text Files", "*.csv")],
            initialdir=initialdir,
            initialfile="parameters.csv",
            parent=self,
            )
        if not f:
            return

        self.psheet.save_as(f)

    def show_simulator(self, debug=True):
        if debug:
            from importlib import reload
            import SecTheory.SecSimulator
            reload(SecTheory.SecSimulator)
        from SecTheory.SecSimulator import SecSimulator

        self.grab_set()  # temporary fix to the grab_release problem
        dialog = SecSimulator(self, self.optimizer, self.params)
        dialog.show()
        self.grab_set()  # temporary fix to the grab_release problem

    def show_conformance(self, debug=True):
        if debug:
            from importlib import reload
            import SecTheory.SecParamsPlot
            reload(SecTheory.SecParamsPlot)
        from SecTheory.SecParamsPlot import plot_sec_params

        lrf_info = self.optimizer.objective_func(self.params, return_lrf_info=True)
        self.grab_set()  # temporary fix to the grab_release problem
        plot_sec_params(self.optimizer, self.params, lrf_info)
        self.grab_set()  # temporary fix to the grab_release problem
