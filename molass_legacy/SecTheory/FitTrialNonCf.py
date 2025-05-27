# coding: utf-8
"""
    SecTheory.FitTrialNonCf.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy._MOLASS.SerialSettings import set_setting
from molass_legacy.Batch.StandardProcedure import StandardProcedure
from molass_legacy.Baseline.BaselineUtils import get_corrected_sd_impl
from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
from molass_legacy.Peaks.ElutionModels import egh
from LPM import get_corrected
from DataUtils import get_in_folder
from SecSaxs.DataTreatment import DataTreatment

class DemoDialog(Dialog):
    def __init__(self, parent, ecurve, rg_curve, rp_params, rg_params, wt_params):
        self.ecurve = ecurve
        self.rg_curve = rg_curve
        self.rp_params = rp_params
        self.rg_params = rg_params
        self.wt_params = wt_params
        Dialog.__init__(self, parent, "Fitting Test", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):

        left_frame = Tk.Frame(body_frame)
        left_frame.grid(row=0, column=0)
        right_frame = Tk.Frame(body_frame)
        right_frame.grid(row=0, column=1)

        # 
        self.coding_text = Tk.Text(left_frame)
        self.coding_text.pack()
        for varname, params in [("rp_params", self.rp_params),
                                ("rg_params", self.rg_params),
                                ("wt_params", self.wt_params),
                                ]:
            self.coding_text.insert(Tk.END, varname + " = " + str(list(params)) + "\n")

        # 
        row = 0

        self.n_species = Tk.IntVar()
        self.n_species.set(len(self.rg_params))

        label = Tk.Label(right_frame, text="n_species: ")
        label.grid(row=row, column=0, sticky=Tk.E, pady=10)

        sbox  = Tk.Spinbox(right_frame, textvariable=self.n_species,
                          from_=0, to=30, increment=1,
                          justify=Tk.CENTER, width=6, state=Tk.DISABLED)
        sbox.grid(row=row, column=1, sticky=Tk.W, pady=10)

        row += 1

        self.n_iters = Tk.IntVar()
        self.n_iters.set(10)

        label = Tk.Label(right_frame, text="n_iters: ")
        label.grid(row=row, column=0, sticky=Tk.E, pady=10)

        sbox  = Tk.Spinbox(right_frame, textvariable=self.n_iters,
                          from_=0, to=100, increment=1,
                          justify=Tk.CENTER, width=6)
        sbox.grid(row=row, column=1, sticky=Tk.W, pady=10)

        row += 1

        self.n_trials = Tk.IntVar()

        label = Tk.Label(right_frame, text="n_trials: ")
        label.grid(row=row, column=0, sticky=Tk.E, pady=10)

        sbox  = Tk.Spinbox(right_frame, textvariable=self.n_trials,
                          from_=0, to=30, increment=1,
                          justify=Tk.CENTER, width=6)
        sbox.grid(row=row, column=1, sticky=Tk.W, pady=10)

        row += 1

        run_btn = Tk.Button(right_frame, text="Run", command=self.run)
        run_btn.grid(row=row, column=0, columnspan=2, pady=10)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.bind("<Escape>", self.cancel)

    def run(self):
        from importlib import reload
        import SecTheory.SinglePoreNonCf
        reload(SecTheory.SinglePoreNonCf)
        from .SinglePoreNonCf import fit_single_pore_pdf
        text = self.coding_text.get("1.0", Tk.END)
        temp_namespace = {}
        exec(text, temp_namespace)
        params_list = []
        for varname in ["rp_params", "rg_params", "wt_params"]:
            params_list.append(np.array(temp_namespace.get(varname)))

        init_params = np.concatenate(params_list)
        fit_single_pore_pdf(self.ecurve, self.rg_curve, init_params,
                                n_species=len(params_list[1]),
                                n_iters=self.n_iters.get(),
                                n_trials=self.n_trials.get(),
                                )

def fitting_demo(root, in_folder, rg_folder, rp_params, rg_params, wt_params):

    sp = StandardProcedure()
    sd = sp.load_old_way(in_folder)

    pre_recog = PreliminaryRecognition(sd)
    treat = DataTreatment(route="v2", trimming=2, correction=1)
    sd_copy = treat.get_treated_sd(sd, pre_recog)

    D, E, qv, ecurve = sd_copy.get_xr_data_separate_ly()

    if rg_folder is None:
        assert False
    else:
        from RgProcess.RgCurveProxy import RgCurveProxy
        rg_curve = RgCurveProxy(ecurve, rg_folder)

    dialog = DemoDialog(root, ecurve, rg_curve, rp_params, rg_params, wt_params)
    dialog.geometry("+500+500")
    dialog.show()
