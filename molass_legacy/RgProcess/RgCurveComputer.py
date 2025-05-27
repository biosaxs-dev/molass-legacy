"""
    Optimizer.RgCurveComputer.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
from bisect import bisect_right
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.Optimizer.TheUtils import get_optimizer_folder

class RgCurveComputerDialog(Dialog):
    def __init__(self, parent, dialog, jobno):
        self.logger = logging.getLogger(__name__)
        self.logger.info("RgCurveComputerDialog opened for job %s.", jobno)
        self.parent = parent
        self.dialog = dialog
        self.fullopt = dialog.fullopt
        self.params = dialog.canvas.get_best_params()
        self.sd = dialog.optinit_info.sd
        self.jobno = jobno
        self.applied = False
        Dialog.__init__(self, parent, "RgCurveComputer", visible=False)

    def show(self):
        # self.after(3000, self.close)
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        fig, axes = plt.subplots(ncols=2, figsize=(12,5))
        self.fig = fig

        self.axes = axes
        self.draw_recomputed_rg_curve()
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

    def draw_recomputed_rg_curve(self):
        fullopt = self.fullopt
        fv, score_list, Pxr, Cxr, Puv, Cuv, mapped_UvD = fullopt.objective_func(self.params, return_full=True)

        rg_curve = fullopt.rg_curve
        D, E, qv, xr_curve = self.sd.get_xr_data_separate_ly()

        ax1, ax2 = self.axes
        self.axt = axt = ax1.twinx()
        axt.grid(False)

        nc = fullopt.n_components

        fig = self.fig
        fig.suptitle("Recomputed Rg Curve Proof Plot", fontsize=20)

        ax1.set_title("XR Elution Components", fontsize=16)
        x = xr_curve.x
        y = xr_curve.y
        ax1.plot(x, y, color="orange")
        for k, cy in enumerate(Cxr):
            color = "red" if k == nc - 1 else None
            ax1.plot(x, cy, ":", color=color)

        # Guinier Plot
        i = bisect_right(qv, 0.05)
        gslice = slice(0, i)
        ax2.set_title("XR Scatteing Components Guinier Plot", fontsize=16)
        qv2 = qv[gslice]**2
        for k, y in enumerate(Pxr.T):
            color = "red" if k == nc - 1 else None
            ax2.plot(qv2, np.log(y[gslice]), color=color)

        fig.tight_layout()
        fig.subplots_adjust(top=0.82)

    def apply(self):
        self.applied = True
        self.dummy_simulation()

    def dummy_simulation(self):
        import shutil
        from molass_legacy.Optimizer.OptDataSets import get_current_rg_folder

        optimizer_folder = get_optimizer_folder()
        job000_folder = os.path.join(os.path.join(optimizer_folder, "jobs"), "000")
        src_rg_folder = get_current_rg_folder(possibly_relocated=True, current_folder=job000_folder)
        current_folder = os.path.join(os.path.join(optimizer_folder, "jobs"), self.jobno)
        tgt_rg_folder = get_current_rg_folder(compute_rg=True, possibly_relocated=True, current_folder=current_folder)
        shutil.copytree(src_rg_folder, tgt_rg_folder)
        self.logger.info("copied for testing from %s to %s", src_rg_folder, tgt_rg_folder)

    def close(self):
        self.logger.info("RgCurveComputerDialog closed.")
        self.cancel()
