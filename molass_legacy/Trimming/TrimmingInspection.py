# coding: utf-8
"""
    TrimmingInspection.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from DataUtils import get_in_folder

MARGIN_WIDTH = 0.01     # angstrom⁻¹

class ElutionTrimmingDialog(Dialog):
    def __init__(self, parent, dialog, data, ecurve, trimming_info):
        self.ecurve = ecurve
        self.trimming_info = trimming_info
        Dialog.__init__(self, parent, "Elution Trimming Inspector", visible=False, location="lower right")

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()

        fig = plt.figure(figsize=(12,6))
        self.fig = fig

        ax1 = fig.add_subplot(111)

        self.draw_elution(ax1)

        fig.tight_layout()
        # fig.subplots_adjust(top=0.85, bottom=0.1)

        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        tframe = Tk.Frame(body_frame)
        tframe.pack(padx=0, fill=Tk.X)      # padx=0: suppress changes from toolbar cursor display
        tframe_left = Tk.Frame(tframe)
        tframe_left.pack(side=Tk.LEFT, fill=Tk.X)
        tframe_right = Tk.Frame(tframe)
        tframe_right.pack(side=Tk.RIGHT)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe_left)
        self.toolbar.update()

    def cancel(self):
        plt.close(self.fig)
        Dialog.cancel(self)

    def draw_elution(self, ax):
        from molass_legacy.Elution.CurveUtils import simple_plot

        in_folder = get_in_folder()
        ax.set_title("Elution Trimming Inspection on %s" % in_folder, fontsize=20)
        ecurve = self.ecurve

        simple_plot(ax, ecurve, color="orange", legend=False)

        trinfo = self.trimming_info
        extra = trinfo.extra

        if extra is not None:

            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)

            emg_peaks = [extra.info_L[0]]
            if extra.num_peaks > 1:
                emg_peaks.append(extra.info_R[0])

            ex = ecurve.x
            for peak in emg_peaks:
                y = peak.get_model_y(ex)
                ax.plot(ex, y, ":", color="blue", label="original elution model", lw=2)

            sinfo = extra.sigma_info
            for k, px in enumerate([sinfo.jmin, sinfo.jmax]):
                label = "original range" if k == 0 else None
                ax.plot([px, px], [ymin, ymax], color="blue", label=label)

            recon_info = self.trimming_info.extra.recon_info
            if recon_info is not None:
                rem_y = recon_info.prm_model(ex)
                ax.plot(ex, rem_y, ":", color="red", label="revised elution model", lw=2)

                for k, j in enumerate([recon_info.jmin, recon_info.jmax]):
                    label = "revised range" if k == 0 else None
                    ax.plot([j, j], [ymin, ymax], color="red", label=label)

            for k, px in enumerate([trinfo.start, trinfo.stop]):
                label = "final range" if k == 0 else None
                ax.plot([px, px], [ymin, ymax], ":", color="yellow", label=label, lw=2)

        ax.legend()

    def save_the_figure(self, file):
        self.fig.savefig(file)
