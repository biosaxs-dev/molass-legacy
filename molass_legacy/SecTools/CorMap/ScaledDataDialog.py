# coding: utf-8
"""
    SecTools.CorMap.ScaledDataDialog.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import os
import logging
from bisect import bisect_right
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
from SerialDataUtils import load_intensity_files
from SaferSpinbox import SaferSpinbox
from molass_legacy.KekLib.TkSupplements import BlinkingFrame
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from molass_legacy.KekLib.TkUtils import split_geometry
from MatrixData import simple_plot_3d
from molass_legacy._MOLASS.SerialSettings import get_setting
from DataUtils import get_in_folder
from .AngularUnit import angstrom_qv
from .CormapMaker import CormapMaker
from Theory.SolidSphere import get_boundary_params_simple

ALLOW_SPAN_CHANGE = False
DATA_VIESW_NAMES = ["2D View", "3D View"]
CORREL_TYPE_NAMES = ["numpy corrcoef", "ATSAS DATCMP"]
TRANSPOSE_NAMES = ["Frames", "Angular"]

def get_version_string():
    import platform
    return 'Scaled Cormap 0.0.1 (2021-12-21 python %s %s)' % (
                platform.python_version(), platform.architecture()[0])

class ScaledDataDialog(Dialog):
    def __init__(self, parent, demo=False, elution=False, atsas=None):
        self.logger = logging.getLogger(__name__)
        self.demo = demo
        self.elution = elution
        if atsas is None:
            from Env.EnvInfo import get_global_env_info
            env_info = get_global_env_info()
            atsas = env_info.atsas_is_available
        self.threed = False
        self.atsas = atsas
        self.rect = None
        self.colorbar = None
        self.cbar_ax = None
        self.cormap_drawn = False
        self.showing_transposed = True
        self.showing_datcmp = False
        self.datafiles = None
        self.popup_menu1 = None
        self.popup_menu3 = None
        self.datcmp_data = None
        Dialog.__init__(self, parent, get_version_string(), visible=False)

    def body(self, body_frame):
        frame = Tk.Frame(body_frame)
        frame.pack(padx=20, pady=10)

        input_frame = Tk.Frame(frame)
        input_frame.pack(pady=20)

        label = Tk.Label(input_frame, text="Input Folder: ")
        label.pack(side=Tk.LEFT)

        self.in_folder = Tk.StringVar()
        self.fe = FolderEntry(input_frame, textvariable=self.in_folder, width=80,
                                on_entry_cb=self.on_in_folder_entry)
        self.fe.pack(side=Tk.LEFT)

        cframe = Tk.Frame(frame)
        cframe.pack(padx=20)

        self.fig = fig = plt.figure(figsize=(18,6))
        ax1 = fig.add_subplot(131, projection="3d")
        ax2 = fig.add_subplot(132, projection="3d")
        ax3 = fig.add_subplot(133)
        self.axes = ax1, ax2, ax3

        self.fig.tight_layout()
        # self.fig.subplots_adjust(top=0.85, left=0.05, right=0.94, bottom=0.1, wspace=1.8)
        self.fig.subplots_adjust(top=0.85, left=0.05, right=0.94, bottom=0.1)

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        tframe = Tk.Frame(frame)
        tframe.pack(padx=0, fill=Tk.X)      # padx=0: suppress changes from toolbar cursor display
        tframe_left = Tk.Frame(tframe)
        tframe_left.pack(side=Tk.LEFT, fill=Tk.X)
        tframe_right = Tk.Frame(tframe)
        tframe_right.pack(side=Tk.RIGHT)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe_left)
        self.toolbar.update()

        self.build_control_bar(tframe_right)

    def reset_cbar_ax(self):
        if self.cbar_ax is not None:
            self.cbar_ax.remove()
        self.cbar_ax = self.fig.add_axes([0.89, 0.0, 0.06, 1.0])
        self.cbar_ax.set_axis_off()

    def build_control_bar(self, frame):
        pass

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=50, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def show(self):
        if self.demo:
            self.after(500, self.perpare_temp_demo)
        self._show()

    def perpare_temp_demo(self):
        from molass_legacy._MOLASS.SerialSettings import get_setting
        in_folder = get_setting("in_folder")
        self.in_folder.set(in_folder)
        self.on_in_folder_entry()

    def on_in_folder_entry(self):
        self.update()
        in_folder = self.in_folder.get()
        try:
            data_array, datafiles = load_intensity_files(in_folder)
            self.datafiles = datafiles
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(None, "on_in_folder_entry: ")
            return

        self.set_data(data_array)

        in_folder_ = get_in_folder(in_folder.replace("\\", "/"))
        self.fig.suptitle("Correlation Analysis on %s" % in_folder_, fontsize=20)
        M_, qv_, aslice, eslice= self.compute_scaled_data()
        self.draw_input_data(M_, qv_, aslice, eslice)
        self.draw_scaled_data(M_, qv_, aslice, eslice)
        self.draw_cormap(M_, qv_, aslice, eslice)
        self.mpl_canvas.draw()

    def set_data(self, data_array):
        from molass_legacy.Elution.CurveUtils import get_xray_elution_vector
        from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
        self.qv = angstrom_qv(data_array[0,:,0])
        self.D = data_array[:,:,1].T
        self.E = data_array[:,:,2].T
        y, slice_ = get_xray_elution_vector(self.qv, data_array)
        self.xray_slice = slice_
        self.xray_index = (slice_.start + slice_.stop)//2
        self.ecurve = ElutionCurve(y)

    def get_range_info(self):
        from molass_legacy.Trimming.PreliminaryRg import PreliminaryRg
        from molass_legacy.Trimming.FlangeLimit import FlangeLimit
        from molass_legacy.Trimming.GuinierLimit import GuinierLimit

        ecurve = self.ecurve

        pinfo = ecurve.get_primary_peak_info()
        f, t = pinfo[0], pinfo[2]
        eslice = slice(f, t+1)

        f_limit = FlangeLimit(self.D, self.E, ecurve, self.qv).get_limit()
        pre_rg = PreliminaryRg(self.D, self.E, ecurve, self.qv, f_limit)
        acceptable_rg_consist = get_setting("acceptable_rg_consist")
        g_limit = GuinierLimit(self.D, self.ecurve, pre_rg, f_limit).get_limit(acceptable_rg_consist)

        b1, b2, k = get_boundary_params_simple(pre_rg.Rg)
        print("Rg=%.3g, b1=%.3g, b2=%.3g, k=%.3g" % (pre_rg.Rg, b1, b2, k))
        i = bisect_right(self.qv, b1)

        aslice = slice(g_limit, i)
        return aslice, eslice

    def compute_scaled_data(self):

        aslice, eslice = self.get_range_info()
        M = self.D[aslice,eslice]
        qv_ = self.qv[aslice]

        pinfo = self.ecurve.get_primary_peak_info()
        ptj = pinfo[1]
        pt_ridge_y = np.average(self.D[aslice,ptj-2:ptj+3], axis=1)

        scaled_list = []
        for k in range(M.shape[1]):
            y = M[:,k]
            def obj_func(p):
                return np.sum((y*p[0] - pt_ridge_y)**2)

            ret = minimize(obj_func, (1,))
            scaled_list.append(y*ret.x[0])

        M_ = np.array(scaled_list).T

        return M_, qv_, aslice, eslice

    def draw_input_data(self, M_, qv_, aslice, eslice):
        ax = self.axes[0]
        ax.cla()
        ax.set_title("Input Data", fontsize=16)
        simple_plot_3d(ax, self.D, x=self.qv)

        x = self.qv[aslice]
        for j in [eslice.start, eslice.stop-1]:
            y = np.ones(len(x))*j
            z = self.D[aslice,j]
            ax.plot(x, y, z, color="cyan")

        y = self.ecurve.x
        z = self.ecurve.y
        x = np.ones(len(y)) * self.qv[self.xray_index]
        ax.plot(x, y, z, color="orange")

    def draw_scaled_data(self, M_, qv_, aslice, eslice):

        frames = np.arange(eslice.start, eslice.stop)
        ax = self.axes[1]
        ax.cla()
        ax.set_title("Scaled Data", fontsize=16)
        simple_plot_3d(ax, M_, x=qv_, y=frames)

        j0 = eslice.start
        j1 = eslice.stop - 1
        x = qv_
        for j in [j0, j1]:
            y = np.ones(len(x))*j
            z = M_[:,j - j0]
            ax.plot(x, y, z, color="cyan")

        y = self.ecurve.x[eslice]
        i = self.xray_index - aslice.start
        if i < M_.shape[0]:
            z = M_[i,:]
            x = np.ones(len(y)) * self.qv[self.xray_index]
            ax.plot(x, y, z, color="orange")
        else:
            # as in 20161216/Backsub
            pass

    def draw_cormap(self, M_, qv_, aslice, eslice):
        from molass_legacy.ATSAS.DatCmp import run_datcmp_from_array

        E_ = self.E[aslice,eslice]
        datcmp_data = run_datcmp_from_array(qv_, M_, E_, P_threshold=0.05)

        ax = self.axes[2]
        ax.cla()
        self.reset_cbar_ax()

        ax.set_title("CorMap of Scaled Data", fontsize=16)

        frames = np.arange(eslice.start, eslice.stop)
        cm = CormapMaker(datcmp_data, qv_, frames, "Frame No.", transpose=True, from_datcmp=True)
        cm.draw(ax, self.cbar_ax)

    def save_the_figure(self, file):
        self.fig.savefig(file)
