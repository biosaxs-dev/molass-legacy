"""
    Optimizer.ComplementaryView.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from bisect import bisect_right
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy._MOLASS.SerialSettings import get_setting
from DataUtils import get_in_folder
from molass_legacy.KekLib.BasicUtils import ordinal_str
from .LrfExporter import LrfExporter
from molass_legacy.SerialAnalyzer.AnalyzerUtil import compute_conc_factor_util
from molass_legacy.Saxs.SaxsCurveUtils import percentile_normalize
from molass_legacy._MOLASS.Version import is_developing_version

TOGGLE_BUTTON_TEXTS = ["Show", "Hide"]

class ComplementaryView(Dialog):
    def __init__(self, parent, optimizer, curr_index, params, work_folder=None, sd=None):
        self.devel = is_developing_version()
        self.parent = parent
        self.optimizer = optimizer
        self.xr_curve = optimizer.xr_curve
        self.uv_curve = optimizer.uv_curve
        self.xrD = optimizer.xrD
        self.uvD = optimizer.uvD
        self.xrD_ = optimizer.xrD_
        self.uvD_ = optimizer.uvD_
        self.qv = optimizer.qvector
        self.wv = optimizer.wvector
        self.curr_index = curr_index
        self.params = params
        self.separate_params = optimizer.split_params_simple(params)
        self.lrf_info = optimizer.objective_func(params, return_lrf_info=True)
        self.work_folder = work_folder
        self.sd = sd
        self.show_baseline = 0
        self.separate_eoii = optimizer.separate_eoii
        self.popup_menu = None
        Dialog.__init__(self, parent, "Complementary View", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(21, 11))
        fig = plt.figure(figsize=(18, 9))
        gs = GridSpec(2, 10)
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[1,0])
        self.label_axes = (ax0, ax1)
        axes_list = []
        for i in range(2):
            row_list = []
            for j in range(3):
                jb = j*3 + 1
                row_list.append(fig.add_subplot(gs[i,jb:jb+3]))
            axes_list.append(row_list)
        axes = np.array(axes_list)
        self.fig = fig
        self.axes = axes
        self.axt = axes[1,0].twinx()
        self.draw()
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas.draw()
        if self.devel:
            fig.canvas.mpl_connect('button_press_event', self.on_click)

    def buttonbox(self):
        lower_frame = Tk.Frame(self)
        lower_frame.pack(fill=Tk.BOTH, expand=1)

        width = int(self.mpl_canvas_widget.cget('width'))
        padx = width*0.03

        tframe = Tk.Frame(lower_frame)
        tframe.pack(side=Tk.LEFT, padx=padx)
        self.toolbar = NavigationToolbar( self.mpl_canvas, tframe )
        self.toolbar.update()

        space = Tk.Frame(lower_frame, width=padx)
        space.pack(side=Tk.RIGHT)

        box = Tk.Frame(lower_frame)
        box.pack(side=Tk.RIGHT)

        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=padx, pady=5)

        text = self.get_toggle_button_text()
        w = Tk.Button(box, text=text, width=12, command=self.toggle_baseline)
        w.pack(side=Tk.LEFT, padx=padx, pady=5)
        self.toggle_btn = w

        w = Tk.Button(box, text="Guinier-Kratky View", width=16, command=self.show_guinier_kratky_view)
        w.pack(side=Tk.LEFT, padx=padx, pady=5)

        w = Tk.Button(box, text="Compare to PDB (CRYSOL)", width=22, command=self.show_compare_to_pdb)
        w.pack(side=Tk.LEFT, padx=padx, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def get_toggle_button_text(self):
        return "%s Baseline" % TOGGLE_BUTTON_TEXTS[self.show_baseline]

    def cancel(self):
        plt.close(self.fig)
        Dialog.cancel(self)

    def draw(self):
        self.draw_axis_labels()
        self.draw_complements()

    def draw_axis_labels(self):
        for ax in self.label_axes:
            ax.cla()
            ax.set_axis_off()
        ax0, ax1 = self.label_axes
        ax0.text(0.5, 0.5, "UV", ha="center", fontsize=20)
        ax1.text(0.5, 0.5, "Xray", ha="center", fontsize=20)

    def draw_complements(self):
        fig = self.fig
        axes = self.axes
        for ax_row in axes:
            for ax in ax_row:
                ax.cla()
        self.axt.cla()
        self.axt.grid(False)

        in_folder = get_in_folder()
        fig.suptitle("Complement View at %s local minimum on %s" % (ordinal_str(self.curr_index), in_folder), fontsize=20)
        ax0, ax1, ax2 = axes[0,:]
        ax0.set_title("Used Elution Components", fontsize=16)
        ax1.set_title("Linear Scale", fontsize=16)
        ax2.set_title("Log Scale", fontsize=16)

        ax00 = axes[0,0]
        ax10 = axes[1,0]
        self.optimizer.objective_func(self.params, plot=True, axis_info=(self.fig, [ax00, ax10, None, self.axt]))
        ax00.set_xlabel(r"$\lambda$")
        ax00.set_ylabel("Absorbance")
        ax10.set_xlabel("Q")
        ax10.set_ylabel("Intensity")

        lrf_info = self.lrf_info
        Pxr, Cxr, Puv, Cuv, mapped_Uv = lrf_info.matrices
        self.draw_complement_impl(axes[0,1:], self.wv, Puv, Cuv,
                color="blue", value_label="Absorbance", vector_label=r"$\lambda$")

        conc_factor = compute_conc_factor_util()
        Cxr_ = Cxr/conc_factor
        self.draw_complement_impl(axes[1,1:], self.qv, Pxr, Cxr_,
                color="orange", value_label="Intensity (max normalized)", vector_label="Q", xray=True)

        self.adjust_xlim_mapping(axes[0,0], axes[1,0])

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

    def draw_complement_impl(self, axes, vector, P, C, color=None, value_label="", vector_label="", xray=False):
        if not xray:
            if False:
                import UV.Foldedness
                from importlib import reload
                reload(UV.Foldedness)
            from molass_legacy.UV.Foldedness import Foldedness
            foldedness = Foldedness(self.wv)

        num_others = 2 if xray and self.separate_eoii else 1
        num_components = len(C) - num_others
        draw_baseline = self.show_baseline == 1

        PT = P.T[0:num_components+1]
        if xray:
            PT = [percentile_normalize(v) for v in PT]

        for k, ax in enumerate(axes):
            if k == 1:
                if xray:
                    ax.set_yscale('log')
                else:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "Suppressed", fontsize=20, alpha=0.3, ha="center", va="center")
                    continue

            for j, v in enumerate(PT):
                if j < num_components:
                    if xray:
                        fnstr = ""
                    else:
                        fnstr = ", FnR=%.1f" % foldedness.compute(v)
                    ax.plot(vector, v, label="component-%d%s" % (j+1, fnstr), alpha=0.5)
                else:
                    if draw_baseline:
                        ax.plot(vector, v, color='red', label="baseline", alpha=0.5)

            if k == 0 and not xray:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
                for w, color in zip(foldedness.get_wavelengths(), foldedness.get_plotcolors()):
                    ax.plot([w, w], [ymin, ymax], ":", label="wavelength=%g" % w, color=color)

            if xray and self.separate_eoii:
                v = P.T[-1]
                ax.plot(vector, v, '-', color="pink", label="effect of ii", alpha=0.5)

            ax.set_xlabel(vector_label)
            ax.set_ylabel(value_label)
            ax.legend()

    def adjust_xlim_mapping(self, ax0, ax1):
        a, b = self.separate_params[3]
        xmin, xmax = [a*x + b for x in ax1.get_xlim()]
        ax0.set_xlim(xmin, xmax)

    def toggle_baseline(self):
        self.show_baseline = 1 - self.show_baseline
        self.draw()
        self.mpl_canvas.draw()
        text = self.get_toggle_button_text()
        self.toggle_btn.config(text=text)
        self.update()

    def show_guinier_kratky_view(self):
        try:
            self.show_guinier_kratky_view_impl()
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(None, "show_guinier_kratky_view: ")

    def show_guinier_kratky_view_impl(self, debug=False):
        if debug:
            from importlib import reload
            import Kratky.GuinierKratkyView
            reload(Kratky.GuinierKratkyView)
        from Kratky.GuinierKratkyView import GuinierKratkyView

        dialog = GuinierKratkyView(self.parent, self.optimizer, self.params, self.lrf_info)
        dialog.show()

    def show_compare_to_pdb(self, devel=True):
        if devel:
            from importlib import reload
            import Theory.PdbCrysolRoute
            reload(Theory.PdbCrysolRoute)
        from Theory.PdbCrysolRoute import compare_to_pdb_impl
        compare_to_pdb_impl(self)

    def on_click(self, event):
        if event.xdata is None:
            return

        if event.button == 3:
            from molass_legacy.KekLib.TkUtils import split_geometry
            self.create_popup_menu(event)
            rootx = self.winfo_rootx()
            rooty = self.winfo_rooty()
            w, h, x, y = split_geometry(self.mpl_canvas_widget.winfo_geometry())
            self.popup_menu.post(rootx + int(event.x), rooty + h - int(event.y))
            return

    def create_popup_menu(self, event):
        if self.popup_menu is None:
            self.popup_menu = Tk.Menu(self, tearoff=0)
            self.popup_menu.add_command(label='Show Simple 3D View', command=self.show_simple_3d_view)

    def show_simple_3d_view(self, debug=True):
        if debug:
            import Tools.SimpleThreedView
            from importlib import reload
            reload(Tools.SimpleThreedView)
        from molass_legacy.Tools.SimpleThreedView import show_simple_3d_view
        show_simple_3d_view(self.sd)
