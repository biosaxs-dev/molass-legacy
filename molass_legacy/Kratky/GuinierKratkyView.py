"""
    Kratky.GuinierKratkyView.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import os
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from GuinierTools.RgCurveUtils import plot_rg_curves
from molass_legacy._MOLASS.SerialSettings import get_setting

PLOT_EOII_IN_GUINIER = False
TOGGLE_BUTTON_TEXTS = ["Show", "Hide"]

class GuinierKratkyView(Dialog):
    def __init__(self, parent, optimizer, params, lrf_info):
        self.parent = parent
        self.optimizer = optimizer
        self.params = params
        self.lrf_info = lrf_info
        self.separate_eoii = get_setting("separate_eoii")
        self.show_baseline = 0
        self.all_axes = []

        Dialog.__init__(self, parent, "Guinier-Kratky View", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()

        # fig, axes = plt.subplots(ncols=4, figsize=(20,5), subplot_kw=dict(projection="3d"))
        fig = plt.figure(figsize=(18,9))

        self.fig = fig
        self.draw()
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas.draw()

    def buttonbox(self):
        lower_frame = Tk.Frame(self)
        lower_frame.pack(fill=Tk.BOTH, expand=1)

        width = int(self.mpl_canvas_widget.cget('width'))
        padx = width*0.05

        tframe = Tk.Frame(lower_frame)
        tframe.pack(side=Tk.LEFT, padx=padx)
        self.toolbar = NavigationToolbar( self.mpl_canvas, tframe )
        self.toolbar.update()

        space = Tk.Frame(lower_frame, width=padx)
        space.pack(side=Tk.RIGHT)

        box = Tk.Frame(lower_frame)
        box.pack(side=Tk.RIGHT)

        if False:
            w = Tk.Button(box, text="Tutorial Figure", width=15, command=self.show_tutorial_figure)
            w.pack(side=Tk.LEFT, pady=5)

        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=padx, pady=5)

        text = self.get_toggle_button_text()
        w = Tk.Button(box, text=text, width=12, command=self.toggle_baseline)
        w.pack(side=Tk.LEFT, padx=padx, pady=5)
        self.toggle_btn = w

        w = Tk.Button(box, text="Show Reducibility", width=15, command=self.show_reducibility)
        w.pack(side=Tk.LEFT, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def get_toggle_button_text(self):
        return "%s Baseline" % TOGGLE_BUTTON_TEXTS[self.show_baseline]

    def draw(self, debug=True):
        from molass_legacy.DataStructure.MatrixData import simple_plot_3d
        from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
        if debug:
            from importlib import reload
            import Kratky.GuinierKratkyInfo
            reload(Kratky.GuinierKratkyInfo)
        from Kratky.GuinierKratkyInfo import GuinierKratkyInfo

        lrf_info = self.lrf_info
        gk_info = GuinierKratkyInfo(self.optimizer, self.params, lrf_info, self.show_baseline)
        self.gk_info = gk_info

        rg_params = gk_info.rg_params       # got valid_rgs
        points = lrf_info.estimate_xr_peak_points()
        heights = points[:,1]
        ip, iq = sorted(np.argpartition(heights,-2)[-2:])
        ib = len(rg_params)

        Pxr, Cxr, Puv, Cuv  = lrf_info.matrices[:4]

        fig = self.fig
        gs = GridSpec(2,12)

        self.clear_axes()

        axes = []
        for i in range(3):
            ax = fig.add_subplot(gs[0,i*4:(i+1)*4])
            axes.append(ax)
            self.all_axes.append(ax)
        ax1_, ax2_, ax3_ = axes

        ax1_.set_title("Xray Elution Decomposition", fontsize=16)
        ax1_.set_xlabel("Eno")
        ax1_.set_ylabel("Intensity")

        nc = len(rg_params)
        x = lrf_info.x
        y = lrf_info.y
        ax1_.plot(x, y, color="orange")
        xr_ty = np.zeros(len(x))
        xr_cy_list = lrf_info.get_xr_cy_list()
        for k, cy in enumerate(xr_cy_list):
            if k == nc:
                color = "red"
                label = "baseline"
            else:
                color = None
                label = "component-%d" % (k+1)
            if k < nc or self.show_baseline:
                ax1_.plot(x, cy, ":", color=color, label=label)

        if self.separate_eoii:
            cy = Cxr[-1,:]
            ax1_.plot(x, cy, ":", color="pink", label="effect of ii")

        ax1_.legend()

        axt = ax1_.twinx()
        self.all_axes.append(axt)
        axt.grid(False)

        xr_ty = lrf_info.xr_ty
        rg_curve = self.optimizer.rg_curve
        plot_rg_curves(axt, heights, rg_params, x, xr_cy_list, xr_ty, rg_curve)

        ax2_.set_title("Xray Component Scattering (Guinier Plot)", fontsize=16)
        ax2_.set_xlabel("$Q^2$")
        ax2_.set_ylabel(r"$L_n(I) - L_n(I_0)$")

        ax3_.set_title("Xray Component Scattering (Kratky Plot)", fontsize=16)
        ax3_.set_xlabel("$QR_g$")
        ax3_.set_ylabel(r"$(QR_g)^2 \times I(Q)/I(0)$")

        rg_ = np.average(rg_params)

        gslice = gk_info.gslice
        qv2 = gk_info.qv2
        qv2_ = gk_info.qv2_
        glny_s = gk_info.glny_s
        qrgs = gk_info.qrgs
        qrgnys = gk_info.qrgnys
        # masses = gk_info.get_molecular_masses()

        for k, cy in enumerate(Pxr.T[0:nc+1]):
            if k == nc:
                color = "red"
                label_g = "baseline"
                label_k = label_g
            else:
                color = None
                label_g = "component-%d, $R_g=%.1f$" % (k+1, gk_info.rgs[k])    # note that gk_info.rgs[k] != gk_info.rg_params[k]
                label_k = "component-%d" % (k+1)

            if k < nc or self.show_baseline:
                glny_ = glny_s[k]
                ax2_.plot(qv2_, glny_, ":", color=color, label=label_g)
                qrg = qrgs[k]
                qrgny = qrgnys[k]
                ax3_.plot(qrg, qrgny, 'o', markersize=1, color=color, label=label_k)

        xmin, xmax = ax3_.get_xlim()
        ymin, ymax = ax3_.get_ylim()
        ax3_.set_xlim(xmin, xmax)
        ax3_.set_ylim(ymin, ymax)
        px = np.sqrt(3)
        py = 3/np.e
        ax3_.plot([px, px], [ymin, ymax], ":", color="gray")
        ax3_.plot([xmin, xmax], [py, py], ":", color="gray")

        dy = (ymax - ymin)*0.01
        ax3_.text(px, ymin+dy, r"$ \sqrt{3} $", ha="right")
        ax3_.text(xmax, py+2*dy, r"$ 3/e $", ha="right")

        ax3_.plot([xmin, xmax], [0, 0], color="red", alpha=0.5)

        if PLOT_EOII_IN_GUINIER:
            if self.separate_eoii:
                cy = Pxr[:,-1]
                ax2_.plot(qv2_, np.log(cy[gslice]), ":", color="pink", label="effect of ii")

        ax2_.legend()
        # ax3_.legend(bbox_to_anchor=(0.5, 1), loc="upper center", borderaxespad=2)
        ax3_.legend(loc="upper center")

        axes = []
        for i in range(4):
            ax = fig.add_subplot(gs[1,i*3:(i+1)*3], projection="3d")
            axes.append(ax)
            self.all_axes.append(ax)

        ax0, ax1, ax2, ax3 = axes

        in_folder = get_in_folder()
        fig.suptitle("LRF into Components and a Baseline from %s" % in_folder, fontsize=24)

        ax0.set_title(r"$M = P \cdot C + Residual$")

        D = self.optimizer.xrD
        qv = self.optimizer.qvector
        simple_plot_3d(ax0, D, x=qv)

        zlim = ax0.get_zlim()

        ax1.set_title(r"$p_{0} \cdot c_{0}$".format(ip+1), fontsize=16)
        D1  = Pxr[:,[ip]] @ Cxr[[ip],:]
        simple_plot_3d(ax1, D1, x=qv)
        ax1.set_zlim(zlim)

        ax2.set_title(r"$p_{0} \cdot c_{0}$".format(iq+1), fontsize=16)
        D2  = Pxr[:,[iq]] @ Cxr[[iq],:]
        simple_plot_3d(ax2, D2, x=qv)
        ax2.set_zlim(zlim)

        ax3.set_title(r"$p_b \cdot c_b$", fontsize=16)
        D3  = Pxr[:,[ib]] @ Cxr[[ib],:]
        simple_plot_3d(ax3, D3, x=qv)
        ax3.set_zlim(zlim)

        fig.tight_layout()
        # fig.subplots_adjust(top=0.9, hspace=0.8, bottom=0.04)

    def clear_axes(self):
        for ax in self.all_axes:
            ax.remove()
        self.all_axes = []

    def toggle_baseline(self):
        self.show_baseline = 1 - self.show_baseline
        self.draw()
        self.mpl_canvas.draw()
        text = self.get_toggle_button_text()
        self.toggle_btn.config(text=text)
        self.update()

    def show_tutorial_figure(self, debug=True):
        if debug:
            from importlib import reload
            import Optimizer.SvdTutorial
            reload(Optimizer.SvdTutorial)
        from molass_legacy.Optimizer.SvdTutorial import SvdTutorial

        dialog = SvdTutorial(self.parent, self)
        dialog.show()

    def show_reducibility(self):
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        from molass_legacy.Optimizer.ElutionComposer import make_composites_from_deviations
        from molass_legacy.Optimizer.CompositeInfo import convert_for_display

        ratios = self.gk_info.compute_adjacent_deviation_ratios()
        composites = make_composites_from_deviations(ratios)

        precision_save = np.get_printoptions()["precision"]
        np.set_printoptions(precision=2)
        ratios_str = str(ratios)
        np.set_printoptions(precision=precision_save)
        composites_str = str(convert_for_display(composites))

        MessageBox.showinfo( "Reducibility Info",
            "Adjacent Deviation Ratios: %s\n"
            "Possible Reduction: %s"
            % (ratios_str, composites_str),
            parent=self)
