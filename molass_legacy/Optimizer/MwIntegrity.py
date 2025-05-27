"""
    MwIntegrity.py

    Copyright (c) 2023-2025, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar, get_color
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
# from molass_legacy.DENSS.DenssUtils import fit_data_impl
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier
from .LrfExporter import LrfExporter
from .FvScoreConverter import convert_score

BASE_WEIGHT = 0.9
PLUS_WEIGHT = 0.1

def compute_mw_integrity_impl(ratios, integer_ratios, preceder):
    integrity = np.log10(distance.jensenshannon(ratios, integer_ratios))
    if integrity < preceder:
        score = preceder*BASE_WEIGHT + integrity*PLUS_WEIGHT
    else:
        score = integrity
    return score

class MwIntegrityPlot(Dialog):
    def __init__(self, parent, js_canvas):
        self.parent = parent
        self.js_canvas = js_canvas
        self.class_code = js_canvas.dialog.class_code
        self.elution_model = js_canvas.elution_model
        self.fullopt = js_canvas.fullopt
        self.params = js_canvas.get_current_params()
        self.separate_params = self.fullopt.split_params_simple(self.params)
        exporter = LrfExporter(self.fullopt, self.params, js_canvas.dsets)
        self.info = exporter.xr_result.info
        self.integer_ratios = get_setting("mw_integer_ratios")
        Dialog.__init__(self, parent, "MW Integer Ratios", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tframe = Tk.Frame(body_frame)
        tframe.pack(fill=Tk.X)

        fig, axes = plt.subplots(ncols=3, figsize=(18,5))

        self.draw_mws(fig, axes)

        fig.tight_layout()
        self.fig = fig
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()
        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def draw_mws(self, fig, axes):
        from importlib import reload
        import Optimizer.IntegerRatios as module
        reload(module)
        from .IntegerRatios import determine_integer_ratios

        M, M_, E, P, C, qv = self.info

        ax1, ax2, ax3 = axes

        X = np.linalg.pinv(M) @ P           # M or M_
        Ep = np.sqrt((E**2) @ (X**2))

        in_folder = get_in_folder()
        fig.suptitle("Molecular Weight Integer Ratios Plot on %s with func=%s" % (in_folder, self.class_code), fontsize=20)

        ax1.set_title("Component Elutions in UV", fontsize=16)

        axes_ = (ax1, None, None, None)
        self.fullopt.objective_func(self.params, plot=True, axis_info=(fig, axes_))

        ax2.semilogy()
        ax2.set_title( "Component Scattering Profiles", fontsize=16)
        ax2.set_xlabel('Q')
        ax2.set_ylabel('Intensity ($Log_{10}$)')

        mw_list = []
        for k in range(P.shape[1] - 1):
            a = P[:,k]
            e = Ep[:,k]
            c = np.max(C[k,:])

            data = np.array([qv, a, e]).T
            sg = SimpleGuinier(data)
            mw = sg.Iz
            print([k], "Iz=%g" % sg.Iz, "Iz/C=%g" % mw)
            mw_list.append(mw)

            ax2.plot(qv, a, label='component-%d' % (k+1))

        ax2.legend()

        mw_array = np.array(mw_list)
        nc = len(mw_array)
        if self.integer_ratios is None:
            self.integer_ratios = determine_integer_ratios(mw_array)
        dist = compute_mw_integrity_impl(mw_array, self.integer_ratios, -10)
        score = convert_score(dist)
        names = ["component-%d" % (k+1) for k in range(nc)]
        colors = ["C%d" % k for k in range(nc)]

        ax3.set_title("Integrity Score: %.2g compared to %s" % (score, str(self.integer_ratios)), fontsize=16)
        ax3.set_ylabel("Molecular Weight: I(0)/C")
        ax3.bar(names, mw_array, color=colors)
