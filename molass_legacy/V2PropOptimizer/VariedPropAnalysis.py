"""
    V2PropOptimizer.VariedPropAnalysis.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import molass_legacy.KekLib.DebugPlot as dplt
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar, get_color
from ScrolledFrame import ScrolledFrame
from molass_legacy._MOLASS.SerialSettings import get_setting
from DataUtils import get_in_folder
from molass_legacy._MOLASS.Version import is_developing_version

VARYING_INUIT_PARAMS = False

class VariedPropAnalysis(Dialog):
    def __init__(self, parent, js_canvas, modelname, num_variations=20):
        self.parent = parent
        self.js_canvas = js_canvas
        self.modelname = modelname
        self.num_variations = num_variations
        Dialog.__init__(self, parent, "Varied Proportion Analysis", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        self.scrolled_frame = ScrolledFrame(body_frame)
        self.scrolled_frame.pack(anchor=Tk.N)
        cframe = self.scrolled_frame.interior

        self.ncols = 5
        self.nrows = self.num_variations//self.ncols
        height = 2.5 * self.nrows
        fig, axes = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=(20,height))
        self.draw_varied_proportions(fig, axes)
        fig.tight_layout()

        self.fig = fig
        self.axes = axes
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.popup_menu = None
        self.mpl_canvas.mpl_connect('button_press_event', self.on_figure_click)

    def buttonbox(self):
        # task: pack the close button so that it won't hide
        box = Tk.Frame(self)
        box.pack(fill=Tk.X)

        tframe = Tk.Frame(box)
        tframe.pack(side=Tk.LEFT, padx=20, pady=10)
        bframe = Tk.Frame(box)
        bframe.pack(side=Tk.RIGHT, padx=20, pady=10)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        w = Tk.Button(bframe, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=20)

        w = Tk.Button(bframe, text="RDR Chart", width=10, command=self.show_rdr_chart)
        w.pack(side=Tk.LEFT, padx=20)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        self.set_geometry()

    def set_geometry(self):
        self.update()
        canvas_width = int(self.mpl_canvas_widget.cget( 'width' ))
        canvas_height = int(self.mpl_canvas_widget.cget( 'height' ))
        print("canvas_height=", canvas_height)
        height = min(900, canvas_height+200)
        margin_width = 40
        wxh = '%dx%d' % (canvas_width + margin_width, height)
        geometry = self.geometry()
        new_geometry = re.sub( r'(\d+x\d+)(.+)', lambda m: wxh + m.group(2), geometry)
        self.geometry(new_geometry)

    def draw_varied_proportions(self, fig, axes, devel=True):
        if devel:
            from importlib import reload
            import V2PropOptimizer.PropOptimizer
            reload(V2PropOptimizer.PropOptimizer)
            import V2PropOptimizer.PropOptimizerUtils
            reload(V2PropOptimizer.PropOptimizerUtils)
            import ElutionCurve
            reload(ElutionCurve)
        from V2PropOptimizer.PropOptimizer import compute_range_rgs
        from V2PropOptimizer.PropOptimizerUtils import PropOptimizer
        from molass_legacy._MOLASS.SerialSettings import set_setting
        from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve

        self.optimizer = optimizer = self.js_canvas.fullopt
        xr_curve = optimizer.xr_curve
        x = xr_curve.x
        y = xr_curve.y

        self.prop_optimizer = prop_optimizer = PropOptimizer(self.modelname, x, y)
        init_peaks = prop_optimizer.get_init_params()
        props = prop_optimizer.compute_props(init_peaks)
        print("props=", props)

        self.pv = np.linspace(props[0], 0.2, self.num_variations)

        fv_list = []
        rdr_list = []
        peaks_list = []

        # set_setting('local_debug', True)
        # ecurve = ElutionCurve(y, x=x)     # there still is a bug
        ecurve = ElutionCurve(y)
        self.paired_ranges = paired_ranges = ecurve.get_default_paired_ranges()
        # set_setting('local_debug', False)
        print("paired_ranges=", paired_ranges)

        qv = optimizer.qvector
        D = optimizer.xrD
        E = optimizer.xrE

        minRDR = None
        n = None

        fig.suptitle("Varied Proportion Avalysis of %s using %s" % (get_in_folder(), self.modelname), fontsize=20)

        for i, p in enumerate(self.pv):
            if i != 17:
                # continue
                pass

            print([i], "optimizing with p=%.3g" % p)
            j, k = divmod(i,5)
            ax = axes[j,k]
            ax.plot(x, y, color="orange")
            opt_ret = prop_optimizer.optimize((p, 1 - p), init_params=init_peaks)
            opt_peaks = opt_ret.x.reshape(init_peaks.shape)
            fv_list.append(opt_ret.fun)
            peaks_list.append(opt_peaks)

            cy_list = prop_optimizer.compute_cy_list(opt_peaks)

            if VARYING_INUIT_PARAMS:
                init_peaks = opt_peaks

            for cy in cy_list:
                ax.plot(x, cy, ":")
            ty = np.sum(cy_list, axis=0)
            ax.plot(x, ty, ":", color="red")
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            tx = xmin*0.9 + xmax*0.1
            ty = ymin*0.2 + ymax*0.8
            ax.text(tx, ty, "p=%.3g" % p, fontsize=20, alpha=0.3)

            C = np.array(cy_list)
            rgs = compute_range_rgs(qv, D, E, paired_ranges, C)

            rdr = abs(rgs[0] - rgs[1])*2/(rgs[0] + rgs[1])
            rdr_list.append(rdr)
            if minRDR is None or rdr < minRDR:
                minRDR = rdr
                n = i
            ty = ymin*0.4 + ymax*0.6
            ax.text(tx, ty, "Rgs=%.3g, %.3g" % tuple(rgs), fontsize=20, alpha=0.3)

        if n is not None:
            j, k = divmod(n,5)
            ax = axes[j,k]
            ax.patch.set_facecolor('green')
            ax.patch.set_alpha(0.1)

        self.fv = np.array(fv_list)
        self.rdr = np.array(rdr_list)
        self.peaks_list = peaks_list

    def show_fv_chart(self):
        indeces = [str(i) for i in range(self.num_variations)]
        with dplt.Dp(window_title='FV Chart', ok_only=True, ok_text="Close"):
            fig, ax = dplt.subplots()
            ax.set_title("FV Chart", fontsize=16)
            ax.bar(indeces, self.fv)
            fig.tight_layout()
            dplt.show()

    def show_rdr_chart(self, devel=True):
        if devel:
            from importlib import reload
            import V2PropOptimizer.RdrChart
            reload(V2PropOptimizer.RdrChart)
        from .RdrChart import draw_rdr_chart
        draw_rdr_chart("RDR_AD Chart with %s" % self.modelname, self.rdr)

    def on_figure_click(self, event):
        if event.button == 3:
            axes = self.axes
            for i in range(axes.shape[0]):
                for j in range(axes.shape[1]):
                    if event.inaxes == axes[i,j]:
                        self.current_index = (i,j)
                        self.current_event = event
                        self.show_popup_menu(event)
                        break

    def show_popup_menu(self, event):
        from molass_legacy.KekLib.TkUtils import split_geometry
        self.create_popup_menu(event)
        canvas = self.mpl_canvas_widget
        cx = canvas.winfo_rootx()
        cy = canvas.winfo_rooty()
        w, h, x, y = split_geometry(canvas.winfo_geometry())
        self.popup_menu.post(cx + int(event.x), cy + h - int(event.y))

    def create_popup_menu(self, event):
        if self.popup_menu is None:
            self.popup_menu = Tk.Menu(self, tearoff=0 )
            self.popup_menu.add_command(label='Observe it as V1 Preview', command=self.observe_it_as_v1_preview)
            if is_developing_version():
                self.popup_menu.add_command(label='Optimize Proportion with EGH', command=lambda: self.optimize_proportion('EGH'))
                self.popup_menu.add_command(label='Optimize Proportion with EDM', command=lambda: self.optimize_proportion('EDM'))
                self.popup_menu.add_command(label='Optimize Proportion with STC', command=lambda: self.optimize_proportion('STC'))

    def optimize_proportion(self, name, devel=True):
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        if devel:
            from importlib import reload
            import V2PropOptimizer.PropOptimizerDialogs
            reload(V2PropOptimizer.PropOptimizerDialogs)
        from V2PropOptimizer.PropOptimizerDialogs import show_optimizer_dialog

        i, j = self.current_index
        k = self.ncols*i + j
        p = self.pv[k]
        print("optimize_proportion: model=%s pv[%d]=%.3g" % (name, k, p))

        yn = MessageBox.askyesno("Proportion Optimization Confirmation",
            "Optimizing the proportion with model=%s, init_proportion=%.3g.\n"
            "Ok?" % (name, p),
            parent=self)

        if yn:
            self.after(500, self.cancel)
            show_optimizer_dialog(self.parent, name, p, self.optimizer)

    def observe_it_as_v1_preview(self, devel=True):
        if devel:
            from importlib import reload
            import V2PropOptimizer.V1PreviewAdapter
            reload(V2PropOptimizer.V1PreviewAdapter)
        from .V1PreviewAdapter import observe_it_as_v1_preview
        observe_it_as_v1_preview(self)

    def get_current_peaks(self):
        i, j = self.current_index
        k = self.ncols*i + j
        return self.peaks_list[k]
