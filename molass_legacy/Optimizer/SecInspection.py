"""
    Optimizer.SecInspection.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from SecTheory.BasicModels import robust_single_pore_pdf as monopore_pdf

class SecInspection(Dialog):
    def __init__(self, parent, js_canvas):
        self.parent = parent
        self.js_canvas = js_canvas
        dsets = js_canvas.dsets
        ecurve = dsets[0][0]
        self.x = ecurve.x
        self.y = ecurve.y

        fullopt = js_canvas.fullopt
        params = js_canvas.get_current_params()
        self.separate_params = fullopt.split_params_simple(params)
        self.xr_baseline = fullopt.xr_baseline(self.x, self.separate_params[1])
        Dialog.__init__(self, parent, "SEC Inspection", visible=False)

    def get_xr_params(self):
        return self.separate_params[0]

    def get_rg_params(self):
        return self.separate_params[2]

    def get_sec_params(self):
        if len(self.separate_params[-1]) == 2:
            sec_params = self.separate_params[-2]
        else:
            sec_params = self.separate_params[-1]
        return sec_params

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tframe = Tk.Frame(body_frame)
        tframe.pack(fill=Tk.X)

        fig = plt.figure(figsize=(12,5))
        self.fig = fig

        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        self.start_interactive()

        self.popup_menu = None
        # self.mpl_canvas.mpl_connect('button_press_event', self.on_figure_click)

    def start_interactive(self):
        fig = self.fig
        gs = GridSpec(1,4)
        ax1 = fig.add_subplot(gs[0,0:2])

        t0, rp, N, me, T, mp = self.get_sec_params()
        print("rp, t0, N, me, T, mp=", rp, t0, N, me, T, mp)

        recs = [
            ("$t_0$", -300, 300, t0),
            ("$r_p$", 30, 200, rp),
            ("N", 30, 3000, N),
            ("$m_e$", -0.1, 3.1, me),
            ("T", 0.1, 10, T),
            ("$m_p$", -0.1, 3.1, mp),
            ("$K=N \cdot T$", 100, 4000, N*T),
            ]

        x = self.x
        y = self.y

        ax1.plot(x, y, color="orange")

        xr_params = self.get_xr_params()
        rg_params = self.get_rg_params()
        rho = rg_params/rp
        rho[rho > 1] = 1

        curves = []
        for k, r in enumerate(rho):
            np_ = N * (1 - r)**me
            tp_ = T * (1 - r)**mp
            my = xr_params[k] * monopore_pdf(x - t0, np_, tp_)
            curve, = ax1.plot(x, my, ":")
            curves.append(curve)

        sliders = []

        self.synchronizing = False

        def update(i, val):
            if self.synchronizing:
                return

            t0, rp, N, me, T, mp, K = self.get_values()
            rho = rg_params/rp
            rho[rho > 1] = 1

            if i == 2:
                T = K/N
                self.after(100, lambda : self.sychrionize(N, T, K))
            elif i == 4:
                N = K/T
                self.after(100, lambda : self.sychrionize(N, T, K))
            elif i == 6:
                ratio = np.sqrt(K/(N*T))
                N = N*ratio
                T = T*ratio
                self.after(100, lambda : self.sychrionize(N, T, K))

            for k, r in enumerate(rho):
                np_ = N * (1 - r)**me
                tp_ = T * (1 - r)**mp
                my = xr_params[k] * monopore_pdf(x - t0, np_, tp_)
                curves[k].set_ydata(my)
            fig.canvas.draw_idle()

        for i, (label, valmin, valmax, valinit) in enumerate(recs):
            ax = fig.add_axes([0.6, 0.9 - 0.1*i, 0.3, 0.03])
            slider  = Slider(ax, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
            slider.on_changed(lambda val, i_=i: update(i_, val))
            sliders.append(slider)

        self.sliders = sliders

        fig.tight_layout()
        self.mpl_canvas.draw()
        self.sliders = sliders

    def get_values(self):
        return [slider.val for slider in self.sliders]

    def sychrionize(self, N, T, K):
        self.synchronizing = True
        self.sliders[2].set_val(N)
        self.sliders[4].set_val(T)
        self.sliders[6].set_val(K)
        self.after(300, self.rest_synchronizing)

    def rest_synchronizing(self):
        self.synchronizing = False

    def on_figure_click(self, event):
        if event.button == 3:
            self.show_popup_menu(event)
            return

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
            self.popup_menu.add_command(label='Draw Factor Curves', command=self.draw_factor_curves)

    def draw_factor_curves(self, debug=True):
        if debug:
            import Optimizer.FactorCurvePlot
            from importlib import reload
            reload(Optimizer.FactorCurvePlot)
        from molass_legacy.Optimizer.FactorCurvePlot import FactorCurvePlot

        print("draw_factor_curves")

        plot = FactorCurvePlot(self.parent, self)
        plot.show()
