"""
    SecTheory.SecSimulator.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Models.ElutionCurveModels import egh
from SaferSpinbox import SaferSpinbox
from SecTheory.MwRgFigure import get_mwrg_info

class SecSimulator(Dialog):
    def __init__(self, parent, optimizer=None, params=None):
        self.optimizer = optimizer
        self.params = params
        if optimizer is None:
            self.initilize_for_unit_test()
        else:
            self.initialize_from_optimizer(optimizer, params)

        self.lrf_info = None
        self.gk_info = None

        Dialog.__init__(self, parent, "SEC Simulator", visible=False)

    def initilize_for_unit_test(self):
        rp = 87.7
        tI = -100
        t0 = 100
        P = 1000
        m = 3.0
        self.set_slider_specs(rp, tI, t0, P, m)

        self.init_rgs = [35.7, 35.0]
        self.init_prs = [80, 20]
        self.init_tss = [1.0, 1.0]
        self.num_plates_disclosed = 48000

    def initialize_from_optimizer(self, optimizer, params):
        self.separate_params = optimizer.split_as_unified_params(params)
        xr_params = self.separate_params[0]
        rg_params = self.separate_params[2]
        seccol_params = self.separate_params[7]
        Npc, rp, tI, t0, P, m = seccol_params
        Np = round(Npc/0.3)

        self.set_slider_specs(rp, tI, t0, P, m)

        x = optimizer.xr_curve.x

        opt_lrf_info = optimizer.objective_func(params, return_lrf_info=True)
        proportions = opt_lrf_info.get_xr_proportions()

        self.init_rgs = rg_params
        self.init_prs = proportions*100
        self.init_tss = xr_params[:,3]/xr_params[:,2]
        self.num_plates_disclosed = round(Np)

    def set_slider_specs(self, rp, tI, t0, P, m):
        self.init_slider_specs = [
            ("$r_p$",           60,         350,    rp),
            ("$t_I$",     tI - 500,    tI + 300,    tI),
            ("$t_0$",     t0 - 1000,    t0 + 300,    t0),
            ("$P$",       1000, 10000,    P),
            ("$m$",       1,      6,     m),
            ]
        self.rp_init = rp
        self.excl_limit_init = get_setting('exclusion_limit')

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        xframe = Tk.Frame(body_frame, bg='white')
        xframe.place(relx=0.74, rely=0.08)
        rframe = Tk.Frame(body_frame, bg='white')
        rframe.place(relx=0.72, rely=0.58)
        bframe = Tk.Frame(body_frame)
        bframe.pack(fill=Tk.X, padx=10)
        tframe = Tk.Frame(bframe)
        tframe.pack(side=Tk.LEFT)

        self.fig = fig = plt.figure(figsize=(18,8))
        self.create_axes()
        self.build_panel(xframe, rframe)
        self.redraw(None)

        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas.draw()
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()
        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def create_axes(self):
        gs = GridSpec(7,9)
        fig = self.fig
        ax10 = fig.add_subplot(gs[0:2,0])
        ax20 = fig.add_subplot(gs[2:,0])
        ax11 = fig.add_subplot(gs[0:2,1:6])
        axt1 = ax11.twinx()
        ax21 = fig.add_subplot(gs[2:,1:6])
        axt2 = ax21.twinx()
        axpn = fig.add_subplot(gs[1,6:9])

        self.axes = [ax10, ax20, ax11, ax21, axt1, axt2, axpn]

    def build_panel(self, xframe, rframe):

        # Column Parameters
        row = 0
        label = Tk.Label(xframe, text="Column Parameters", font=('MS Sans Serif', 16), bg='white')
        label.grid(row=row, column=0, columnspan=4)

        row += 1
        label = Tk.Label(xframe, text="Exclusion Limit", bg='white')
        label.grid(row=row, column=0, pady=5, sticky=Tk.E)

        self.excl_limit = Tk.DoubleVar()
        self.excl_limit.set(self.excl_limit_init)
        entry = Tk.Entry(xframe, textvariable=self.excl_limit, justify=Tk.CENTER, width=8)
        entry.grid(row=row, column=1, pady=5)

        label = Tk.Label(xframe, text="kDa", bg='white')
        label.grid(row=row, column=2, pady=5)

        button  = Tk.Button(xframe, text="MwRg Figure", command=self.show_mwrg_figure)
        button.grid(row=row, column=3, padx=20, pady=5)

        row += 1
        label = Tk.Label(xframe, text="Theoretical N of Plates", bg='white')
        label.grid(row=row, column=0, pady=5, sticky=Tk.E)

        self.num_plates_var = Tk.IntVar()
        self.num_plates_var.set(self.num_plates_disclosed)
        entry = Tk.Entry(xframe, textvariable=self.num_plates_var, justify=Tk.CENTER, width=8)
        entry.grid(row=row, column=1, pady=5)
        label = Tk.Label(xframe, text="N/m", bg='white')
        label.grid(row=row, column=2, pady=5)

        # SEC Parameters
        axpn = self.axes[6]
        axpn.set_axis_off()
        axpn.text(0.5, 0.2, "SEC Parameters", va="center", ha="center", fontsize=16)

        fig = self.fig

        slider_axes = []
        sliders = []
        for k, (label, valmin, valmax, valinit) in enumerate(self.init_slider_specs, start=0):
            ax_ = fig.add_axes([0.73, 0.65 - 0.05*k, 0.2, 0.03])
            slider  = Slider(ax_, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
            slider.label.set_size(16)
            # slider.on_changed(lambda val, k_=k: slider_update(k_, val))
            slider_axes.append(ax_)
            sliders.append(slider)

        self.slider_axes = slider_axes
        self.sliders = sliders

        # Rg and Proportions
        """
            import matplotlib
            print(matplotlib.rcParams['font.family'])
            which gave ['sans-serif']

            import tkinter
            print(tkinter.font.families())
        """

        for col, name in enumerate(["Comp. No", "Rg", "Proportion(%)", "τ/σ"]):
            label = Tk.Label(rframe, text=name, font=('MS Sans Serif', 16), bg='white')
            label.grid(row=0, column=col, padx=5)

        rg_vars = []
        pr_vars = []
        ts_vars = []
        for k, (rg, pr, ts) in enumerate(zip(self.init_rgs, self.init_prs, self.init_tss), start=1):
            label = Tk.Label(rframe, text="%d" % k, bg='white')
            label.grid(row=k, column=0, pady=5)

            var = Tk.DoubleVar()
            var.set(rg)
            rg_vars.append(var)
            sbox = SaferSpinbox(rframe, textvariable=var,
                            from_=10.0, to=80.0, format="%.1f", increment=0.1,
                            justify=Tk.CENTER, width=6)
            sbox.grid(row=k, column=1, pady=5)

            var = Tk.DoubleVar()
            var.set(pr)
            pr_vars.append(var)
            sbox = SaferSpinbox(rframe, textvariable=var,
                            from_=10.0, to=80.0, format="%.0f", increment=1.,
                            justify=Tk.CENTER, width=6)
            sbox.grid(row=k, column=2, pady=5)

            var = Tk.DoubleVar()
            var.set(ts)
            ts_vars.append(var)
            sbox = SaferSpinbox(rframe, textvariable=var,
                            from_=0.0, to=2.0, format="%.1f", increment=0.1,
                            justify=Tk.CENTER, width=6)
            sbox.grid(row=k, column=3, pady=5)

        self.rg_vars = rg_vars
        self.pr_vars = pr_vars
        self.ts_vars = ts_vars

        self.reset_axis = fig.add_axes([0.7, 0.05, 0.07, 0.07])
        self.reset_btn = Button(self.reset_axis, 'Reset')
        self.reset_btn.on_clicked(self.reset)

        self.redraw_axis = fig.add_axes([0.8, 0.05, 0.07, 0.07])
        self.refraw_btn = Button(self.redraw_axis, 'Redraw')
        self.refraw_btn.on_clicked(self.redraw)

        self.conform_axis = fig.add_axes([0.9, 0.05, 0.07, 0.07])
        self.conform_btn = Button(self.conform_axis, 'Conform')
        self.conform_btn.on_clicked(self.conform)

    def get_values_from_sliders(self):
        values = []
        for slider in self.sliders:
            values.append(slider.val)
        return values

    def show_mwrg_figure(self, debug=True):
        if debug:
            from importlib import reload
            import SecTheory.MwRgFigure
            reload(SecTheory.MwRgFigure)
        from SecTheory.MwRgFigure import MwRgFigure
        fig = MwRgFigure(self, self.excl_limit.get(), location="upper right")
        fig.show()

    def compute_draw_params(self, devel=True):
        if devel:
            from importlib import reload
            import SecTheory.SecParamsPlot
            reload(SecTheory.SecParamsPlot)
        from SecTheory.SecParamsPlot import compute_sec_peak_params

        rp, tI, t0, P, m = self.get_values_from_sliders()
        rgs = np.array([var.get() for var in self.rg_vars])
        Npc = self.num_plates_var.get() * 0.3   # 0.3m = 30cm

        trs, sigmas, xlim, x = compute_sec_peak_params(rp, tI, t0, P, m, rgs, Npc)

        props = np.array([var.get()/100 for var in self.pr_vars])
        ts_ratios = np.array([var.get() for var in self.ts_vars])

        # sigmas**2 == sig**2 + tau**2 = sig**2 * (1  + ts_ratio**2)
        #   where ts_ratio == tau/sig, tau = sig*ts_ratio
        # sig**2 = sigmas**2/(1  + ts_ratio**2)
        # sig = sigmas * sqrt(1/(1 + ts_ratio**2))
        # tau = sig*ts_ratio
        sig = sigmas * np.sqrt(1/(1 + ts_ratios**2))
        tau = ts_ratios * sig

        n_props = props/np.sum(props)

        hs = n_props
        model_params = np.vstack([hs, trs, sig, tau]).T

        def objective(p):
            model_params[:,0] = p
            areas = []
            for h, m, s, t in model_params:
                areas.append(np.sum(egh(x, h, m, s, t)))
            n_areas = areas/np.sum(areas)
            return np.sum((n_areas - n_props)**2)

        # get vertical scales which result in area proportions
        ret = minimize(objective, hs)
        model_params[:,0] = ret.x
        print("compute_draw_params", n_props, ret.x)

        return rgs, x, xlim, model_params, sigmas


    def reset(self, event):
        self.excl_limit.set(self.excl_limit_init)
        self.num_plates_var.set(self.num_plates_disclosed)

        for slider in self.sliders:
            slider.reset()

        for k, rg_var in enumerate(self.rg_vars):
            rg_var.set(round(self.init_rgs[k], 1))

        for k, pr_var in enumerate(self.pr_vars):
            pr_var.set(round(self.init_prs[k]))

        for k, ts_var in enumerate(self.ts_vars):
            ts_var.set(round(self.init_tss[k], 2))

        self.redraw(event)
        self.update()

    def redraw(self, event):
        rgs, x, xlim, model_params, sigmas = self.compute_draw_params()

        fig = self.fig

        fig.suptitle("Simple SEC Theory Simulation", fontsize=32)

        y_list = []
        for h, mu, s, tau in model_params:
            cy = egh(x, h, mu, s, tau)
            y_list.append(cy)
        ty = np.sum(y_list, axis=0)
        y_list.insert(0,ty)

        ax10, ax20 = self.axes[0:2]
        for ax, title in (ax10, "Entire\nView"), (ax20, "Peak\nView"):
            ax.set_axis_off()
            ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=16)

        ax11, ax21 = self.axes[2:4]

        for ax in ax11, ax21:
            ax.cla()
            ax.grid(False)
            for k, y in enumerate(y_list):
                style = "-" if k == 0 else ":"
                ax.plot(x, y, style)

        ax21.set_xlim(*xlim)

        xmin, xmax = ax11.get_xlim()
        dx = (xmax - xmin)*0.05
        ymin, ymax = ax11.get_ylim()
        ax11.set_ylim(ymin, ymax)
        ty = ymin*0.7 + ymax*0.3
        label_fontsize = 14

        rp, tI, t0, P, m = self.get_values_from_sliders()
        tL = t0 + P
        k = 0
        for px, label in [(tI, "$t_I$"), (t0, "$t_0$"), (t0 + P, "$t_0 + P$")]:
            ax11.plot([px, px], [ymin, ymax], ":")
            px_ = (px - tI)/(tL - tI)
            dx_ = dx if k < 2 and px_ < 0.2 else -dx
            ax11.annotate(label, xy=(px, 0), xytext=(px + dx_, ty), ha='center',
                arrowprops=dict(arrowstyle="-", color='k'), fontsize=label_fontsize )
            k += 1

        rectangle = Rectangle(
                (xlim[0], ymin),    # (x,y)
                xlim[1] - xlim[0],  # width
                ymax - ymin,        # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        ax11.add_patch(rectangle)

        r = np.flip(np.arange(0, rp*0.9, 0.1))
        t = t0 + P*np.power(1 - r/rp, m)

        if event is not None:
            for k, ax in enumerate(self.axes[4:6]):
                ax.remove()
                self.axes[4+k] = self.axes[2+k].twinx()
        axts = self.axes[4:6]

        for k, axt in enumerate(axts):
            axt.set_ylabel("$R_g$", fontsize=label_fontsize)
            ms = 3 if k == 0 else 5
            axt.plot(t, r, "o", color="yellow", markersize=ms, label=r"$t_R = t_0 + P \cdot (1-\rho)^m$")
            axt.legend(fontsize=label_fontsize)

        axt = axts[1]
        axt.set_ylim(rgs[-1] - 2, rgs[0] + 2)

        xmin, xmax = axt.get_xlim()
        ymin, ymax = axt.get_ylim()
        dx = (xmax - xmin)*0.1
        dy = (ymax - ymin)*0.05

        trs = model_params[:,1]

        for rg, tr in zip(rgs, trs):
            rg_text = r"$R_g=%.1f$" % rg
            axt.annotate(rg_text, xy=(tr, rg), xytext=(tr + dx, rg + dy), ha='center',
                arrowprops=dict(arrowstyle="->", color='k'), fontsize=label_fontsize )

        if False:
            RgD = (rgs[0] - rgs[1])/(trs[1] - trs[0]) * np.average(sigmas)

            xmin, xmax = ax21.get_xlim()
            ymin, ymax = ax21.get_ylim()
            tx = xmin*0.8 + xmax*0.2
            ty = (ymin + ymax)/2
            ax21.text(tx, ty, "$RgD=%.3g$" % RgD, ha="center", va="center", fontsize=16)

        fig.tight_layout()
        # plt.show()

        if event is not None:
            self.mpl_canvas.draw()

    def get_lrf_info(self):
        if self.lrf_info is None:
            self.lrf_info = self.optimizer.objective_func(self.params, return_lrf_info=True)
        return self.lrf_info

    def get_gk_info(self, debug=True):
        if self.gk_info is None:
            if debug:
                from importlib import reload
                import Kratky.GuinierKratkyInfo
                reload(Kratky.GuinierKratkyInfo)
            from Kratky.GuinierKratkyInfo import GuinierKratkyInfo
            lrf_info = self.get_lrf_info()
            self.gk_info = GuinierKratkyInfo(self.optimizer, self.params, lrf_info)  # need_baseline=False
        return self.gk_info

    def conform(self, event, debug=True):
        if self.optimizer is None:
            raise Exception("Conform is not supported in Unit Test")

        gk_info = self.get_gk_info()
        print("rgs=", gk_info.rgs)
        for rg, var in zip(gk_info.rgs, self.rg_vars):
            var.set(round(rg,1))

        if debug:
            from importlib import reload
            import SecTheory.SecEstimator
            reload(SecTheory.SecEstimator)
        from SecTheory.SecEstimator import SecEstimator

        xr_curve = self.optimizer.xr_curve
        x = xr_curve.x
        y = xr_curve.y

        xr_params = self.separate_params[0]
        tI, Np_ignore = self.separate_params[8]
        Npm = self.num_plates_var.get()
        Npc = Npm*0.3

        seccol_params = self.separate_params[7]
        Npc_ignore, rp_ignore, tI, t0, P, m = seccol_params
        rp = self.sliders[0].val

        estimator = SecEstimator(Npc, rp, tI, t0, P, m)
        scaled_xr_params = self.lrf_info.get_scaled_xr_params(xr_params, debug=False)
        estimator.fit_to_decomposition(x, y, scaled_xr_params, gk_info.rgs)
        Npc, rp, tI, t0, P, m = estimator.get_params()
        print("ret: Npc, rp, tI, t0, P, m =",Npc, rp, tI, t0, P, m)
        for slider, val in zip(self.sliders, [rp, tI, t0, P, m]):
            slider.set_val(val)

        self.redraw(event)

def demo():
    from molass_legacy.KekLib.TkUtils import get_tk_root

    root = get_tk_root()
    root.withdraw()

    def demo_impl():
        dialog = SecSimulator(root)
        dialog.show()
        root.quit()

    root.after(0, demo_impl)
    root.mainloop()
