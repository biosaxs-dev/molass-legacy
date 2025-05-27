"""
    RangeInspector.py

    Copyright (c) 2020-2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from molass_legacy.GuinierAnalyzer.AutorgKekAdapter import AutorgKekAdapter
from molass_legacy.Env.EnvInfo import get_global_env_info
from molass_legacy.KekLib.SaferSpinbox import SaferSpinbox 
from molass_legacy.KekLib.ExceptionTracebacker import log_exception

TITLE_FONTSIZE = 16
AVERAGE_WIDTH = 10
NSRD_INDEX = 2
SCALE_TYPE_PARAMS = [("log", 0), ("linear", 1)]
LINE_TYPE_NAMES = ["peaktop", "init", "kept", "almerge"]
LINE_TYPE_IVARS = [1, 1, 1, 0]
ALMERGE_ROW = -1

class RangeInspectorDialog(Dialog):
    def __init__(self, parent, solver, j0, paired_range, selected, ad, conc_depend=None, init_result=None, decomp_info=None, fx=None):
        assert conc_depend is not None

        self.solver = solver
        self.q = solver.qv
        self.zero_diff = np.zeros(len(self.q))
        self.data = solver.data
        self.error = solver.error
        self.x_curve = solver.ecurve
        self.mc_vector = solver.mc_vector

        enf_info = get_global_env_info()
        self.atsas_ok = enf_info.atsas_is_available
        self.almerge_executer = None
        self.almerge_y = None

        self.show_diff_mode = Tk.IntVar()
        self.parent = parent
        self.j0 = j0
        self.paired_range = paired_range
        self.selected = selected    # selected peakset_info
        self.ad = ad
        self.conc_depend = conc_depend
        self.scale_type = Tk.IntVar()
        self.compute_peak_top_scattering()
        self.create_line_variables()

        f, t = self.get_initial_range()
        if init_result is None:
            A, E = self.solve_this_range(f, t+1)
            # self.C has been set in the above call
        else:
            A, B, Z, E_, _, C = init_result
            E = E_[0]
            self.C = C
            self.compute_almerge_scattering(f, t+1)
        self.A_init = A
        self.E_init = E
        self.nSRD_init = self.compute_nSRD_info(A, E, f, t+1)
        self.max_nsrd = self.nSRD_init[NSRD_INDEX]*10
        self.Rg_init = self.compute_rg_info(A, E)
        if decomp_info is None:
            self.scaled_recs = None
            self.fx = None
        else:
            conc_factor = solver.conc_factor
            self.scaled_recs = decomp_info.get_scaled_recs(conc_factor)
            self.fx = decomp_info.fx
        self.create_range_variables()
        self.low_quality_warned = False
        self.applied = False
        Dialog.__init__( self, parent, "Range Inspector", visible=False)

    def compute_peak_top_scattering(self):
        top_ = self.paired_range.top_x
        hw = AVERAGE_WIDTH//2
        y = np.average(self.data[:,top_-hw:top_+hw], axis=1)
        self.top_c = self.mc_vector[top_]
        self.pty = y_  = y / self.top_c
        error = np.average(self.error[:,top_-hw:top_+hw], axis=1)
        self.pt_sn = self.pty/error
        Y = y_
        wider_range_start = int(len(y)*0.7)
        Yw = Y[wider_range_start:]
        Yw_ = Yw[np.isfinite(Yw)]
        self.min_log_y = np.average(Yw_) - np.std(Yw_)
        self.peak_top_x = top_ + self.j0
        self.peak_top_y = Y
        self.j_min = self.j0
        self.j_max = self.j0 + len(y) - 1

    def get_initial_range(self):
        fromto_list = self.paired_range.get_fromto_list()
        fv, tv = [ self.j0 + j  for j in fromto_list[self.ad]]
        return fv, tv

    def create_range_variables(self):
        fv, tv = self.get_initial_range()
        print('paired_range=', self.paired_range, (fv, tv))

        self.init_interval = (fv, tv)
        self.show_interval = [fv, tv]   # using a list to allow asignment
        f = Tk.IntVar()
        f.set(fv)
        t = Tk.IntVar()
        t.set(tv)
        self.range_vars = [f, t]

    def create_line_variables(self):
        self.line_vars = []
        for val in LINE_TYPE_IVARS:
            ivar = Tk.IntVar()
            ivar.set(val)
            self.line_vars.append(ivar)

    def show(self):
        self._show()
        return self.applied

    def body(self, body_frame):
        canvas_frame = Tk.Frame(body_frame)
        canvas_frame.pack()
        panel_frame = Tk.Frame(body_frame)
        panel_frame.pack(fill=Tk.X)
        tframe = Tk.Frame(panel_frame)
        tframe.pack(side=Tk.LEFT)
        dframe = Tk.Frame(panel_frame)
        dframe.pack(side=Tk.RIGHT)
        pframe = Tk.Frame(panel_frame)
        pframe.pack(side=Tk.RIGHT)

        self.build_canvas(canvas_frame, tframe)
        self.build_panel(pframe)
        self.builf_distinction(dframe)

    def build_canvas(self, canvas_frame, tool_frame):
        figsize = (23,7)
        fig = plt.figure(figsize=figsize)
        nrows, ncols = 1, 3
        height_ratios = [5, 1]
        gs = GridSpec( 2, ncols, height_ratios=height_ratios )
        self.height_ratio = height_ratios[1]/height_ratios[0]

        ax1 = fig.add_subplot(gs[:,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax2.get_xaxis().set_visible(False)
        ax2d = fig.add_subplot(gs[1,1])
        ax3 = fig.add_subplot(gs[:,2])
        self.ax2d = ax2d
        self.ax3t = ax3.twinx()
        self.axes = [ax1, ax2, ax2d, ax3, self.ax3t]

        self.mpl_canvas = FigureCanvasTkAgg(fig, canvas_frame)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tool_frame)
        self.toolbar.update()
        # self.mpl_canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)

        in_folder = get_in_folder()
        fig.suptitle("Range Dependency Inspection for " + in_folder, fontsize=20)

        axt = ax1.twinx()
        axt.grid(False)
        self.draw_elution(ax1, axt)
        self.draw_scattering(ax2)

        fig.tight_layout()
        fig.subplots_adjust(top=0.88, wspace=0.3, left=0.05, right=0.95)

        self.mpl_canvas.draw()
        self.fig = fig
        self.axes = [ax1, ax2, ax3]
        self.place_scale_buttons(canvas_frame)
        self.place_line_buttons(canvas_frame)
        self.place_diff_toggle_button(canvas_frame)

    def place_scale_buttons(self, frame):
        button_frame = Tk.Frame(frame)
        button_frame.place(relx=0.63, rely=0.13)
        for name, value in SCALE_TYPE_PARAMS:
            rb = Tk.Radiobutton(button_frame, text=name, variable=self.scale_type, value=value)
            rb.pack(anchor=Tk.W)
        self.scale_type.trace("w", self.scale_type_tracer)

    def scale_type_tracer(self, *args):
        self.update_yscale(self.axes[1])
        self.mpl_canvas.draw()

    def update_yscale(self, ax):
        ax.set_yscale("linear" if self.scale_type.get() else "log")

    def place_line_buttons(self, frame):
        button_frame = Tk.Frame(frame)
        button_frame.place(relx=0.63, rely=0.3)
        for name, ivar in zip(LINE_TYPE_NAMES, self.line_vars):
            if name == "almerge":
                state = Tk.NORMAL if self.atsas_ok else Tk.DISABLED
            else:
                state = Tk.NORMAL
            cb = Tk.Checkbutton(button_frame, text=name, variable=ivar, state=state)
            cb.pack(anchor=Tk.W)
            ivar.trace("w", self.line_buttons_tracer)

    def line_buttons_tracer(self, *args):
        if self.line_vars[ALMERGE_ROW].get() == 1:
            self.update_almerge_line()

        for ivar, lines in zip(self.line_vars, self.line_table):
            for line in lines:
                if line is not None:
                    line.set_visible(ivar.get() == 1)
        self.mpl_canvas.draw()

    def place_diff_toggle_button(self, frame):
        switch_frame = Tk.Frame(frame)
        switch_frame.place(relx=0.63, rely=0.8)
        for k, t in enumerate(["PT-diff", "SR-diff"]):
            rb = Tk.Radiobutton(switch_frame, text=t, variable=self.show_diff_mode, value=k)
            rb.pack()
        self.show_diff_mode.trace("w", self.show_diff_mode_tracer)

    def show_diff_mode_tracer(self, *args):
        self.range_ends_tracer(self.ad)

    def draw_elution(self, ax, axt):
        ax.set_title("Elution Curves", fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Elution №")
        ax.set_ylabel("Concentration (Absorbance x Conc. Factor)")
        axt.set_ylabel("Scattering Intensity")
        curve = self.x_curve
        x = self.j0 + curve.x
        fx = self.fx
        y = curve.y
        ax.plot(x, self.mc_vector, color='blue', alpha=0.5, label='measured UV')
        axt.plot(x, y, color='orange', alpha=0.5, label='measured Xray')

        if self.scaled_recs is not None:
            for k, rec in enumerate(self.scaled_recs):
                func = rec[1]
                y_ = func(fx)
                ax.plot(x, y_, ':', label='Scaled Xray component-%d' % k)

        ax.legend(bbox_to_anchor=(0, 1.0), loc='upper left')
        axt.legend(loc='upper right')

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)

        fv, tv = [var.get() for var in self.range_vars]
        self.rect_patch = Rectangle(
                (fv, ymin),  # (x,y)
                tv - fv,   # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        ax.add_patch(self.rect_patch)

        lines = []
        for x in [fv, tv]:
            line, = ax.plot( [x, x], [ymin, ymax], ':', color='gray', alpha=0.5 )
            lines.append(line)
        self.lines = lines
        self.elution_ylim = (ymin, ymax)
        self.last_range = (fv, tv)

    def draw_scattering(self, ax):
        self.init_control_vars()

        ax.cla()
        self.update_yscale(ax)
        ax.set_title("Scattering Curves", fontsize=TITLE_FONTSIZE)
        ax.set_ylabel("Intensity")
        self.scatter_x = self.q

        line, = ax.plot(self.scatter_x, self.peak_top_y, alpha=0.5, label='raw data average near peak top at %d' % self.peak_top_x)
        self.line_table.append([line])

        fv, tv = self.get_initial_range()
        line, = ax.plot(self.scatter_x, self.A_init, alpha=0.5, label='extrapolated with initial range (%d, %d)' % (fv, tv))
        self.line_table.append([line])

        self.line_table.append(self.scattering_curves)  # for kept lines
        self.line_table.append([None])  # for almerge

        _, ymax = ax.get_ylim()
        # ax.set_ylim(self.min_log_y, ymax)
        ax.legend()
        ax2d = self.ax2d
        ax2d.cla()
        ax2d.set_ylabel("diff")
        ax2d.set_xlabel("Q")
        m = self.show_diff_mode.get()
        self.diff_line, = ax2d.plot(self.q, self.nSRD_init[m], color='C1')

    def update_almerge_line(self):
        if not self.atsas_ok:
            return
        if not self.line_vars[ALMERGE_ROW].get():
            return

        if self.almerge_y is None:
            f, t = [var.get() - self.j0 for var in self.range_vars]
            self.compute_almerge_scattering(f, t+1)

        x = self.scatter_x
        y = self.almerge_y
        line = self.line_table[ALMERGE_ROW][0]
        if line is None:
            ax = self.axes[1]
            line, = ax.plot(x, y, ':', color='red', label='almerge')
            self.line_table[ALMERGE_ROW][0] = line
        else:
            line.set_data(x, y)

    def init_control_vars(self):
        self.line_table = []
        self.scattering_curve = None
        self.scattering_curves = []
        self.manip_infos = []
        self.manip_history = []
        self.last_manip = (self.Rg_init[0], self.A_init, *self.init_interval, *self.Rg_init[3:6], True)
        self.show_interval = list(self.init_interval)

    def build_panel(self, panel_frame):
        range_frame = Tk.Frame(panel_frame)
        range_frame.pack(side=Tk.LEFT, padx=10)

        spinbox_label = Tk.Label(range_frame, text="Range: ")
        spinbox_label.pack(side=Tk.LEFT)

        spinbox_frame = Tk.Frame(range_frame)
        spinbox_frame.pack(side=Tk.LEFT)

        f, t = self.range_vars
        spinbox1 = SaferSpinbox( spinbox_frame, textvariable=f,
                            from_=self.j_min, to=self.j_max, increment=1,
                            justify=Tk.CENTER, width=6 )
        spinbox1.pack(side=Tk.LEFT, padx=5, pady=5)
        spinbox1.set_tracer(lambda *args: self.range_ends_tracer(0))
        f.trace('w', spinbox1.tracer)

        spinbox2 = SaferSpinbox( spinbox_frame, textvariable=t,
                            from_=self.j_min, to=self.j_max, increment=1,
                            justify=Tk.CENTER, width=6 )
        spinbox2.pack(side=Tk.LEFT, padx=5, pady=5)
        spinbox2.set_tracer(lambda *args: self.range_ends_tracer(1))
        t.trace('w', spinbox2.tracer)

        self.range_ends_tracing  = True

        length_frame = Tk.Frame(range_frame)
        length_frame.pack(side=Tk.LEFT, padx=5)

        self.fixed_length = Tk.IntVar()
        self.fixed_length.set(0)
        self.fixed_length_cb = Tk.Checkbutton(length_frame, variable=self.fixed_length, text="Fixed Length:")
        self.fixed_length_cb.pack(side=Tk.LEFT)

        fv, tv = [w.get() for w in self.range_vars]
        self.range_length = Tk.IntVar()
        self.range_length.set(tv - fv + 1)
        length_label = Tk.Label(length_frame, textvariable=self.range_length, width=3)
        length_label.pack(side=Tk.LEFT)

        self.keep_btn = Tk.Button(range_frame, text="Keep", command=self.keep_scattering_curve, state=Tk.DISABLED)
        self.keep_btn.pack(side=Tk.LEFT, padx=5)

        self.clear_btn = Tk.Button(range_frame, text="Clear", command=self.clear_scattering_curves, state=Tk.DISABLED)
        self.clear_btn.pack(side=Tk.LEFT, padx=5)

    def builf_distinction(self, dframe):
        self.plot_distinction(legend=True)
        self.update()
        canvas_width = int(self.mpl_canvas_widget.cget( 'width' ))//3
        space_width = canvas_width//2
        space = Tk.Frame(dframe, width=space_width)
        space.pack(side=Tk.LEFT)
        auto_btn = Tk.Button(dframe, text="Guinier/Kratky Plots", command=self.show_guinier_kratky_plots)
        auto_btn.pack(side=Tk.LEFT)
        space = Tk.Frame(dframe, width=space_width)
        space.pack(side=Tk.LEFT)

    def plot_distinction(self, legend=False):
        ax = self.axes[2]
        ax.set_title("Supporting Information (nSRD, Rg)", fontsize=TITLE_FONTSIZE)
        ax.set_ylabel("nSRD")
        ax.set_xlabel("Elution №")
        x = self.init_interval[self.ad]
        ax.plot(x, self.nSRD_init[NSRD_INDEX], 'o', color='C1')
        ax3t = self.ax3t
        ax3t.set_ylabel("Rg")
        ax3t.grid(False)
        self.plot_rg_errorbar(ax3t, x, self.Rg_init, color='C1', label='initial Rg (%d,%d)' % self.init_interval)
        if legend:
            self.layout_distinction()

    def plot_rg_errorbar(self, ax, x, Rg_rec, color=None, label=None):
        rg, rg_error, basic_quality = Rg_rec[0:3]
        if basic_quality < 0.5:
            if not self.low_quality_warned:
                self.after(2000, self.show_quality_warning)
            ax.plot(x, rg, 'o', color='red', alpha=0.5, markersize=20)
        ax.errorbar(x, rg, rg_error, fmt='o', markersize=5, capsize=3, elinewidth=3, mfc='black', color=color, label=label)

    def show_quality_warning(self):
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        self.update()
        if self.conc_depend == 1:
            rank_mismatch_message = ".\n"
        else:
            rank_mismatch_message = (
                ', and that "low quality"\nmay have been caused by rank mismatch,\n'
                "in that case, consider retrying with Conc. Dependency=1."
                )
        MessageBox.showwarning("Guinier Analysis Quality Warning",
            "Be aware that low quality Rg values are marked\n"
            "with a red circle"
            + rank_mismatch_message,
            parent=self,
            )
        self.low_quality_warned = True

    def layout_distinction(self):
        ax3 = self.axes[2]
        ax3.set_ylim(-self.max_nsrd*0.05, self.max_nsrd)
        loc = 'lower right' if self.ad == 0 else 'lower left'
        ax3.legend(loc=loc)

        ax3t = self.ax3t
        ax3t.set_xlim(self.show_interval[0]-2, self.show_interval[1]+2)
        ax3t.legend()

    def range_ends_tracer(self, i):
        self.update()
        if not self.range_ends_tracing:
            return

        fixed_length = self.fixed_length.get()
        # print([i], 'fixed_length=', fixed_length)
        error = False
        try:
            self.range_ends_tracing = False
            fv, tv = [var.get() for var in self.range_vars]
            if fv < tv:
                if fixed_length:
                    range_length = self.range_length.get()
                    i_ = 1 - i
                    if i_ == 0:
                        self.range_vars[i_].set(tv + 1 - range_length)
                    else:
                        self.range_vars[i_].set(fv + range_length - 1)
                    self.update()
                else:
                    self.range_length.set(tv - fv + 1)
            else:
                for var, val in zip(self.range_vars, self.last_range):
                    var.set(val)
        except:
            # as in cases where the values are cleared
            error = True

        self.update()

        if error:
            self.range_ends_tracing = True
            return

        try:
            self.update_drawing(fv, tv)

            if len(self.scattering_curves) == 1:
                for w in [self.keep_btn, self.clear_btn]:
                    w.config(state=Tk.NORMAL)

            self.last_range = (fv, tv)
            self.range_ends_tracing = True
        except:
            log_exception(None, "error in range_ends_tracer:")

    def update_drawing(self, fv, tv):
        patch = self.rect_patch
        x, y = patch.xy
        patch.set_xy((fv, y))
        patch.set_width(tv - fv)

        ax1, ax2, ax3 = self.axes
        ymin, ymax = ax1.get_ylim()

        for line, x_ in zip(self.lines, [fv, tv]):
            line.set_data([x_, x_], [ymin, ymax])

        A, E = self.solve_this_range(fv, tv+1)
        scatter_y = A

        label='extrapolated with new range (%d, %d)' % (fv, tv)
        if self.scattering_curve is None:
            line, = ax2.plot(self.scatter_x, scatter_y, label=label)
            self.scattering_curve = line
            self.scattering_curves.append(line)
            self.manip_infos.append(None)
        else:
            self.scattering_curve.set_data(self.scatter_x, scatter_y)
            self.scattering_curve.set_label(label)

        self.update_current_manip_info(A, E)

        ax2.legend()
        self.mpl_canvas.draw()

    def solve_this_range(self, f, t, surplus_rank=0):
        start, stop = [j - self.j0 for j in [f, t]]
        A, B, Z, E, _, C = self.solver.extrapolate_wiser(start, stop, self.selected,
                                conc_depend = self.conc_depend,
                                surplus_rank=surplus_rank)
        self.C = C
        self.compute_almerge_scattering(start, stop)
        return A, E[0]

    def compute_nSRD_info(self, A, E, f, t):
        AE = A/E
        A_, E_ = self.solve_this_range(f, t, surplus_rank=1)
        sr_diff = A_/E_ - AE
        pt_diff = AE - self.pt_sn
        return pt_diff, sr_diff, np.sqrt(np.average(sr_diff**2))

    def update_diff_info(self, diff):
        self.diff_line.set_data(self.q, diff)
        k = len(self.scattering_curves)
        self.diff_line.set_color('C%d' % (k+1))
        self.ax2d.set_ylim(diff.min(), diff.max())

    def compute_rg_info(self, A, E):
        A_data = np.vstack( [self.q, A, E] ).T
        autorg_kek = AutorgKekAdapter( A_data )
        result = autorg_kek.run()
        if result.Rg is None:
            ret = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        else:
            ret = (result.Rg, result.Rg_stdev, result.basic_quality, result.From, result.To, result.I0)
        return ret

    def update_current_manip_info(self, A=None, E=None):
        ax3 = self.axes[2]
        for ax in [ax3, self.ax3t]:
            ax.cla()

        self.plot_distinction()
        ax = self.ax3t

        if A is not None:
            fv, tv = [ self.range_vars[k].get() for k in range(2) ]
            self.show_interval[0] = min(fv, self.show_interval[0])
            self.show_interval[1] = max(tv, self.show_interval[1])
            Rg, Rg_stdev, basic_quality, gf, gt, I0 = self.compute_rg_info(A, E)
            nSRD = self.compute_nSRD_info(A, E, fv, tv+1)
            m = self.show_diff_mode.get()
            self.update_diff_info(nSRD[m])
            nsrd_ = nSRD[NSRD_INDEX]
            self.manip_infos[-1] = ((fv, tv), (Rg, Rg_stdev, basic_quality, nsrd_))
            if len(self.manip_history) == 0:
                self.manip_history.append((self.init_interval[self.ad], self.Rg_init[0], self.nSRD_init[NSRD_INDEX]))
            pos = fv if self.ad == 0 else tv
            self.manip_history.append((pos, Rg, nsrd_))
            self.last_manip = (Rg, A, fv, tv, gf, gt, I0, False)

            manip_array = np.array(self.manip_history)
            ax.plot(manip_array[:,0], manip_array[:,1], ':', color='red', alpha=0.5, label='Rg history')
            ax3.plot(manip_array[:,0], manip_array[:,2], ':', color='blue', alpha=0.5, label='nSRD history')

            for k, (interval, manip_info) in enumerate(self.manip_infos, start=2):
                x = interval[self.ad]
                color = 'C%d' % k
                ax3.plot(x, manip_info[3], 'o', color=color)
                self.plot_rg_errorbar(ax, x, manip_info[0:3], color=color, label='Rg (%d,%d)' % interval)

        self.layout_distinction()

    def keep_scattering_curve(self):
        self.scattering_curve = None

    def clear_scattering_curves(self):
        self.draw_scattering(self.axes[1])
        self.update_current_manip_info()
        self.mpl_canvas.draw()

    def validate(self):
        if self.scattering_curve is None:
            ret = True
        else:
            import molass_legacy.KekLib.CustomMessageBox as MessageBox
            interval, manip_info = self.manip_infos[-1]
            assert interval == self.last_range
            Rg, Rg_stdev, basic_quality = manip_info[0:3]
            if basic_quality < 0.5:
                low_quality_warning = "\nAnd Rg=%.1f has been estimated with a low basic_quality=%.3g.\n\n" % (Rg, basic_quality)
            else:
                low_quality_warning = ""
            ret = MessageBox.askyesno( "Change Confirmation",
                'Pressing "OK" here implies\n'
                + 'to apply the current changed range (%d, %d).\n' % self.last_range
                + low_quality_warning
                + "Are you sure to change it as stated?", parent=self.parent )
        return ret

    def apply(self):
        if self.scattering_curve is None:
            return

        print("apply")
        self.applied = True

    def get_range(self, shifted=True):
        if shifted:
            return self.last_range
        else:
            return [j - self.j0 for j in self.last_range]

    def auto_inspection(self):
        from .AutoRangeInspector import AutoRangeInspector
        ari = AutoRangeInspector(self, self)
        ari.inspect()

    def show_guinier_kratky_plots(self, debug=False):
        if debug:
            from importlib import reload
            import Kratky.GuinierKratkyPlots
            reload(Kratky.GuinierKratkyPlots)
        from Kratky.GuinierKratkyPlots import GuinierKratkyPlots
        rg, y, f, t, gf, gt, I0, initial = self.last_manip
        if rg is None:
            pass
        else:
            color = "C1" if initial else "C2"
            gp = GuinierKratkyPlots(self.parent, self.q, y, rg, I0, (f, t), (gf, gt), color=color)
            gp.show()

    def compute_almerge_scattering(self, start, stop):
        if not self.atsas_ok:
            return
        if not self.line_vars[ALMERGE_ROW].get():
            return

        if self.almerge_executer is None:
            from molass_legacy.ATSAS.Almerge import AlmergeExecutor
            self.almerge_executer = AlmergeExecutor()

        slice_ = slice(start, stop)
        M = self.data[:,slice_]
        E = self.error[:,slice_]
        c_vector = np.sum(self.C, axis=0)
        result = self.almerge_executer.execute_matrix(self.q, M, E, c_vector)
        print('result.exz_array.shape', result.exz_array.shape)
        self.almerge_y = result.exz_array[:,1]/self.top_c
        if False:
            import molass_legacy.KekLib.DebugPlot as dplt
            dplt.push()
            fig, ax = dplt.subplots()
            ax.set_yscale("log")
            ax.plot(self.q, self.almerge_y)
            fig.tight_layout()
            dplt.show()
            dplt.pop()

    def on_mouse_motion(self, event):
        # working on this
        print("on_mouse_motion", event.inaxes, self.axes[2], event.xdata, event.ydata)
        if event.inaxes == self.axes[2]:
            print("on_mouse_motion", event.xdata, event.ydata)
