"""
    Peaks.PeakEditor.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""
import copy
import logging
import queue
from bisect import bisect_right
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.KillableThread import Thread
import molass_legacy.KekLib.DebugPlot as dplt
from molass_legacy.Batch.FullBatch import FullBatch
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.TkSupplements import BlinkingFrame
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from molass_legacy.QuickAnalysis.PeakUtils import guess_peak_params
from .ElutionModels import egh
from molass_legacy.Optimizer.FullOptInput import FullOptInput
from molass_legacy.Optimizer.FuncImporter import get_objective_function_info
from molass_legacy.Optimizer.OptConstants import MIN_NUM_PEAKS, MAX_NUM_PEAKS
from molass_legacy.Baseline.Constants import SLOPE_SCALE
from molass_legacy.Optimizer.FvScoreConverter import convert_score
from molass_legacy.Optimizer.OptimizerSettings import get_advanced_settings_text
from molass_legacy.RgProcess.RgCurve import ProgressCallback, draw_rg_bufer
from .PeProgressConstants import MAXNUM_STEPS, STOCH_INIT_STEPS, STARTED, RG_CURVE_OK, PREPARED
from molass_legacy._MOLASS.Version import is_developing_version

SKIP_RG_CURVE_CALCULATION = False
BETA_RELEASE = get_setting("beta_release")
ENABLE_SEPARATE_FOULING = get_setting("enable_separate_fouling")

if ENABLE_SEPARATE_FOULING:
    DRIFT_TYPES = ["Linear", "Linear + Uniform Fouling", "Linear + Separate Fouling"]
else:
    DRIFT_TYPES = ["Linear", "Linear + Fouling", "Disabled"]

def get_default_maxnum_trials(num_peaks):
    return max(30, num_peaks*10)

class PeakEditor(FullBatch, Dialog):
    def __init__(self, parent, sd, pre_recog, corrected_sd, treat, rg_folder=None, exact_num_peaks=None, strict_sec_penalty=False):
        FullBatch.__init__(self)
        self.logger = logging.getLogger(__name__)
        self.parent = parent

        if False:
            xr_curve1 = sd.get_xr_curve()
            xr_curve2 = corrected_sd.get_xr_curve()
            with dplt.Dp():
                fig, ax = dplt.subplots()
                ax.set_title("PeakEditor.__init__")
                ax.plot(xr_curve1.x, xr_curve1.y, label="raw data")
                ax.plot(xr_curve2.x, xr_curve2.y, label="corrected data")
                ax.legend()
                fig.tight_layout()
                dplt.show()

        self.sd = sd
        self.corrected_sd = corrected_sd
        self.pre_recog = pre_recog
        self.base_curve_info = treat.get_base_curve_info()
        uv_basemodel = get_setting("uv_basemodel")
        self.advanced = uv_basemodel == 1
        assert self.advanced
        
        self.dsets = None
        self.exact_num_peaks = exact_num_peaks
        self.strict_sec_penalty = strict_sec_penalty
        self.fullopt_class, self.class_code = None, None
        self.fullopt_input = FullOptInput(sd=sd, corrected_sd=corrected_sd, rg_folder=rg_folder)
        self.applied = False
        self.user_positions = None
        self.rg_line = None
        self.is_busy_ = True
        self.watching = False
        self.stop_watching = False
        self.title_text = None
        self.baselines = None
        self.baseline_params = None
        self.is_ready = False
        self.t0_upper_bound = None
        self.func_info = get_objective_function_info(self.logger)
        self.func_dict = self.func_info.func_dict
        self.key_list = self.func_info.key_list
        Dialog.__init__(self, parent, "Initial Estimator", visible=False)
    
    def is_busy(self):
        return self.is_busy_

    def cancel(self):
        # overiding cancel to cleanup self.fig
        # because the call to the destructor __del__ seems to be delayed
        plt.close(self.fig)
        # print("ExtrapolSolverDialog: closed fig")
        Dialog.cancel(self)

    def show( self ):
        self.update()
        if not SKIP_RG_CURVE_CALCULATION:
            self.after(500, self.start_rg_curve_thread)
        self._show()

    def body(self, body_frame):
        upper_frame = Tk.Frame(body_frame)
        upper_frame.pack(fill=Tk.X)
        lower_frame = Tk.Frame(body_frame)
        lower_frame.pack(fill=Tk.X)

        lower_left = Tk.Frame(lower_frame)
        lower_left.pack(side=Tk.LEFT, padx=10)
        lower_right = Tk.Frame(lower_frame)
        lower_right.pack(side=Tk.LEFT, padx=10)

        if is_developing_version():
            devel_frame = Tk.Frame(lower_frame)
            devel_frame.pack(side=Tk.LEFT, padx=10)
            devel_button = Tk.Button(devel_frame, text="Devel Test", command=self.devel_test)
            devel_button.pack()

        lower_left1 = Tk.Frame(lower_left)
        lower_left1.pack(fill=Tk.X)
        tool_frame = Tk.Frame(lower_left1)
        tool_frame.pack(side=Tk.LEFT)
        lower_left2 = Tk.Frame(lower_left)
        lower_left2.pack()

        fig, axes = plt.subplots(ncols=3, figsize=(18,5))
        self.fig = fig
        self.axes = axes
        ax2 = axes[1]
        self.axt = ax2.twinx()
        self.axt.grid(False)
        self.mpl_canvas = FigureCanvasTkAgg(fig, upper_frame)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        uv_x, uv_y, xr_x, xr_y, baselines = self.get_curve_xy(return_baselines=True)
        uv_y_ = uv_y - baselines[0]
        xr_y_ = xr_y - baselines[1]
        uv_peaks, xr_peaks = self.get_modeled_peaks(uv_x, uv_y_, xr_x, xr_y_)
        self.set_lrf_src_args1(uv_x, uv_y, xr_x, xr_y, baselines)
        self.xr_peaks_orig = copy.deepcopy(xr_peaks)

        self.draw_elements(uv_peaks, xr_peaks, uv_x, uv_y, xr_x, xr_y, baselines=baselines)
        self.build_lower(lower_left2, lower_right)
        self.draw_title()       # must be done after build_lower

        self.toolbar = NavigationToolbar(self.mpl_canvas, tool_frame)
        self.toolbar.update()

        self.popup_menu = None
        self.mpl_canvas.mpl_connect('button_press_event', self.on_figure_click)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack(fill=Tk.X, padx=20)

        num_buttons = 8
        for j in range(num_buttons):
            box.columnconfigure(j, weight=1)

        w = Tk.Button(box, text="◀ Cancel", width=10, command=self.user_cancel)
        col = 0
        w.grid(row=0, column=col, pady=10)
        self.cancel_btn = w

        self.buttons_to_update = []

        col += 1
        w = Tk.Button(box, text="▽ Bounds Inspection", width=18, command=self.show_bounds, state=Tk.DISABLED)
        w.grid(row=0, column=col, pady=10)
        self.bounds_btn = w
        self.buttons_to_update.append(w)

        col += 1
        w = Tk.Button(box, text="▽ Show Parameters", width=18, command=self.show_params, state=Tk.DISABLED)
        w.grid(row=0, column=col, pady=10)
        self.params_btn = w
        self.buttons_to_update.append(w)

        if is_developing_version():
            col += 1
            w = Tk.Button(box, text="▽ Redraw Scores", width=18, command=self.redraw_scores, state=Tk.DISABLED)
            w.grid(row=0, column=col, pady=10)
            self.redraw_btn = w
            self.buttons_to_update.append(w)

        col += 1
        w = Tk.Button(box, text="▽ Complementary View", width=20, command=self.show_complementary_view, state=Tk.DISABLED)
        w.grid(row=0, column=col, pady=10)
        self.opposite_btn = w
        self.buttons_to_update.append(w)

        col += 1
        w = Tk.Button(box, text="▽ Edit Canvas", width=18, command=self.show_editcanvas, state=Tk.DISABLED)
        w.grid(row=0, column=col, pady=10)
        self.editcanvas_btn = w
        self.buttons_to_update.append(w)

        col += 1
        w = Tk.Button(box, text="▽ CPD Decompose", width=18, command=self.show_cpd_decompose, state=Tk.DISABLED)
        w.grid(row=0, column=col, pady=10)
        self.editcanvas_btn = w
        self.buttons_to_update.append(w)

        col += 1
        self.optimize_btn_blink = BlinkingFrame(box)
        self.optimize_btn_blink.grid(row=0, column=col, pady=10)
        w = Tk.Button(self.optimize_btn_blink, text="▶ Optimize", width=14, command=self.ok, state=Tk.DISABLED)
        w.pack()
        self.optimize_btn_blink.objects = [w]
        self.optimize_btn = w
        self.buttons_to_update.append(w)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def user_cancel(self, ask=True):
        test_pattern = get_setting("test_pattern")
        if test_pattern is None and ask:
            import molass_legacy.KekLib.CustomMessageBox as MessageBox
            ret = MessageBox.askyesno("Cancel Confirmation",
                "Are you sure to cancel?",
                parent=self)
            if not ret:
                return

        if self.watching:
            self.stop_watching = True
        else:
            self.cancel()

    def apply(self):
        self.applied = True

    def draw_title(self):
        from molass_legacy.Optimizer.OptimizerUtils import get_model_name, get_method_name
        fullopt_class, class_code = self.get_function_class()
        text = "Elution Decomposition of %s with model=%s method=%s" % (get_in_folder(), get_model_name(class_code), get_method_name())
        if self.title_text is None:
            self.title_text = self.fig.suptitle(text, fontsize=20)
        else:
            self.title_text.set_text(text)

    def draw_elements(self, uv_peaks, xr_peaks, uv_x, uv_y, xr_x, xr_y, baselines=None, draw_canvas=True):
        if False:
            import QuickAnalysis.ModeledPeaks
            from importlib import reload
            reload(QuickAnalysis.ModeledPeaks)
        from molass_legacy.QuickAnalysis.ModeledPeaks import plot_curve

        fig = self.fig
        axes = self.axes

        for ax in axes:
            ax.cla()

        ax1, ax2, ax3 = axes
        ax1.set_title("UV Elution at $\lambda$=%.3g" % get_setting("absorbance_picking"), fontsize=16)
        ax2.set_title("Xray Elution at Q=%.3g" % get_setting("intensity_picking"), fontsize=16)
        ax3.set_title("Objective Function Scores", fontsize=16)

        if baselines is None:
            uv_baseline, xr_baseline = None, None
        else:
            uv_baseline, xr_baseline = baselines
        self.uv_ty = plot_curve(ax1, uv_x, uv_y, uv_peaks, color='blue', baseline=uv_baseline)
        plot_curve(ax2, xr_x, xr_y, xr_peaks, color='orange', baseline=xr_baseline)
        self.xr_draw_info = [xr_x, xr_y, xr_peaks, xr_baseline]

        ax3.text(0.5, 0.5, "Not Ready", color="gray", ha="center", va="center", fontsize=20)

        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        if draw_canvas:
            self.mpl_canvas.draw()

    def build_lower(self, lower_left, lower_right):

        self.pbar = ttk.Progressbar(lower_left, orient ="horizontal", length=700, mode="determinate")
        self.pbar["maximum"] = MAXNUM_STEPS
        self.pbar.pack(pady=10)

        self.status_msg = Tk.StringVar()
        label = Tk.Label(lower_left, textvariable=self.status_msg, bg='white')
        label.pack(fill=Tk.X, pady=10)
        self.status_bar = label

        self.build_panel(lower_right)    # 

    def build_panel(self, frame):
        pframe = Tk.Frame(frame)
        pframe.pack(padx=50)

        temp_frame = Tk.Frame(pframe)
        temp_frame.grid(row=0, column=0, padx=5, pady=5)

        label = Tk.Label(temp_frame, text="Number of Components: ")
        label.grid(row=0, column=0, sticky=Tk.W)

        self.user_num_peaks = Tk.IntVar()
        num_peaks=len(self.peak_params_set[1])
        self.user_num_peaks.set(num_peaks)

        sbox  = Tk.Spinbox(temp_frame, textvariable=self.user_num_peaks,
                          from_=MIN_NUM_PEAKS, to=MAX_NUM_PEAKS, increment=1,
                          justify=Tk.CENTER, width=5,
                          state=Tk.DISABLED)
        sbox.grid(row=0, column=1)

        temp_frame = Tk.Frame(pframe)
        temp_frame.grid(row=1, column=0, padx=5, pady=5)

        label = Tk.Label(temp_frame, text="Baseline: ")
        label.grid(row=0, column=0, sticky=Tk.W)

        unified_baseline_type = get_setting("unified_baseline_type")
        self.drift_type = Tk.StringVar()
        self.drift_type.set(DRIFT_TYPES[unified_baseline_type - 1])
        self.drift_type_box = ttk.Combobox(master=temp_frame,
                    values=DRIFT_TYPES, textvariable=self.drift_type, width=15, state=Tk.DISABLED)
        self.drift_type_box.grid(row=0, column=1)

        temp_frame = Tk.Frame(pframe)
        temp_frame.grid(row=0, column=1, padx=5, pady=5)

        label = Tk.Label(temp_frame, text="Number of Iterations: ")
        label.grid(row=0, column=0, sticky=Tk.W)

        self.num_iter = Tk.IntVar()
        self.num_iter.set(20)
        self.num_iter_entry = Tk.Entry(temp_frame, textvariable=self.num_iter, width=5, justify=Tk.CENTER)
        self.num_iter_entry.grid(row=0, column=1)

        temp_frame = Tk.Frame(pframe)
        temp_frame.grid(row=1, column=1, padx=5, pady=5)

        label = Tk.Label(temp_frame, text="Random Number Seed: ")
        label.grid(row=0, column=0, sticky=Tk.W)

        self.seed = Tk.IntVar()
        self.seed.set(np.random.randint(100000, 999999))
        self.seed_entry = Tk.Entry(temp_frame, textvariable=self.seed, width=8, justify=Tk.CENTER)
        self.seed_entry.grid(row=0, column=1)

        self.param_init_type = Tk.IntVar()
        self.param_init_type.set(1)     # Known best

        temp_frame = Tk.Frame(pframe)
        temp_frame.grid(row=0, column=2, padx=5, pady=5)

        row = 0
        label = Tk.Label(temp_frame, text="Max Number of Trials: ")
        label.grid(row=row, column=0, sticky=Tk.E)

        self.maxnum_trials = Tk.IntVar()
        mt_init = get_default_maxnum_trials(num_peaks)
        self.maxnum_trials.set(mt_init)
        self.mn_sbox = Tk.Spinbox(temp_frame, textvariable=self.maxnum_trials,
                          from_=1, to=100, increment=1,
                          justify=Tk.CENTER, width=5)
        self.mn_sbox.grid(row=row, column=1, sticky=Tk.W)

        label_text = get_advanced_settings_text()
        if label_text > "":
            label = Tk.Label(pframe, text=label_text)
            label.grid(row=2, column=0, columnspan=3)

    def start_rg_curve_thread(self):
        self.queue = queue.Queue()
        self.rg_curve_thread = Thread(target=self.prepare_rg_curve, args=[self.queue], name="Rg Curve Thread")
        self.rg_curve_thread.start()
        self.watching = True
        self.after(500, self.watch_rg_curve_thread)

    def prepare_rg_curve(self, queue):
        queue.put([STARTED, None])

        if self.elution_model in [0, 2]:
            rg_curve_ok = RG_CURVE_OK
        else:
            rg_curve_ok = RG_CURVE_OK - STOCH_INIT_STEPS

        progress_cb = ProgressCallback(queue, STARTED, rg_curve_ok)
        self.dsets = self.fullopt_input.get_dsets(progress_cb=progress_cb, compute_rg=True, possibly_relocated=False)
        queue.put([rg_curve_ok, None])
        queue.put([PREPARED, None])

    def watch_rg_curve_thread(self):
        try:
            progress, p_info = self.queue.get()
            self.pbar["value"] = progress

            if p_info is not None and type(p_info) == tuple:
                xr_curve = self.ecurves[1]
                drawn = draw_rg_bufer(self.axt, p_info, self, xr_curve.x)   # this updates self.rg_line
                if drawn:
                    j = p_info[1]
                    self.update_status_bar("Computing Rg values near the %d-th elution." % j)
                    self.mpl_canvas.draw()

            elif progress < 0:
                self.rg_curve_thread.join()
                self.watching = False
                progress = MAXNUM_STEPS
                if self.elution_model == 1:
                    progress -= STOCH_INIT_STEPS
                self.pbar["value"] = progress
                self.update_status_bar("Rg curve is ready.")
                self.rg_curve_thread = None
                self.get_ready_for_scores_etc()
                # self.save_rg_buffer()
                return

        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(self.logger, "watch_prepare_thread")

        if self.stop_watching:
            self.rg_curve_thread.terminate()
            self.watching = False
            self.is_busy_ = False
            self.cancel()
        else:
            self.after(100, self.watch_rg_curve_thread)      # 100 must be shorter than Rg computation

    def update_status_bar(self, msg):
        self.logger.info(msg)
        self.status_msg.set(msg)
        self.update()

    def get_ready_for_scores_etc(self):
        # for the time being until None baseline is available
        self.after(1000, self.get_ready_for_optimization)

    def update_button_states(self):
        for k, widget in enumerate(self.buttons_to_update):
            if BETA_RELEASE and k in [0]:
                pass
            else:
                widget.config(state=Tk.NORMAL)

    def save_rg_buffer(self):
        rc_curve = self.dsets[1]
        rc_curve.save_buffer("rg_buffer.dat")

    def re_construct_optimizer(self):
        self.construct_optimizer()
        init_params = self.compute_init_params()
        self.fullopt.prepare_for_optimization(init_params)

    def draw_scores(self, init_params=None, draw_rg_curve=True, create_new_optimizer=True):

        if create_new_optimizer:
            self.construct_optimizer()
        else:
            # as in on-the-fly debug from self.devel_test
            pass

        if init_params is None:
            init_params = self.compute_init_params()

        self.fullopt.prepare_for_optimization(init_params)

        for ax in self.axes:
            ax.cla()
        self.axt.remove()

        ax1, ax2 = self.axes[0:2]
        axt = ax2.twinx()
        axt.grid(False)
        self.axt = axt

        axis_info = (self.fig, (*self.axes, axt))       # i.e., draw all
        ax1.set_title("UV Decomposition", fontsize=16)
        ax2.set_title("Xray Decomposition", fontsize=16)

        try:
            fv = self.fullopt.objective_func(self.fullopt.init_params, plot=True, axis_info=axis_info)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(self.logger, "draw_scores: ", n=10)
            fv = np.inf

        ax3 = self.axes[2]
        ax3.set_title("Objective Function Scores in SV=%.3g" % convert_score(fv), fontsize=16)
        self.fig.tight_layout()
        self.mpl_canvas.draw()
        self.is_busy_ = False
        # self.optimize_btn_blink.start()

    def get_ready_for_optimization(self):
        self.draw_scores()      # includes self.construct_optimizer()
        self.update_button_states()
        self.is_ready = True

    def get_optimizer(self):
        return self.fullopt

    def get_init_params(self):
        return self.fullopt.init_params     # this includes extended params if self.fullopt supports them

    def clear_pframe(self):
        assert False

    def get_drift_type(self):
        drift_type = self.drift_type.get()
        if drift_type == DRIFT_TYPES[0]:
            return 'linear'
        elif drift_type == DRIFT_TYPES[1]:
            return 'integral'
        elif drift_type == DRIFT_TYPES[2]:
            return 'integral'
        else:
            assert False

    def on_figure_click(self, event):
        if event.button == 3 and self.is_ready:
            if event.inaxes in [self.axes[2]]:
                self.show_popup_menu(event)
                return

        if event.xdata is None:
            return

        if False:
            print("double click")
            i = None
            for k, ax in enumerate(self.axes):
                if ax == event.inaxes:
                    i = k
                    break

            print([i], "axis")
            if i == 0:
                a, b = self.peak_params_set[2:4]
                xr_x = (event.xdata - b)/a
            else:
                xr_x = event.xdata

            print("event.xdata=", event.xdata, "xr_x=", xr_x)
            self.try_insert_user_peak(xr_x)

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
            self.popup_menu.add_command(label='Show Score Details', command=self.show_score_details)

    def try_insert_user_peak(self, xr_x):
        # to be moved to EditCanvas.py

        if self.user_positions is None:
            self.user_positions = np.array([params[1]  for params in self.peak_params_set[1]])

        xr_curve = self.ecurves[1]
        x = xr_curve.x
        min_dist = np.min(np.abs(self.user_positions - xr_x))
        ratio = min_dist/len(x)
        print("min_dist=", min_dist, "ratio=", ratio)
        if ratio > 0.05:
            # self.insert_user_peak(xr_x)
            assert False
        else:
            print("user_positions=", self.user_positions)

    def get_dsets(self):
        if self.dsets is None:
            sd = self.sd
            D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
            U, wlvec, uv_curve = sd.conc_array, sd.lvector, sd.get_uv_curve()
            dsets = ((xr_curve, D), None, (uv_curve, U))
            return dsets
        else:
            return self.dsets

    def get_n_iterations(self):
        return self.num_iter.get()

    def get_seed(self):
        return self.seed.get()

    def get_param_init_type(self):
        return self.param_init_type.get()

    def get_maxnum_trials(self):
        return self.maxnum_trials.get()

    def show_bounds(self, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.Optimizer.BoundsInspection
            reload(molass_legacy.Optimizer.BoundsInspection)
        from molass_legacy.Optimizer.BoundsInspection import BoundsInspection
        dialog = BoundsInspection(self.parent, self.fullopt)
        dialog.show()

    def show_params(self, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.Optimizer.ParamsInspection
            reload(molass_legacy.Optimizer.ParamsInspection)
        from molass_legacy.Optimizer.ParamsInspection import ParamsInspection
        if debug and is_developing_version():
            self.re_construct_optimizer()

        init_params = self.get_init_params()
        dialog = ParamsInspection(self.parent, init_params, self.dsets, self.fullopt)
        dialog.show()

    def redraw_scores(self):
        # nothing else required?
        self.draw_scores()

    def show_complementary_view(self, debug=False):
        if debug:
            from importlib import reload
            import molass_legacy.Optimizer.ComplementaryView
            reload(molass_legacy.Optimizer.ComplementaryView)
        from molass_legacy.Optimizer.ComplementaryView import ComplementaryView
        from molass_legacy.DataStructure.SvdDenoise import get_denoised_data

        dsets = self.get_dsets()
        work_folder = get_setting("analysis_folder")
        init_params = self.get_init_params()
        ov = ComplementaryView(self.parent, self.fullopt, -1, init_params, work_folder)
        ov.show()

    def show_editcanvas(self, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.Peaks.EditCanvas
            reload(molass_legacy.Peaks.EditCanvas)
        from .EditCanvas import EditCanvas
        xr_x, xr_y, xr_peaks, xr_baseline = self.xr_draw_info
        original_info = [xr_x, xr_y, self.xr_peaks_orig, xr_baseline]
        ec = EditCanvas(self, self.sd, self.ecurves, original_info, working_info=self.xr_draw_info)
        ec.show()
        if ec.applied:
            new_pps = ec.get_peak_params_set()
            self.redraw_with_new_params(new_pps)

    def show_cpd_decompose(self, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.GuinierTools.CpdDecompDirect
            reload(molass_legacy.GuinierTools.CpdDecompDirect)
        from molass_legacy.GuinierTools.CpdDecompDirect import cpd_direct_impl
        cpd_direct_impl(self)

    def redraw_with_new_params(self, new_pps):        
        self.peak_params_set = new_pps
        uv_curve, xr_curve = self.ecurves
        xr_x = xr_curve.x
        xr_y = xr_curve.y
        uv_x = uv_curve.x
        uv_y = uv_curve.y
        new_uv_peaks, new_xr_peaks = new_pps[0:2]
        self.draw_elements(new_uv_peaks, new_xr_peaks, uv_x, uv_y, xr_x, xr_y, baselines=self.baselines, draw_canvas=False)
        # recompute init_params
        # task:
        #   it would be better to code this update explicitly
        #   rather than specifing the intension with arguments as here below
        self.draw_scores(create_new_optimizer=True, init_params=None)

    def save_the_figure(self, file):
        self.fig.savefig(file)

    def show_score_details(self, debug=True):
        if debug:
            import molass_legacy.Optimizer.FvScoreDetails as details_module
            from importlib import reload
            reload(details_module)
        from molass_legacy.Optimizer.FvScoreDetails import FvScoreDetails
        if debug and is_developing_version():
            self.re_construct_optimizer()

        details = FvScoreDetails(self.parent, self.fullopt, self.init_params)
        details.show()

    def devel_test(self):
        from importlib import reload
        import molass_legacy.Peaks.PeakDevel
        reload(molass_legacy.Peaks.PeakDevel)
        from molass_legacy.Peaks.PeakDevel import devel_test_impl

        devel_test_impl(self)
