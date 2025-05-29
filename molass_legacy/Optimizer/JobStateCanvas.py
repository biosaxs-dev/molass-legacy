"""
    Optimizer.JobStateCanvas.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import timedelta
from molass_legacy.KekLib.OurTkinter import Tk
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.TkUtils import split_geometry
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from molass_legacy.KekLib.NumpyUtils import get_proportional_points
from molass_legacy.KekLib.BasicUtils import ordinal_str
from molass_legacy.KekLib.TimeUtils import friendly_time_str
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from .NaviFrame import NaviFrame
from .FvScoreConverter import convert_score

SHOW_FV_MIN = 0
SHOW_FV_MAX = 100
PROGRESS_X_MARGIN = 1000    # not too large for xmax = 500000
DEVELOP_MODE = True
ENABLE_DEVEL_POPUP_WHILE_BUSY = True

class JobStateCanvas(Tk.Frame):
    def __init__(self, parent, dialog):
        self.parent = parent
        self.dialog = dialog
        self.elution_model = get_setting("elution_model")
        self.update_init_state(dialog.state_info)
        self.iconified = False
        self.logger = dialog.logger
        self.optinit_info = dialog.optinit_info
        self.composite = self.optinit_info.composite
        self.suptitle = None
        Tk.Frame.__init__(self, parent)

        cframe = Tk.Frame(self)
        cframe.pack()
        tframe = Tk.Frame(self)
        tframe.pack(fill=Tk.X, padx=10)
        tframe_left = Tk.Frame(tframe)
        tframe_left.pack(side=Tk.LEFT)
        navi_frame = NaviFrame(tframe, self)
        navi_frame.pack(side=Tk.RIGHT)

        self.fig = fig = plt.figure(figsize=(18, 9))
        gs = GridSpec(33, 15)
        axes = []
        for j in range(3):
            j_ = j*5
            ax = fig.add_subplot(gs[0:16,j_:j_+5])
            axes.append(ax)

        axt = axes[1].twinx()
        axes.append(axt)
        self.axes = axes
        self.prog_ax = fig.add_subplot(gs[17:21,2:])
        peak_ax = fig.add_subplot(gs[21:25,2:])
        rg_ax = fig.add_subplot(gs[25:29,2:])
        map_ax = fig.add_subplot(gs[29:33,2:])
        self.prog_axes = [self.prog_ax, peak_ax, rg_ax, map_ax]
        for ax in self.prog_axes[0:3]:
            ax.set_xticklabels([])
        self.prog_title_axes = [fig.add_subplot(gs[17+i*4:21+i*4,0:2]) for i in range(0,4)]
        prog_titles = ["Function SV", "Peak Top Positions", "Rg Values", "Mapped Range"]
        for ax, title in zip(self.prog_title_axes, prog_titles):
            ax.set_axis_off()
            ax.text(-0.3, 0.5, title, fontsize=16)
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.optview_ranges = None
        self.initial_view_ranges = None

        minimize_btn = Tk.Button(self.mpl_canvas_widget, bg="white", text="Minimize Window", command=self.minimize_window)
        minimize_btn.place(relx=0.925, rely=0)   # anchor=Tk.NE?

        button_text = self.get_view_width_button_text()
        self.view_width_btn = Tk.Button(self.mpl_canvas_widget, bg="white", text=button_text, command=self.change_view_widths)
        self.view_width_btn.place(relx=0.94, rely=0.50)

        residual_view_btn = Tk.Button(self.mpl_canvas_widget, bg="white", text="Residual View", command=self.show_residual_view)
        residual_view_btn.place(relx=0.94, rely=0.55)

        boundary_view_btn = Tk.Button(self.mpl_canvas_widget, bg="white", text="Boundary View", command=self.show_boundary_view)
        boundary_view_btn.place(relx=0.935, rely=0.60)

        if False:
            state = Tk.NORMAL if self.elution_model == 0 else Tk.DISABLED
            sec_inspect_btn = Tk.Button(self.mpl_canvas_widget, bg="white", text="SEC Conformance", command=self.show_sec_conformance, state=state)
            sec_inspect_btn.place(relx=0.925, rely=0.65)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()
        self.popup_menu_0 = None
        self.popup_menu_1 = None
        self.popup_menu_prog = None
        self.popup_menu_devel = None
        self.mpl_canvas.mpl_connect('button_press_event', self.on_figure_click)

    def get_view_width_button_text(self):
        if self.optview_ranges is None:
            text = "Narrower View"
        else:
            text = "Wider View"
        return text

    def update_view_width_button(self):
        self.view_width_btn.config(text=self.get_view_width_button_text())

    def update_init_state(self, state_info):
        self.state_info = state_info
        self.dsets = state_info.dsets
        self.demo_index = state_info.demo_index
        self.demo_info = state_info.demo_info
        self.fullopt = state_info.fullopt

    def close(self):
        plt.close(self.fig)
        self.on_focus_in()  # to focus to the parent dialog

    def minimize_window(self):
        self.dialog.iconify()
        self.dialog.parent_dialog.iconify()
        self.iconified =True
        self.bind("<FocusIn>", self.on_focus_in)

    def on_focus_in(self, *args):
        if self.iconified:
            print("deiconify")
            self.dialog.parent_dialog.deiconify()
            self.iconified =False

    def draw_main(self):
        if self.demo_index is None:
            self.demo_info = None
            self.draw_state()
        else:
            self.draw_state()
            self.draw_progress()
        self.fig.subplots_adjust(top=0.9, left=0.05, right=0.9, hspace=1.1, wspace=1.5, bottom=0.05)

    def draw_state(self):
        if self.demo_index is None:
            self.draw_suptitle()
        else:
            """
            refactor with set_demo_info, draw_state_impl, etc.
            """
            i = self.get_best_index()
            self.best_index = i
            self.curr_index = i
            self.draw_indexed_state(i)

    def draw_suptitle(self):
        from molass_legacy.Optimizer.OptimizerUtils import get_model_name, get_method_name
        job_info = self.dialog.get_job_info()
        job_name = job_info[0]
        in_folder = get_in_folder()
        model_name = get_model_name(self.dialog.class_code)
        text = "Job %s State at %s local minimum on %s with model=%s method=%s" % (
            job_name, ordinal_str(self.curr_index), in_folder, model_name, get_method_name())
        if self.suptitle is None:
            self.suptitle = self.fig.suptitle(text, fontsize=20)
        else:
            self.suptitle.set_text(text)

    def draw_progress(self):
        for ax in self.prog_axes:
            ax.cla()
        for ax in self.prog_axes[0:3]:
            ax.set_xticklabels([])

        # Function Values
        prog_ax = self.prog_ax

        fv, max_num_evals = self.get_fv_array()
        x_, y_ = fv[:,0:2].T
        prog_ax.plot(x_, convert_score(y_))
        prog_ax.set_xlim(-PROGRESS_X_MARGIN, max_num_evals + PROGRESS_X_MARGIN)
        ymin, ymax = prog_ax.get_ylim()
        prog_ax.set_ylim(ymin, ymax)     # these limits will be reset below

        m = self.best_index
        ymin, ymax = prog_ax.get_ylim()
        prog_ax.set_ylim(max(SHOW_FV_MIN, ymin), SHOW_FV_MAX)

        params_type = self.fullopt.params_type

        # Peak Top Positions
        peak_ax = self.prog_axes[1]
        x_array = self.demo_info[1]
        n = self.fullopt.n_components
        x_ = fv[:,0]
        pos_array_list = params_type.get_peak_pos_array_list(x_array)
        for y_ in pos_array_list:
            peak_ax.plot(x_, y_)

        xmin, xmax = self.axes[1].get_xlim()
        xmin_, xmax_ = get_proportional_points(xmin, xmax, [-0.1, 1.1])
        peak_ax.set_ylim(xmin_, xmax_)

        # Rg Values
        rg_ax = self.prog_axes[2]
        rg_start = params_type.get_rg_start_index()
        for k in range(n):
            y_ = x_array[:,rg_start+k]
            rg_ax.plot(x_, y_)
        ymin_rg, ymax_rg = rg_ax.get_ylim()
        rg_ax.set_ylim(10, ymax_rg*1.2)

        # Mapped Range
        map_ax = self.prog_axes[3]
        mr_start = params_type.get_mr_start_index()
        for k in range(2):
            y_ = x_array[:,mr_start+k]
            map_ax.plot(x_, y_)
        map_ax.set_ylim(xmin_, xmax_)

        # Best and Current Result Indicator
        xmin, xmax = self.get_xlim_prog_axes()
        best_x = fv[m,0]
        for k, ax in enumerate(self.prog_axes):
            if k > 0:
                ax.set_xlim(xmin, xmax)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
            ax.plot([best_x, best_x], [ymin, ymax], color='red')
            if self.curr_index != m:
                x = fv[self.curr_index,0]
                ax.plot([x, x], [ymin, ymax], color='yellow')

            if fv.shape[0]-1 not in [m, self.curr_index]:
                x = fv[-1,0]
                ax.plot([x, x], [ymin, ymax], color='gray', alpha=0.3)

        ymin, ymax = map_ax.get_ylim()
        tx = xmax*1.07

        w = 2.2
        ty = ymin*(1-w) + ymax*w
        map_ax.text(tx, ty, "Starting Time", ha="center")

        time_str = self.get_time_started()
        w = 2.0
        ty = ymin*(1-w) + ymax*w
        map_ax.text(tx, ty, time_str, ha="center", va="center")

        w = 1.6
        ty = ymin*(1-w) + ymax*w
        map_ax.text(tx, ty, "Time Elapsed", ha="center")

        time_str = self.get_time_elapsed()
        w = 1.4
        ty = ymin*(1-w) + ymax*w
        map_ax.text(tx, ty, time_str, ha="center", va="center")

        w = 0.4
        ty = ymin*(1-w) + ymax*w
        map_ax.text(tx, ty, "Ending Time", ha="center")

        # guess_ending_time() must be called before get_remaining_time()
        time_str = self.guess_ending_time(fv)
        w = 0.2
        ty = ymin*(1-w) + ymax*w
        map_ax.text(tx, ty, time_str, ha="center", va="center")

        w = 1.0
        ty = ymin*(1-w) + ymax*w
        map_ax.text(tx, ty, "Time Ahead", ha="center")

        time_str = self.get_remaining_time()
        w = 0.8
        ty = ymin*(1-w) + ymax*w
        map_ax.text(tx, ty, time_str, ha="center", va="center")

    def get_xlim_prog_axes(self):
        fv, max_num_evals = self.get_fv_array()
        xmin = -PROGRESS_X_MARGIN
        xmax = max_num_evals + PROGRESS_X_MARGIN
        return xmin, xmax

    def set_demo_info(self, demo_info):
        self.demo_info = demo_info
        self.curr_fv = self.demo_info[0]
        self.draw_suptitle()
        i = self.get_best_index()
        self.best_index = i
        self.curr_index = i
        self.draw_state_impl(i)
        self.draw_progress()

    def get_running_solver_info(self):
        return self.dialog.runner.solver, self.dialog.optinit_info.n_iterations

    def get_fv_array(self):
        if self.demo_info is None:
            assert False
            xmax = 500000
        else:
            fv = self.demo_info[0]
            solver_name, niter = self.get_running_solver_info()
            if solver_name == "ultranest":
                from molass_legacy.Solvers.UltraNest.SolverUltraNest import get_max_ncalls
                # task: unify this estimation
                xmax = get_max_ncalls(niter)
            else:
                xmax = self.demo_info[2]
        return fv, xmax

    def draw_indexed_state(self, i):
        self.draw_suptitle()

        fv = convert_score(self.demo_info[0][i][1])

        for ax in self.axes:
            ax.cla()
        ax1, ax2, ax3, axt = self.axes
        axt.grid(False)
        ax1.set_title("UV Decomposition", fontsize=16)
        ax2.set_title("Xray Decomposition", fontsize=16)
        ax3.set_title("Objective Function Scores in SV=%.3g" % fv, fontsize=16)
        x_array = self.demo_info[1]
        p = x_array[i,:] 
        self.fullopt.objective_func(p, plot=True, axis_info=(self.fig, self.axes))
        self.set_view_widths()

    def get_best_index(self, stop=None):
        fv, xmax = self.get_fv_array()
        m = np.argmin(fv[:stop,1])
        return m

    def get_curr_index(self):
        return self.curr_index

    def get_first(self):
        self.curr_index = 0
        self.draw_state_impl(self.curr_index)

    def get_last(self):
        fv = self.demo_info[0]
        self.curr_index = fv.shape[0] - 1
        self.draw_state_impl(self.curr_index)

    def get_previous(self):
        if self.curr_index > 0:
            self.curr_index -= 1
        self.draw_state_impl(self.curr_index)

    def get_previous_best(self):
        i = self.get_best_index(stop=self.curr_index)
        self.curr_index = i
        self.draw_state_impl(self.curr_index)

    def get_best(self):
        self.curr_index = self.get_best_index()
        self.draw_state_impl(self.curr_index)

    def get_next(self):
        fv, xmax = self.get_fv_array()
        if self.curr_index < fv.shape[0] - 1:
            self.curr_index += 1
        self.draw_state_impl(self.curr_index)

    def get_next_best(self):
        fv, xmax = self.get_fv_array()
        stop = min(len(fv), self.curr_index + 1)
        best_fv_until_now = np.min(fv[:stop,1])
        w  = np.where(fv[stop:,1] < best_fv_until_now)[0]
        if len(w) > 0:
            self.curr_index = stop + w[0]
            self.draw_state_impl(self.curr_index)

    def get_time_started(self):
        try:
            fv_array = self.demo_info[0]
            start_time = fv_array[0,3]
            time = friendly_time_str(start_time)
        except:
            # IndexError: index 3 is out of bounds for axis 1 with size 2
            time = ""
        return time

    def get_time_elapsed(self):
        try:
            fv_array = self.demo_info[0]
            start_time = fv_array[0,3]
            curr_time = fv_array[-1,3]
            hhmmss = str(curr_time - start_time).split(":")
            time = "%3d.%02d" % tuple([int(s) for s in hhmmss[0:2]])
            # %3d instead of %2d is just for positioning purpose with non-fixed-width fonts.
        except:
            log_exception(self.logger, "get_time_elapsed: ")
            # IndexError: index 3 is out of bounds for axis 1 with size 2
            time = ""
        return time

    def guess_ending_time(self, fv):
        finish_time = None
        time = ""

        if fv.shape[0] > 3:
            # unreliable when fv.shape[0] <= 3
            if self.demo_index == 0:
                self.num_iter = self.dialog.num_iter.get()
                try:
                    fv_array = self.demo_info[0]
                    start_time = fv_array[0,3]
                    curr_time = fv_array[-1,3]
                    finish_time = start_time + (curr_time - start_time)*(self.num_iter/fv_array.shape[0])
                    # add 1 minute so that it won't be too early
                    time = friendly_time_str(finish_time + timedelta(minutes=1))
                except:
                    pass

        self.finish_time = finish_time
        return time

    def get_remaining_time(self):
        if self.finish_time is None:
            return ""

        try:
            fv_array = self.demo_info[0]
            curr_time = fv_array[-1,3]
            # add 1 minute so that it won't be too short
            hhmmss = str(self.finish_time - curr_time + timedelta(minutes=1) ).split(":")
            time = "%3d.%02d" % tuple([int(s) for s in hhmmss[0:2]])
            # %3d instead of %2d is just for positioning purpose with non-fixed-width fonts.
            # to be fixed: ValueError: invalid literal for int() with base 10: '-1 day, 23'
        except:
            # log_exception(self.logger, "get_remaining_time: ")
            # IndexError: index 3 is out of bounds for axis 1 with size 2
            # ValueError: invalid literal for int() with base 10: '-1 day, 23'
            time = ""
        return time

    def draw_state_impl(self, i):
        print("draw_state_impl: i=", i)
        self.draw_indexed_state(i)
        self.draw_progress()
        self.mpl_canvas.draw()

    def get_params(self, index):
        return self.demo_info[1][index]

    def get_current_params(self):
        return self.get_params(self.curr_index)

    def get_best_params(self):
        best_index = self.get_best_index()
        return self.get_params(best_index)

    def get_current_fv(self):
        fv, xmax = self.get_fv_array()
        return fv[self.curr_index][1]

    def show_params(self, debug=True):
        if debug:
            from importlib import reload
            import Optimizer.ParamsInspection
            reload(Optimizer.ParamsInspection)
        from .ParamsInspection import ParamsInspection
        self.dialog.grab_set()  # temporary fix to the grab_release problem
        params = self.get_current_params()
        dialog = ParamsInspection(self.dialog.parent, params, self.dsets, self.fullopt, state_info=self.state_info)
        dialog.show()
        self.dialog.grab_set()  # temporary fix to the grab_release problem

    def show_complementary_view(self, debug=True):
        if debug or DEVELOP_MODE:
            from importlib import reload
            import Optimizer.ComplementaryView
            reload(Optimizer.ComplementaryView)
        from molass_legacy.Optimizer.ComplementaryView import ComplementaryView

        self.dialog.grab_set()  # temporary fix to the grab_release problem

        params = self.demo_info[1][self.curr_index]
        work_folder = self.dialog.get_curr_work_folder()
        cv = ComplementaryView(self.dialog.parent, self.fullopt, self.curr_index, params, work_folder, sd=self.optinit_info.sd)
        cv.show()

        self.dialog.grab_set()  # temporary fix to the grab_release problem

    def show_2p_analysis(self, debug=True):
        if debug:
            from importlib import reload
            import Optimizer.TwoParamAnalysis
            reload(Optimizer.TwoParamAnalysis)
        from .TwoParamAnalysis import TwoParamAnalysis

        self.dialog.grab_set()  # temporary fix to the grab_release problem
        dialog = TwoParamAnalysis(self.dialog.parent, self.dsets, self.dialog.fullopt, self.demo_info, self.curr_index)
        dialog.show()
        self.dialog.grab_set()  # temporary fix to the grab_release problem

    def save_the_figure(self, fig_file):
        self.fig.savefig(fig_file)

    def show_residual_view(self):
        from .ResidualView import ResidualView
        view = ResidualView(self.dialog.parent, self.dialog, self)
        view.show()

    def show_boundary_view(self, debug=True):
        if debug:
            from importlib import reload
            import Optimizer.BoundaryView
            reload(Optimizer.BoundaryView)
        from .BoundaryView import BoundaryView
        view = BoundaryView(self.dialog.parent, self.dialog, self)
        view.show()

    def show_sec_conformance(self, debug=True):
        if debug:
            from importlib import reload
            import Optimizer.SecConformance
            reload(Optimizer.SecConformance)
        from .SecConformance import SecConformance
        inspect = SecConformance(self.dialog.parent, self.dialog, self)
        inspect.show()

    def on_figure_click(self, event):
        if event.button == 3:
            if event.inaxes in [self.axes[0]]:
                self.show_popup_menu_0(event)
            elif event.inaxes in [self.axes[1], self.axes[3]]:
                self.show_popup_menu_1(event)
            else:
                if event.inaxes in self.prog_axes:
                    if ENABLE_DEVEL_POPUP_WHILE_BUSY:
                         self.show_popup_menu_prog(event)
                    else:
                        if not self.dialog.is_busy():
                            self.show_popup_menu_prog(event)
                elif event.inaxes == self.axes[2]:
                    self.show_popup_menu_devel(event)

    def show_popup_menu_0(self, event):
        from molass_legacy.KekLib.PopupMenuUtils import post_popup_menu
        self.create_popup_menu_0(event)
        post_popup_menu(self.popup_menu_0, self.mpl_canvas_widget, event, mpl_event=True)

    def create_popup_menu_0(self, event):
        if self.popup_menu_0 is None:
            self.popup_menu_0 = Tk.Menu(self, tearoff=0 )
            self.popup_menu_0.add_command(label='Show Simple 3D View', command=self.show_simple_3d_view)

    def show_popup_menu_1(self, event):
        self.create_popup_menu_1(event)
        canvas = self.mpl_canvas_widget
        cx = canvas.winfo_rootx()
        cy = canvas.winfo_rooty()
        w, h, x, y = split_geometry(canvas.winfo_geometry())
        self.popup_menu_1.post(cx + int(event.x), cy + h - int(event.y))

    def create_popup_menu_1(self, event):
        from molass_legacy._MOLASS.Version import is_developing_version
        if self.popup_menu_1 is None:
            self.popup_menu_1 = Tk.Menu(self, tearoff=0 )
            self.popup_menu_1.add_command(label='Show This Figure', command=self.show_this_figure)
            self.popup_menu_1.add_command(label='Show Rg-visible Figure', command=self.show_rg_visible_figure)
            self.popup_menu_1.add_command(label='Show Rg-visible Figure with SEC Range', command=lambda: self.show_rg_visible_figure(with_range=True))
            self.popup_menu_1.add_command(label='Show Simple 3D View', command=self.show_simple_3d_view)
            self.popup_menu_1.add_command(label='Show Asc/Dsc-Difference', command=self.show_asc_dsc_difference)
            if is_developing_version():
                self.popup_menu_1.add_separator()
                self.popup_menu_1.add_command(label='Investigate SDM Params', command=self.show_sdm_investigator)
                self.popup_menu_1.add_command(label='Investigate SDM LRF Params', command=self.show_sdm_lrf_investigator)
                self.popup_menu_1.add_command(label='Show Ad hoc Figure', command=self.show_adhoc_figure)


    def show_popup_menu_prog(self, event):    
        self.create_popup_menu_prog(event)
        canvas = self.mpl_canvas_widget
        cx = canvas.winfo_rootx()
        cy = canvas.winfo_rooty()
        w, h, x, y = split_geometry(canvas.winfo_geometry())
        self.popup_menu_prog.post(cx + int(event.x), cy + h - int(event.y))

    def create_popup_menu_prog(self, event):
        if self.popup_menu_prog is None:
            self.popup_menu_prog = Tk.Menu(self, tearoff=0 )
            self.popup_menu_prog.add_command(label='Draw all Parameters', command=self.draw_all_parameters)
            if self.elution_model == 1:
                self.popup_menu_prog.add_command(label='SEC Inspection', command=self.sec_inspection)
            self.popup_menu_prog.add_command(label='SEC Conformance Demo', command=self.sec_conformance_demo)
            self.popup_menu_prog.add_command(label='Function Debugger', command=lambda: self.show_function_debugger(composite=self.fullopt.composite))
            self.popup_menu_prog.add_command(label='Function Debugger (no composite)', command=self.show_function_debugger)
            self.popup_menu_prog.add_command(label='FV Score Inspection', command=self.show_fvscore_inspector)
            # self.popup_menu_prog.add_command(label='Compare Scores', command=self.compare_scores)

    def show_popup_menu_devel(self, event):
        self.create_popup_menu_devel(event)
        canvas = self.mpl_canvas_widget
        cx = canvas.winfo_rootx()
        cy = canvas.winfo_rooty()
        w, h, x, y = split_geometry(canvas.winfo_geometry())
        self.popup_menu_devel.post(cx + int(event.x), cy + h - int(event.y))

    def create_popup_menu_devel(self, event):
        if self.popup_menu_devel is None:
            self.popup_menu_devel = Tk.Menu(self, tearoff=0 )
            self.popup_menu_devel.add_command(label='Show Score Details', command=self.show_score_details)
            self.popup_menu_devel.add_command(label='Inspect Score Transition', command=self.inspect_score_transition)
            self.popup_menu_devel.add_command(label='Update Guinier Region', command=self.test_update_guinier_region)
            self.popup_menu_devel.add_command(label='Show Guinier Deviation', command=self.show_guinier_deviation)
            if not self.dialog.is_busy():
                self.popup_menu_devel.add_command(label='Change Function', command=self.change_function)

    def show_rg_visible_figure(self, with_range=False, debug=True):
        if debug:
            import Optimizer.RgVisibleFigure
            from importlib import reload
            reload(Optimizer.RgVisibleFigure)
        from molass_legacy.Optimizer.RgVisibleFigure import RgVisibleFigure
        dialog = RgVisibleFigure(self.parent, self, with_range=with_range)
        dialog.show()

    def show_simple_3d_view(self, debug=True):
        if debug:
            import Tools.SimpleThreedView
            from importlib import reload
            reload(Tools.SimpleThreedView)
        from molass_legacy.Tools.SimpleThreedView import show_simple_3d_view
        show_simple_3d_view(self.optinit_info.sd)

    def show_asc_dsc_difference(self, debug=True):
        if debug:
            import Optimizer.AscDscDifference
            from importlib import reload
            reload(Optimizer.AscDscDifference)
        from molass_legacy.Optimizer.AscDscDifference import show_asc_dsc_difference
        show_asc_dsc_difference(self)

    def draw_all_parameters(self, debug=False):
        if debug:
            import Optimizer.AllParameters
            from importlib import reload
            reload(Optimizer.AllParameters)
        from molass_legacy.Optimizer.AllParameters import AllParameters
        fv, max_num_evals = self.get_fv_array()
        xmin, xmax = self.get_xlim_prog_axes()
        plot = AllParameters(self.parent, self.fullopt, self.demo_info[1], self.curr_index, self.best_index, fv[:,0], xmin, xmax)
        plot.show()

    def sec_inspection(self, debug=True):
        if debug:
            import Optimizer.SecInspection
            from importlib import reload
            reload(Optimizer.SecInspection)
        from molass_legacy.Optimizer.SecInspection import SecInspection
        dialog = SecInspection(self.parent, self)
        dialog.show()

    def sec_conformance_demo(self, debug=True):
        if debug:
            import SecTheory.ConformanceDemo
            from importlib import reload
            reload(SecTheory.ConformanceDemo)
        from SecTheory.ConformanceDemo import ConformanceDemo
        dialog = ConformanceDemo(self.parent)
        dialog.show()

    def show_function_debugger(self, composite=None, debug=True):
        if debug:
            import Optimizer.FunctionDebugger
            from importlib import reload
            reload(Optimizer.FunctionDebugger)
        from molass_legacy.Optimizer.FunctionDebugger import FunctionDebugger
        debugger = FunctionDebugger(self.parent, self.dialog, self, composite=composite)
        debugger.show()

    def show_fvscore_inspector(self, debug=True):
        if debug:
            import Optimizer.FvScoreInspecor
            from importlib import reload
            reload(Optimizer.FvScoreInspecor)
        from molass_legacy.Optimizer.FvScoreInspecor import FvScoreInspecor
        designer = FvScoreInspecor(self.parent, self)
        designer.show()

    def inspect_score_transition(self, debug=True):
        if debug:
            import Optimizer.ScoreTransition
            from importlib import reload
            reload(Optimizer.ScoreTransition)
        from molass_legacy.Optimizer.ScoreTransition import ScoreTransition
        fv, xmax = self.get_fv_array()
        compare = ScoreTransition(self.dialog, self, self.fullopt, self.demo_info[1], fv, self.best_index)
        compare.show()

    def show_score_details(self, debug=True):
        if debug:
            import Optimizer.FvScoreDetails as details_module
            from importlib import reload
            reload(details_module)
        from .FvScoreDetails import FvScoreDetails
        params = self.get_current_params()
        details = FvScoreDetails(self.parent, self.fullopt, params)
        details.show()

    def test_update_guinier_region(self, debug=True):
        if debug:
            import GuinierTools.GuinierDeviationTester
            from importlib import reload
            reload(GuinierTools.GuinierDeviationTester)
        from ..GuinierTools.GuinierDeviationTester import test_update_guinier_region_impl
        test_update_guinier_region_impl(self)

    def show_guinier_deviation(self, debug=True):
        if debug:
            import GuinierTools.GuinierDeviationTester
            from importlib import reload
            reload(GuinierTools.GuinierDeviationTester)
        from ..GuinierTools.GuinierDeviationTester import show_guinier_deviation
        show_guinier_deviation(self)

    def change_function(self, debug=False):
        if debug:
            import Optimizer.FunctionChanger
            from importlib import reload
            reload(Optimizer.FunctionChanger)
        from molass_legacy.Optimizer.FunctionChanger import FunctionChanger
        dialog = FunctionChanger(self.parent, self)
        dialog.show()

    def show_vp_analysis(self, modelname, debug=True):
        if debug:
            import V2PropOptimizer.VariedPropAnalysis
            from importlib import reload
            reload(V2PropOptimizer.VariedPropAnalysis)
        from V2PropOptimizer.VariedPropAnalysis import VariedPropAnalysis
        dialog = VariedPropAnalysis(self.parent, self, modelname)
        dialog.show()

    def show_mw_integrity(self, debug=True):
        if debug:
            import Optimizer.MwIntegrity
            from importlib import reload
            reload(Optimizer.MwIntegrity)
        from molass_legacy.Optimizer.MwIntegrity import MwIntegrityPlot
        dialog = MwIntegrityPlot(self.parent, self)
        dialog.show()

    def show_sdm_investigator(self, debug=True):
        if debug:
            import Models.Stochastic.DispersiveUtils
            from importlib import reload
            reload(Models.Stochastic.DispersiveUtils)
        from molass_legacy.Models.Stochastic.DispersiveUtils import investigate_sdm_params_from_optimizer_params
        print("show_sdm_investigator")
        optimizer = self.fullopt
        i = self.get_best_index()
        x_array = self.demo_info[1]
        p = x_array[i,:]
        investigate_sdm_params_from_optimizer_params(optimizer, p)

    def show_sdm_lrf_investigator(self, debug=True):
        if debug:
            import Models.Stochastic.DispersiveUtils
            from importlib import reload
            reload(Models.Stochastic.DispersiveUtils)
        from molass_legacy.Models.Stochastic.DispersiveUtils import investigate_sdm_lrf_params_from_optimizer_params
        print("show_sdm_lrf_investigator")
        optimizer = self.fullopt
        i = self.get_best_index()
        x_array = self.demo_info[1]
        p = x_array[i,:]
        investigate_sdm_lrf_params_from_optimizer_params(optimizer, p)

    def show_adhoc_figure(self, debug=True):
        if debug:
            import Optimizer.AdhocFigure
            from importlib import reload
            reload(Optimizer.AdhocFigure)
        from molass_legacy.Optimizer.AdhocFigure import show_adhoc_figure_impl
        show_adhoc_figure_impl(self)
    
    def change_view_widths(self):
        from molass_legacy._MOLASS.SerialSettings import set_setting
        from molass_legacy.Trimming.OptViewRange import OptViewRange
        if self.optview_ranges is None:
            optimizer = self.fullopt
            ranges = []
            for curve in [optimizer.uv_curve, optimizer.xr_curve]:
                x = curve.x
                y = curve.y
                range_ = OptViewRange(x, y, upper=0.03)
                ranges.append(range_.get_range())
        else:
            ranges = None

        self.optview_ranges = ranges
        self.set_view_widths(ranges=ranges)
        self.fig.canvas.draw()
        self.update_view_width_button()
    
    def set_view_widths(self, ranges=None):
        if self.initial_view_ranges is None:
            ranges = []
            for ax in self.axes[0:2]:
                ranges.append(ax.get_xlim())
            self.initial_view_ranges = ranges
 
        if ranges is None:
            ranges = self.initial_view_ranges
        
        for ax, range_ in zip(self.axes[0:2], ranges):
            ax.set_xlim(range_)
    
    def show_this_figure(self):
        from importlib import reload
        import Optimizer.XrStateFigure
        reload(Optimizer.XrStateFigure)
        from molass_legacy.Optimizer.XrStateFigure import show_this_figure_impl
        show_this_figure_impl(self)

