"""
    ExtrapolSolverDialog.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF
"""
import os
import copy
import re
from bisect import bisect_right
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpl_patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkUtils import adjusted_geometry
from molass_legacy.KekLib.TkSupplements import BlinkingFrame
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar, get_color
from molass_legacy.KekLib.DebugPlot import set_plot_env
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from .ExtrapolationControlPanel import ControlPanel, ENABLE_KNOWN_INPUT
from .ExtrapolationAnimation import TOO_SMALL_TO_PLOT
from .ExtrapolationDenssMenu import ExtrapolationDenssMenu
from molass_legacy.Test.TesterLogger import write_to_tester_log
from molass_legacy.SerialAnalyzer.DevSettings import set_dev_setting
from molass_legacy.SerialAnalyzer.DataUtils import compact_path_name
from .PreviewButtonFrame import ELUTION_DATATYPE_NAMES
from molass_legacy.Selective.AdvancedFrame import get_change_name
from molass_legacy._MOLASS.Version import is_developing_version

SMALL_ANGLE_LIMIT_Q = 0.1
B_SHIFT_POINT_Q     = 0.02
AUTO_CDI_BUTTON = False

def format_func(value, tick_number):
    return r'$10^{%d}$' % (value)

class ExtrapolSolverDialog( Dialog ):
    def __init__( self, parent, pool, editor_frame=None, last_change_id=None, scrollable=True ):
        self.mplt_ge_2_2 = matplotlib.__version__ >= '2.2'
        self.logger = logging.getLogger(__name__)
        self.grab = 'local'     # used in grab_set
        self.parent = parent
        set_plot_env(sub_parent=self.parent)    # for debug plot in case when set_plot_env has not yet called, e.g., running the while app.
        pdata = pool.pdata
        self.pdata = pdata
        self.popts = pool.popts
        self.mapper = pdata.mapper
        self.editor_frame = editor_frame
        self.last_change_id = last_change_id

        self.sd = pdata.sd
        self.j0 = pdata.sd.xr_j0
        self.conc = pdata.conc
        self.conc_curves = pdata.conc_curves
        self.elution_y =  pdata.make_conc_vector()
        self.mc_vector  = pdata.mc_vector
        self.doing_sec = pdata.is_for_sec
        self.arg_paired_ranges = pdata.paired_ranges    # keep this to call resursively in solve_unknowns
        self.cnv_ranges = pdata.cnv_ranges
        self.num_ranges = pdata.num_ranges
        self.known_info_list = None
        self.rank_increment = get_setting('rank_increment')

        solver = pool.solver
        self.solver = solver
        self.data   = solver.data
        self.error  = solver.error
        self.x_curve = solver.ecurve
        self.aq_smoothness = solver.aq_smoothness
        self.aq_positivity = solver.aq_positivity

        if pdata.is_for_sec:
            self.q = pdata.sd.intensity_array[0,:,0]
        else:
            self.q = pdata.xdata.vector

        self.selector = pool.selector
        self.solver_results = pool.get_better_results()
        self.use_elution_models = pool.popts.use_elution_models
        self.lrf_bound_correction = get_setting("lrf_bound_correction")
        self.scrollable = scrollable
        self.DEFAULT_WEIGHTS = get_setting('penalty_weighting')
        self.applied    = False
        self.geometry_init = True
        Dialog.__init__( self, self.parent, "Extrapolation Preview", visible=False)

    def cancel(self):
        # overiding cancel to cleanup self.fig
        # because the call to the destructor __del__ seems to be delayed
        plt.close(self.fig)
        # print("ExtrapolSolverDialog: closed fig")
        Dialog.cancel(self)

    def show( self ):
        self._show()

    def body( self, body_frame ):

        title_frame = Tk.Frame( body_frame )
        title_frame.pack()
        in_folder_name = compact_path_name(get_setting('in_folder'))
        concentration_datatype = get_setting('concentration_datatype')
        data_type_name = ELUTION_DATATYPE_NAMES[concentration_datatype]
        preview_model = get_setting('preview_model')
        if preview_model is not None:
            data_type_name = data_type_name.replace("elution", preview_model)

        last_change = "" if self.last_change_id is None else " after " + get_change_name(self.last_change_id)
        title = "LRF Preview of %s using %s%s" % (in_folder_name, data_type_name, last_change)
        label = Tk.Label(title_frame, text=title, font=("", 20) )
        label.pack(pady=5)

        bottom_frame = Tk.Frame(body_frame)
        bottom_frame.pack( side=Tk.BOTTOM, fill=Tk.X )
        self.bottom_frame = bottom_frame

        if self.scrollable:
            from molass_legacy.KekLib.ScrolledFrame import ScrolledFrame
            self.scrolled_frame = ScrolledFrame(body_frame)
            # self.scrolled_frame.pack( fill=Tk.BOTH, expand=1 )
            self.scrolled_frame.pack(anchor=Tk.N)
            fframe = self.scrolled_frame.interior
        else:
            self.scrolled_frame = None
            fframe = Tk.Frame( body_frame )
            fframe.pack(anchor=Tk.N)

        bottom_frame1 = Tk.Frame( bottom_frame )
        bottom_frame1.pack(fill=Tk.X, pady=5)

        self.guide_frame = Tk.Frame(bottom_frame)
        self.guide_frame.pack(fill=Tk.X)

        from molass_legacy.QuickAnalysis.AnalysisGuide import get_data_correction_state_text
        text = "Showing data processed with " + get_data_correction_state_text()
        guide_label = Tk.Label(self.guide_frame, text=text, bg='white')
        guide_label.pack(fill=Tk.X, pady=10)
        self.guide_message = guide_label

        tframe = Tk.Frame( bottom_frame1 )
        tframe.pack( side=Tk.LEFT, fill=Tk.X, expand=1 )
        space = Tk.Frame(bottom_frame1, width=50)
        space.pack(side=Tk.LEFT)
        bframe = Tk.Frame( bottom_frame1 )
        bframe.pack( side=Tk.RIGHT )

        cframe = Tk.Frame( fframe )
        cframe.pack( side=Tk.LEFT )

        self.pframe = pframe = Tk.Frame( fframe )
        pframe.pack( side=Tk.LEFT, fill=Tk.BOTH, padx=10 )

        x_curve = self.x_curve

        sd  = self.sd
        data = self.data

        # num_peaks = len( x_curve.peak_info )
        num_panels = self.num_ranges
        h = 4
        self.fig = fig = plt.figure( figsize=(20, num_panels*h) )
        gs = GridSpec(num_panels, 11)

        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.canvas_draw()

        self.toolbar = NavigationToolbar( self.mpl_canvas, tframe )
        self.toolbar.update()

        self.fx = np.arange( data.shape[1] )
        self.elution_x = self.j0 + self.fx

        self.sa_limit = bisect_right( self.q, SMALL_ANGLE_LIMIT_Q )
        self.qs = self.q[0:self.sa_limit]
        self.b_shift_point = bisect_right( self.q, B_SHIFT_POINT_Q )

        # previous_penalty_matrix = get_setting('zx_penalty_matrix')
        # print('previous_penalty_matrix=', previous_penalty_matrix)
        previous_penalty_matrix = None
        # TODO: remove this control by previous_penalty_matrix

        self.zx_penalty_matrix = []
        self.axis_array = []
        self.super_panel_list = []
        self.panel_list = []

        ignore_all_bqs = get_setting('ignore_all_bqs')
        # ignore_bq_list = get_setting('ignore_bq_list')
        ignore_bq_list = None
        selector = self.selector

        row = 0
        for pno, range_ in enumerate(self.cnv_ranges):
            fromto_list = range_.get_fromto_list()
            for ad, _ in enumerate(fromto_list):
                pframe.grid_rowconfigure( row, weight=1 )
                peakset_info = selector.select_peakset(row)
                if previous_penalty_matrix is None:
                    weights = copy.deepcopy(self.DEFAULT_WEIGHTS)
                else:
                    weights = previous_penalty_matrix[row]
                self.zx_penalty_matrix.append(weights)
                if ignore_bq_list is None:
                    ignore_bq = ignore_all_bqs
                else:
                    ignore_bq = ignore_bq_list[row]
                panel = ControlPanel( pframe, self, pno, ad, range_, row, peakset_info, weights, ignore_bq, editor=self.editor_frame)
                panel.grid( row=row, column=0 )
                self.super_panel_list.append(panel)
                self.panel_list.append(panel.unknown_panel)

                ax1 = fig.add_subplot(gs[row, 0:2])
                ax2A1 = fig.add_subplot(gs[row, 2:5])
                ax2B1 = ax2A1.twinx()
                ax3 = fig.add_subplot(gs[row, 5:8])
                ax4 = fig.add_subplot(gs[row, 8:11])
                self.axis_array.append([ax1, ax2A1, ax2B1, ax3, ax4])
                row += 1

        self.to_solve_ranges = []
        self.Y_list = []
        self.topx_list = []
        self.sg_list = []
        self.data_list = []
        row = 0
        self.rg_quality_list = []
        quality_warning = False
        for pno, paired_range in enumerate(self.cnv_ranges):
            # print( 'paired_range=', paired_range )
            top_ = paired_range.top_x
            fromto_list = paired_range.get_fromto_list()
            for range_ in fromto_list:
                start   = range_[0]
                stop    = range_[1] + 1
                self.to_solve_ranges.append( [ paired_range, start, stop ] )
                top_c = self.mc_vector[top_]
                # top_c = elution_y[top_]
                Y   = data[:,top_]
                Y_  = Y / top_c
                Y_[Y_ < TOO_SMALL_TO_PLOT] = 0  # to avoid deforming the figure with these values
                self.Y_list.append( Y_ )
                self.topx_list.append(top_)

                peakset_info = selector.select_peakset(row)
                # print([pno], 'peakset_info=', peakset_info)

                if ignore_bq_list is None:
                    ignore_bq = ignore_all_bqs
                else:
                    ignore_bq = ignore_bq_list[row]
                self.data_list.append(None)
                lrf_info, C = self.solve_plot( row, peakset_info, init_mode=True, ignore_bq=ignore_bq )
                self.rg_quality_list.append((lrf_info.Rg, lrf_info.basic_quality))
                if lrf_info.basic_quality < 0.5:
                    quality_warning = True
                self.panel_list[row].set_C_matrix(C)
                self.sg_list.append(lrf_info.sg)
                self.plot_guinier_kratky(row)
                row += 1

        if quality_warning:
            fully_automatic = get_setting('fully_automatic')
            suppress_low_quality_warning = get_setting('suppress_low_quality_warning')
            if fully_automatic or suppress_low_quality_warning:
                pass
            else:
                # after(1000, ..) seems too early for slower machines
                self.after(2000, self.show_quality_warning)

        fig.tight_layout()
        fig.subplots_adjust(left=0.02, wspace=1.5)
        # plt.show()
        self.log_rg_info()

        self.canvas_draw()

        if AUTO_CDI_BUTTON:
            self.auto_cdi_btn = Tk.Button(bframe, text="auto-CDI for all", command=self.auto_cdi_for_all )
            self.auto_cdi_btn.pack(side=Tk.RIGHT, padx=10)

        self.denss_menu = ExtrapolationDenssMenu(bframe, self, self.data_list)
        self.denss_menu.pack(side=Tk.RIGHT, padx=30)

        if ENABLE_KNOWN_INPUT:
            self.su_btn_blink = BlinkingFrame(bframe)
            self.su_btn_blink.pack(side=Tk.RIGHT, padx=10)
            btn = Tk.Button(self.su_btn_blink, text="Solve Unknowns", command=self.solve_unknowns, state=Tk.NORMAL )
            btn.pack()
            self.su_btn_blink.objects = [btn]

        self.save_btn = Tk.Button(bframe, text="Save results", command=self.save_results )
        self.save_btn.pack(side=Tk.RIGHT, padx=10)

        self.add_dnd_bind()

        self.set_geometry()

    def show_quality_warning(self):
        if get_setting('test_pattern') is not None:
            # suppress warning while testing
            return

        import molass_legacy.KekLib.CustomMessageBox as MessageBox

        self.update()
        message =  (
                "     Rg    basic quality\n"
                "   ---------------------\n"
                )
        for rg, quality in self.rg_quality_list:
            rg_ = " None" if rg is None else "%5.1f" % rg
            low_qaulity_text = "   ← low quality" if quality < 0.5 else ""
            message += "    %s     %4.2f%s\n" % (rg_, quality, low_qaulity_text)

        if self.popts.conc_depend > 1:
            message += ("\nand that\n"
                        '"low quality" may have been caused by rank mismatch,\n'
                        "in that case, consider retrying with Conc. Dependency=1."
                        )

        MessageBox.showwarning("Guinier Analysis Quality Warning",
            "Be aware that this preview includes some low quality results\n"
            "as shown below\n\n"
            "%s" % message,
            parent=self)

    def set_geometry(self):
        from molass_legacy.KekLib.MultiMonitor import get_selected_monitor
        self.update()
        canvas_width = int(self.mpl_canvas_widget.cget( 'width' ))
        canvas_height = int(self.mpl_canvas_widget.cget( 'height' ))
        panel_width = 180
        print("canvas_height=", canvas_height)
        monitor = get_selected_monitor()
        monitor_height = monitor.height
        height = min(int(monitor_height*0.8), canvas_height+200)
        wxh = '%dx%d' % (canvas_width+panel_width, height)
        geometry = self.geometry()
        new_geometry = re.sub( r'(\d+x\d+)(.+)', lambda m: wxh + m.group(2), geometry)
        self.geometry(new_geometry)

    def canvas_draw( self ):
        if self.mplt_ge_2_2:
            self.mpl_canvas.draw()
        else:
            self.mpl_canvas.show()

    def buttonbox( self ):
        box = Tk.Frame(self.bottom_frame)
        box.pack()

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=50, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=50, pady=5)

        if is_developing_version():
            w = Tk.Button(box, text="On the Fly", width=10, command=self.on_the_fly)
            w.pack(side=Tk.LEFT, padx=50, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def solve_plot( self, row, peakset_info, penalty_weights=None, init_mode=False, animation=False, ignore_bq=0 ):
        paired_range, start, stop = self.to_solve_ranges[row]

        # A, B = self.solver.extrapolate( start, stop, with_intercept=False )
        if init_mode:
            A, B, Z, E, lrf_info, C = self.solver_results[row]
        else:
            A, B, Z, E, lrf_info, C = self.solver.extrapolate_wiser( start, stop, peakset_info,
                                ignore_bq=ignore_bq,
                                penalty_weights=penalty_weights, animation=animation )
            self.solver_results[row] = [A, B, Z, E, lrf_info, C]

        need_bq = lrf_info.need_bq()

        file_name = self.panel_list[row].get_file_name()
        self.data_list[row] = (self.q, A, E[0], file_name)
        Bs = B[self.b_shift_point]

        i = (start + stop)//2

        ax1, ax2A1, ax2B1 = self.axis_array[row][0:3]
        for ax in [ax1, ax2A1, ax2B1]:
            ax.cla()

        ax1.set_title("Range", fontsize=16)
        ax1.axes.get_xaxis().set_ticks([])
        ax1.axes.get_yaxis().set_ticks([])
        ax1.plot(self.elution_x, self.elution_y, color='blue')
        topx = self.topx_list[row]
        ax1.plot(self.j0+topx, self.elution_y[topx], 'o', color='yellow')

        if self.use_elution_models:
            pno, nth, peakset, known_peak_info = peakset_info
            paired_ranges_ = [self.cnv_ranges[i] for i in peakset]
            for k, paired_range in enumerate(paired_ranges_):
                if known_peak_info is None:
                    known = False
                else:
                    known = False if known_peak_info[k] is None else True
                for elm_rec in paired_range.elm_recs:
                    e   = elm_rec[0]
                    fnc = elm_rec[1]
                    if fnc.accepts_real_x:
                        # temporary fix under V1PreviewAdapter
                        fy = fnc(self.elution_x)
                    else:
                        fy = fnc(self.fx)
                    if known:
                        ax1.plot( self.elution_x, fy, color='yellow', linewidth=7, alpha=0.3)
                    ax1.plot( self.elution_x, fy, ':', color=get_color(e), linewidth=3 if k == nth else 1.5)
        else:
            if self.doing_sec:
                pass
            else:
                x_ = np.arange(stop - start)
                y_ = self.conc_curves[row](x_)
                ax1.plot(start + x_, y_, ':', color=get_color(row), linewidth=3)

        ymin, ymax = ax1.get_ylim()
        p = mpl_patches.Rectangle(
                (self.j0+start, ymin),  # (x,y)
                stop - start,   # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        ax1.add_patch( p )

        rank = C.shape[0]
        xmin, xmax = ax1.get_xlim()
        ymin, ymax = ax1.get_ylim()
        tx = xmin * 0.95 + xmax * 0.05
        ty = ymin * 0.3 + ymax * 0.7
        rank_increment = 0 if self.solver.rank_control else self.rank_increment
        rank_inc = '+%d' % rank_increment if rank_increment > 0 else ''
        ax1.text(tx, ty, "Rank: %d%s" % (rank, rank_inc), alpha=0.2, fontsize=20 )

        for ax in [ax2A1]:
            ax.set_xlabel( 'Q' )
            ax.set_ylabel( 'log( I/C )' )
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

        if need_bq:
            ax2B1.axes.get_yaxis().set_ticks([])
            ax2B1.set_axis_on()
            # ax2B1.set_ylabel( 'I/C' )
        else:
            ax2B1.set_axis_off()

        Y_ = self.Y_list[row]

        logY = np.log10(Y_)
        logA = np.log10(A)

        ax2A1.set_title("Log Plot", fontsize=16)
        ax2A1.plot( self.q, logY, label='ln(I(q))' )
        ax2A1.plot( self.q, logA, label='ln(A(q))' )
        if need_bq:
            ax2B1.plot( self.q, B, color='pink', label='B(q)' )
            if self.lrf_bound_correction:
                try:
                    ymin, ymax = ax2B1.get_ylim()
                    ax2B1.set_ylim(ymin, ymax)
                    for k, bound in enumerate(lrf_info.bq_bounds):
                        label = "B(q) bounds" if k == 0 else None
                        ax2B1.plot(self.q, bound, ":", color='red', label=label)
                except:
                    log_exception(self.logger, "plot bb_bounds failure: ")
            ax2_ = ax2B1
        else:
            ax2_ = ax2A1

        ax2A1.legend()
        if need_bq:
            ax2B1.legend(bbox_to_anchor=(1, 0.1), loc='lower right')
        ax2B1.grid(False)
        if animation:
            return self.solver.get_anim_data()
        else:
            return lrf_info, C

    def plot_guinier_kratky(self, row, devel=False):
        if devel:
            from importlib import reload
            import molass_legacy.Kratky.GuinierKratkyPlots
            reload(molass_legacy.Kratky.GuinierKratkyPlots)
        from molass_legacy.Kratky.GuinierKratkyPlots import guinier_kratky_plots_impl
        ax3, ax4 = self.axis_array[row][3:]
        ay = self.solver_results[row][0]     # A
        sg = self.sg_list[row]
        color = "C1"
        this_rg = sg.Rg
        if this_rg is None or this_rg == 0:
            this_rg = None
            interval = None
        else:
            interval = sg.guinier_start, sg.guinier_stop

        guinier_kratky_plots_impl(ax3, ax4, self.q, ay, this_rg, sg.Iz, color, interval=interval, markersize=1)

        last_rg = self.sg_list[row-1].Rg if row > 0 else None

        rg_text_color = 'black'
        if last_rg is not None and this_rg is not None:
            if self.doing_sec:
                if last_rg < this_rg:
                    rg_text_color = 'red'
            else:
                if last_rg > this_rg:
                    rg_text_color = 'red'

        xmin, xmax = ax3.get_xlim()
        ymin, ymax = ax3.get_ylim()
        tx = xmin * 0.95 + xmax * 0.05
        ty = ymin * 0.8 + ymax * 0.2

        Rg_ = 'None' if this_rg is None else '%.1f' % this_rg
        rg_text = ax3.text(tx, ty, 'Rg: %s' % Rg_, color=rg_text_color, alpha=0.2, fontsize=20 )

    def validate( self ):
        solving_required = False
        for panel in self.panel_list:
            if panel.solving_required:
                solving_required = True
                break
        if solving_required:
            import molass_legacy.KekLib.OurMessageBox as MessageBox
            ok = MessageBox.showinfo( "Confirmation",
                    "You have to 'Cancel' to finish because you have not 'Solve'd with changed parameters.",
                    parent=self )
            return False

        return True

    def apply( self ):  # overrides parent class method
        self.applied = True

        ignore_bq_list = []
        for k, panel in enumerate(self.panel_list):
            self.zx_penalty_matrix[k] = panel.get_penalty_weights()
            ignore_bq_list.append(panel.ignore_bq.get())

        ignore_bq_set = set(ignore_bq_list)
        num_states = len(ignore_bq_set)
        if num_states == 1:
            unique_value = list(ignore_bq_set)[0]
            ignore_all_bqs = unique_value
            if unique_value == 0:
                ignore_bq_list = None
        else:
            ignore_all_bqs = 0

        # when this set is not done, default values are used in ExtrapolationSolver
        set_setting( 'zx_penalty_matrix', self.zx_penalty_matrix )
        set_setting( 'ignore_bq_list', ignore_bq_list )
        set_setting( 'ignore_all_bqs', ignore_all_bqs )
        known_info_list = self.get_known_info_list()
        set_setting('known_info_list', known_info_list)

        print('applied: zx_penalty_matrix=', self.zx_penalty_matrix)
        self.try_image_save()

    def try_image_save( self ):
        test_pattern = get_setting('test_pattern')
        if test_pattern is None or test_pattern < 7:
            return

        preview_image_folder = get_setting('preview_image_folder')
        if preview_image_folder is not None and os.path.exists(preview_image_folder):
            self.save_the_figure( preview_image_folder, get_setting('analysis_name') )

    def save_the_figure( self, folder, analysis_name ):
        from molass_legacy.SerialAnalyzer.DataUtils import cut_upper_folders
        in_folder = cut_upper_folders(get_setting('in_folder'))
        self.fig.suptitle(in_folder)
        self.fig.subplots_adjust(top=0.9)

        # print( 'save_the_figure: ', folder, analysis_name )
        filename = analysis_name.replace( 'analysis', 'figure' )
        path = os.path.join( folder, filename )
        self.fig.savefig( path )

    def get_ignore_bq( self, row ):
        return self.panel_list[row].ignore_bq.get()

    def save_results( self ):
        from .PreviewResultSaver import PreviewResultSaverDialog
        self.saver = PreviewResultSaverDialog(self, self)
        self.saver.show()

    def plot_known(self, row, selected, data):
        from molass_legacy.GuinierAnalyzer.AutorgKekAdapter import AutorgKekAdapter
        # unify with the part of solve_plot

        print('plot_known: row=', row, data.shape)
        ax1, ax2A1, ax2B1 = self.axis_array[row]

        autorg_kek = AutorgKekAdapter( data )
        autorg_result = autorg_kek.run()

        print(autorg_result.Rg)

        for ax in [ax2A1, ax2B1]:
            ax.cla()

        for ax in [ax2A1]:
            ax.set_xlabel( 'Q' )
            ax.set_ylabel( 'log( I/C )' )
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

        Y_ = self.Y_list[row]
        A = data[:,1]

        logY = np.log10(Y_)
        logA = np.log10(A)

        last_rg = self.sg_list[row-1].Rg if row > 0 else None
        this_rg = autorg_result.Rg

        rg_text_color = 'black'
        if last_rg is not None and this_rg is not None:
            if self.doing_sec:
                if last_rg < this_rg:
                    rg_text_color = 'red'
            else:
                if last_rg > this_rg:
                    rg_text_color = 'red'

        ax2A1.plot( self.q, logY, label='ln(I(q))' )
        ax2A1.plot( self.q, logA, label='ln(A(q))' )

        xmin, xmax = ax2A1.get_xlim()
        ymin, ymax = ax2A1.get_ylim()
        tx = xmin * 0.95 + xmax * 0.05
        ty = ymin * 0.8 + ymax * 0.2
        Rg_ = 'None' if this_rg is None else '%.1f' % autorg_result.Rg
        rg_text = ax2A1.text(tx, ty, 'Rg: %s' % Rg_, color=rg_text_color, alpha=0.2, fontsize=20 )

        ax2A1.legend()
        self.canvas_draw()

    def solve_unknowns(self):
        from .PreviewController import PreviewController

        self.su_btn_blink.stop()

        known_info_list = self.get_known_info_list()

        self.preview_ctl = PreviewController(dialog=self.editor_dialog, editor=self.editor_frame)
        self.preview_ctl.run_solver(self, self.pdata, self.popts, known_info_list=known_info_list)

        if self.preview_ctl.ok():
            self.preview_ctl.show_dialog()
            # self.update_widgets()
            if self.preview_ctl.dialog.applied:
                # self.update_settings()
                # TODO:
                pass
        else:
            # TODO
            pass

    def get_known_info_list(self):
        from .ExtrapolationSolver import KnownInfo
        known_info_list = []
        num_knowns = 0
        for k, panel in enumerate(self.super_panel_list):
            known = panel.known.get()
            print([k], known)
            if known:
                try:
                    info = KnownInfo(panel.known_panel.data)
                    num_knowns += 0
                except:
                    # temp fix for the case when panel.known_panel.data is not ready
                    info = None
            else:
                info = None
            known_info_list.append(info)
        return None if num_knowns == 0 else known_info_list

    def update_to_solve_range(self, row, f, t):
        to_solve_row = self.to_solve_ranges[row]
        to_solve_row[1] = f
        to_solve_row[2] = t

    def auto_cdi_for_all(self, out_folder=None):
        from molass_legacy.Conc.CdInspection import CdInspectionDailog
        print('auto_cdi_for_all')
        if out_folder is None:
            out_folder = get_setting('analysis_folder') + '/cdi-images'
            if not os.path.exists(out_folder):
                from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry
                mkdirs_with_retry(out_folder)

        xray_scale = self.sd.get_xray_scale()
        row = 0
        for pno, paired_range in enumerate(self.cnv_ranges):
            fromto_list = paired_range.get_fromto_list()
            for ad, _ in enumerate(fromto_list):
                ignore_bq = False
                print([pno, ad], paired_range)
                q = self.q
                f, t = paired_range[ad]
                eslice = slice(f, t+1)
                M = self.data[:, eslice]
                E = self.error[:, eslice]
                from_ax = self.axis_array[row][0]
                C = self.panel_list[row].C
                self.logger.info("cdi for peak %d-%d elution range(%d, %d)", pno, ad, f, t+1)
                dialog = CdInspectionDailog(self.parent, self, M, E, C, q, eslice, from_ax=from_ax, xray_scale=xray_scale)
                def auto_action():
                    dialog.save_the_figure(out_folder, pno, ad)
                    dialog.ok()
                dialog.after(1000, auto_action)
                dialog.show()
                row += 1

    def log_rg_info(self):
        rg_list = [sg.Rg for sg in self.sg_list]
        self.logger.info("rg_list=%s", str(rg_list))

    def add_dnd_bind(self):
        self.mpl_canvas_widget.register_drop_target("*")

        def dnd_handler(event):
            self.on_drop(event)

        self.mpl_canvas_widget.bind("<<Drop>>", dnd_handler)

    def on_drop(self, event, debug=True):
        from molass_legacy.SerialAnalyzer.SerialDataUtils import serial_np_loadtxt
        if debug:
            import molass_legacy.SecTools.LogGuinierKratkyPlotter
            from importlib import reload
            reload(molass_legacy.SecTools.LogGuinierKratkyPlotter)
        from molass_legacy.SecTools.LogGuinierKratkyPlotter import LogGuinierKratkyPlotter

        files = event.data.split(' ')
        print('on_drop:', files)

        path = files[0]
        data, _ = serial_np_loadtxt(path)
        print("data.shape=", data.shape)
        plotter = LogGuinierKratkyPlotter(self, data, path)
        plotter.show()

    def on_the_fly(self):
        from OnTheFly.DebugDialog import DebugDialog
        dialog = DebugDialog(self)
        dialog.show()
