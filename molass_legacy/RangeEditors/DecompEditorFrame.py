"""
    DecompEditorFrame.py

    Copyright (c) 2018-2025, SAXS Team, KEK-PF
"""
import copy
import numpy                as np
import matplotlib.pyplot    as plt
from matplotlib.gridspec    import GridSpec
import matplotlib.patches   as mpl_patches      # 'as patches' does not work properly
from matplotlib import colors
import matplotlib
import seaborn as sns
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk
from molass_legacy.KekLib.TkUtils import split_geometry
from molass_legacy.KekLib.OurMatplotlib import get_color, get_hex_color
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from .SuperEditorFrame import SuperEditorFrame
from molass_legacy.DataStructure.PeakInfo import PeakInfo
from molass_legacy.DataStructure.RangeInfo import DecompEditorInfo, shift_editor_ranges
from molass_legacy.DataStructure.AnalysisRangeInfo import AnalysisRangeInfo, convert_to_paired_ranges
from molass_legacy.SerialAnalyzer.PairedRangeLogger import log_paired_ranges
from molass_legacy.Decomposer.DecompUtils import (
                                    make_range_info_impl,
                                    decompose_elution_better,
                                    )
from molass_legacy.Decomposer.UnifiedDecompResult import UnifiedDecompResult
from .DecompSpecPanel import SpecPanel, RANGE_FRAME_HEIGHT
from molass_legacy.SerialAnalyzer.DataUtils import compact_path_name

MIN_NUM_PANEL_SPACES    = 3
CANVAS_HIGHT    = 6
DECOMP_EDITOR_LOG_HEADER  = "--- decomp editor log begin ---"
DECOMP_EDITOR_LOG_TRAILER = "--- decomp editor log end ---"
RESTORE_HINTS_DICT = False      # TODO

toggle_button_texts = [ "Show Xray decomposition", "Show UV decomposition" ]

def format_coord(x, y):
    return 'x=%.4g    y=%.4g' % (x, y)

class ModelDecompInfo:
    def __init__(self, decomp_ret):
        self.model_name     = decomp_ret.model_name
        self.opt_recs       = decomp_ret.opt_recs
        self.opt_recs_uv    = decomp_ret.opt_recs_uv
        self.num_eltns      = len(self.opt_recs)

class DecompEditorFrame(SuperEditorFrame):
    def __init__(self, parent, dialog, sd, mapper, corbase_info, model, decomp_result=None, global_flag=False):
        self.dialog = dialog
        self.logger = dialog.logger
        Tk.Frame.__init__(self, parent)

        self.mplt_ge_2_2 = matplotlib.__version__ >= '2.2'

        self.mapper = mapper
        self.xr_j0 = sd.xr_j0
        self.corbase_info = corbase_info
        self.model = model
        name = model.get_name()
        self.model_name = name[0:3]     # task: add a method for this name

        memorized_model = get_setting('editor_model')
        range_info = get_setting('decomp_editor_info')  # temporary fix
        self.is_memorized = memorized_model is not None and memorized_model == self.model_name and range_info is not None

        self.params_controllable = True
        self.elution_fig_type = 0   # 0 : Xray, 1:UV
        self.popup_menu = None
        self.ex_solver = None

        self.recompute_decomposition(decomp_result=decomp_result, redraw=False)
        self.decomp_result_init = self.decomp_result

        self.toggle = True
        gf = 'Global ' if global_flag else ''
        in_folder_name = compact_path_name(get_setting('in_folder'))
        title = "%sDecomposition Specification for %s using %s" % (gf, in_folder_name, self.model_name)
        SuperEditorFrame.__init__(self, parent, title)
        if global_flag:
            self.toggle_btn.config(state=Tk.DISABLED)

    def is_delayed(self):
        return self.model.is_delayed()

    def debug_plot(self):
        from importlib import reload
        from molass_legacy.DataStructure.AnalysisRangeInfo import AnalysisRangeInfo
        import molass_legacy.Decomposer.DecompEditorDebug
        reload(molass_legacy.Decomposer.DecompEditorDebug)
        from molass_legacy.Decomposer.DecompEditorDebug import debug_plot_impl
        ranges = AnalysisRangeInfo(self.make_range_info()).get_ranges()
        debug_plot_impl(self, self.dialog)

    def get_figsize(self):
        fig_height = min( CANVAS_HIGHT, self.num_eltns*2 )
        figsize = (3, fig_height)
        print("figsize=", figsize)
        return figsize

    def update_error_label(self):
        text='fit_error=%.3g' % self.fit_error
        self.error_label.config(text=text)

    def draw1(self, debug=False):
        if debug:
            from importlib import reload
            import molass_legacy.RangeEditors.DecompPlotUtils
            reload(molass_legacy.RangeEditors.DecompPlotUtils)

        from .DecompPlotUtils import draw1_impl
        draw1_impl(self)

    def write_residual_amount_to_testerlog(self, y, resid_y):
        from molass_legacy.Test.TesterLogger import write_to_tester_log

        max_y = np.percentile(y, 95)
        normalized_resid_amount = np.average(np.abs(resid_y)) / len(resid_y) / max_y
        type_ = 'Xray' if self.elution_fig_type == 0 else 'UV'

        write_to_tester_log( "normalized_resid_amount=%g with %s\n" % (normalized_resid_amount, type_))

    def draw2(self, peak_no=None, debug=False):
        self.fig2.clear()
        figsize = self.get_figsize()
        self.fig2.set_size_inches(*figsize, forward=True)   # forward=True is required when the canvas is resized
                                                            # see also self.refresh_figs below

        fx = self.fx
        x  = self.x
        if self.elution_fig_type == 0:
            y   = self.y
            color = 'orange'
            opt_recs = self.opt_recs
        else:
            y   = self.uv_y
            linestyle = ':'
            color = 'blue'
            opt_recs = self.opt_recs_uv

        num_eltns = self.num_eltns
        gs = GridSpec( num_eltns, 1 )
        self.axes = [ self.fig2.add_subplot( gs[i] ) for i in range(num_eltns) ]

        self.fig2_range_parts_list = []
        for k, ax in enumerate(self.axes):
            if debug:
                print([k], peak_no)
            ax.cla()
            # ax.set_axis_off()
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.plot(x, y, color=color)
            ee_select = self.select_matrix[k]
            for j, e in enumerate(ee_select):
                if debug:
                    print("j, e=", j, e)
                if e == 1:
                    func = opt_recs[j][1]
                    ax.plot(x, func(fx), ':', color=get_color(j), linewidth=3)

            range_parts = []

            if  ( len(self.specpanel_list) == 0 and not self.ignorable_flags[k]
                or len(self.specpanel_list) > 0 and self.specpanel_list[k].ignore.get() == 0 ):
                self.add_range_patchs(ax, range_parts, k=k)

            self.fig2_range_parts_list.append(range_parts)

    def add_range_patchs(self, ax, range_parts, k=None):
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)

        range_parts.append([ymin, ymax])

        if k is None:
            editor_ranges_ = self.editor_ranges
        else:
            editor_ranges_ = [self.editor_ranges[k]]

        for j, range_list in enumerate(editor_ranges_):
            if k is None and self.ignorable_flags[j]:
                range_parts.append([j])
                continue

            for f, t in range_list:
                p = mpl_patches.Rectangle(
                        (f, ymin),  # (x,y)
                        t - f,   # width
                        ymax - ymin,    # height
                        facecolor   = 'cyan',
                        alpha       = 0.2,
                    )
                ax.add_patch( p )
                lines = []
                for x in [f, t]:
                    line, = ax.plot( [x, x], [ymin, ymax], ':', color='gray', alpha=0.5 )
                    lines.append(line)
                range_parts.append([j, p, lines])

    def refresh_figs(self, devel=False):
        if devel:
            from importlib import reload
            import molass_legacy.KekLib.TkMplUtils
            reload(molass_legacy.KekLib.TkMplUtils)
        from molass_legacy.KekLib.TkMplUtils import adjust_the_tkframe_size
        self.draw1()
        self.draw2()
        self.canvas_draw1()
        self.canvas_draw2()
        # self.update()
        adjust_the_tkframe_size(self.fig2, self.mpl_canvas2_widget, self.pcframe)    # call this after fig2 is drawn

    def update1(self):
        print('update1 start')
        try:
            self.update1_impl(self.fig1_range_parts)
            self.update1_impl(self.fig1_range_parts_resid)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            print(etb)
        print('update1 end')

    def update1_impl(self, range_parts):
        print('update1_impl start')
        ymin, ymax = range_parts[0]
        for k, parts in enumerate(range_parts[1:]):
            if parts is None:
                continue

            j, patch, lines = parts
            range_list = self.editor_ranges[j]
            print([k, j], 'range_list=', range_list)
            for f, t in range_list:
                x, y = patch.xy
                patch.set_xy((f, y))
                patch.set_width(t - f)
                for n, x_ in enumerate([f, t]):
                    line = lines[n]
                    line.set_data([x_, x_], [ymin, ymax])

    def update2(self, peak_no):
        print('update2: peak_no=', peak_no)
        pk = peak_no - 1
        range_parts = self.fig2_range_parts_list[pk]

        ymin, ymax = range_parts[0]
        for k, parts in enumerate(range_parts[1:]):
            if parts is None:
                continue

            j, patch, lines = parts
            range_list = self.editor_ranges[j]
            for f, t in range_list:
                x, y = patch.xy
                patch.set_xy((f, y))
                patch.set_width(t - f)

    def update_figs(self, peak_no):
        self.update1()
        self.update2(peak_no)
        self.canvas_draw1()
        self.canvas_draw2()

    def add_widgets(self):
        num_eltns = self.num_eltns

        canvas_width = int(self.mpl_canvas2_widget.cget('width')) + self.pcframe_padx*2
        self.col_widths  = [canvas_width, 60, 120, 80, 60, 160, 60, 60, 160]
        # self.col_widths[2] = 120 required for SUB_TRN1 with EMG
        label_texts = [ 'Figure',
                        'Peak No',
                        'Element Curve(s)',
                        'Number of\nRanges',
                        'Range Id',
                        'Range(s)\nFrom       To',
                        'Ignore',
                        ]
        if self.params_controllable:
            label_texts.append( 'τ Value' )
            label_texts.append( 'τ Constraints' )

        self.col_title_frames = []
        for k, text in enumerate(label_texts):
            frame = Tk.Frame(self.ptframe, width=self.col_widths[k], height=40)
            frame.grid(row=0, column=k, sticky=Tk.S)
            self.col_title_frames.append(frame)
            # https://stackoverflow.com/questions/16363292/label-width-in-tkinter
            frame.pack_propagate(0)
            label = Tk.Label(frame, text=text)
            label.pack(side=Tk.BOTTOM)

        if self.params_controllable:
            self.col_title_frames[-2].grid_forget()
            self.col_title_frames[-1].grid_forget()

        for k in range(max(MIN_NUM_PANEL_SPACES, num_eltns)):
            self.ppframe.grid_rowconfigure(k, weight=1)

        if self.is_memorized and RESTORE_HINTS_DICT:
            hints_dict = get_setting('tau_hints_dict')
            self.logger.info('restored hints_dict will be used: ' + str(hints_dict))
        else:
            hints_dict = None
        self.refresh_spec_panels(hints_dict=hints_dict)

        for k in range( max(0, MIN_NUM_PANEL_SPACES - num_eltns) ):
            space = Tk.Frame(self.ppframe, height=RANGE_FRAME_HEIGHT)
            space.grid(row=num_eltns+k, column=0)

    def make_range_info(self, concentration_datatype=2, debug=False):
        """
            called at PreviewButtonFrame.get_preview_data()
        """
        shifted_ranges = shift_editor_ranges(-self.xr_j0, self.editor_ranges)
        self.control_info.editor_ranges = shifted_ranges

        if concentration_datatype == 0:       # Xray elution model
            from importlib import reload
            import molass_legacy.Decomposer.OptRecsUtils
            reload(molass_legacy.Decomposer.OptRecsUtils)
            from molass_legacy.Decomposer.OptRecsUtils import eoii_correct_opt_recs
            opt_recs_ = copy.deepcopy(self.opt_recs)
            eoii_correct_opt_recs(self, opt_recs_)

        elif concentration_datatype == 1:     # Xray measured data
            opt_recs_ = self.opt_recs   # will not be used, currenctly
        else:
            opt_recs_ = self.opt_recs_uv

        set_setting('concentration_datatype', concentration_datatype)
        return make_range_info_impl(opt_recs_, self.control_info, self.specpanel_list, logger=self.logger, debug=debug, parent=self)

    def make_restorable_ranges(self):
        paired_ranges = []
        for panel in self.specpanel_list:
            paired_ranges.append(panel.get_paired_range())
        return paired_ranges

    def apply( self ):  # overrides parent class method
        set_setting( 'use_elution_models', 1 )

        analysis_range_info = AnalysisRangeInfo( self.make_range_info(), editor='DecompEditor' )
        print('analysis_range_info=', analysis_range_info)
        set_setting( 'analysis_range_info', analysis_range_info )
        set_setting( 'range_type', 4 )
        paired_ranges = self.make_restorable_ranges()
        set_setting( 'decomp_editor_info', DecompEditorInfo(paired_ranges, self.ignorable_flags))
        set_setting( 'editor_model', self.model_name )

        if RESTORE_HINTS_DICT:
            hints_dict = self.get_param_hints_dict()
            set_setting( 'tau_hints_dict', hints_dict )

        ret = convert_to_paired_ranges(analysis_range_info.get_ranges())
        paired_ranges = ret[0]
        self.log_info_for_reset()
        log_paired_ranges(self.logger, paired_ranges)

    def set_data_label(self):
        data_type = "Xray" if self.elution_fig_type == 0 else "UV"
        self.data_label.config( text=data_type + " elution decomposition" )

    def set_toggle_text(self):
        self.toggle_btn.config( text=toggle_button_texts[1-self.elution_fig_type] )

    def toggle_show( self ):
        self.elution_fig_type = 1 - self.elution_fig_type
        self.refresh_frame()
        self.dialog.update_global_btn_state(self.elution_fig_type)

    def refresh_frame(self, elution_fig_type=None):
        if elution_fig_type is not None:
            self.elution_fig_type = elution_fig_type
        self.set_data_label()
        self.set_toggle_text()
        self.refresh_figs()
        self.update_error_label()

    def reset_to_defaults( self ):
        pass

    def get_fit_error( self ):
        return self.fit_error

    def toggle_params_constraints( self ):
        btn_text = self.param_btn.cget('text')
        is_show_button = btn_text.find("Show") >= 0
        if is_show_button:
            k = len(self.col_title_frames) - 1
            self.col_title_frames[-2].grid(row=0, column=k-1, sticky=Tk.S)
            self.col_title_frames[-1].grid(row=0, column=k, sticky=Tk.S)
        else:
            self.col_title_frames[-2].grid_forget()
            self.col_title_frames[-1].grid_forget()

        for specpanel in self.specpanel_list:
            if is_show_button:
                specpanel.constraints_restore()
            else:
                specpanel.constraints_forget()

        if is_show_button:
            text_ = btn_text.replace("Show", "Hide")
            self.recompute_btn.grid(row=0, column=3, padx=5)
        else:
            text_ = btn_text.replace("Hide", "Show")
            self.recompute_btn.grid_forget()

        self.param_btn.config(text=text_)

        self.update()   # necessary to make effective the following xview_moveto

        if is_show_button:
            if self.dialog.scrollable:
                self.dialog.scrolled_frame.canvas.xview_moveto(1)

    def recompute_decomposition(self, decomp_result=None, redraw=True , debug=False):
        if debug:
            print('recompute_decomposition')
            
        hints_dict = None   # task: remove this hints_dict

        init_mode = decomp_result is None

        if init_mode:
            if self.model.is_traditional():
                decomp_result = decompose_elution_better(self.corbase_info, self.mapper, self.model,
                                    hints_dict=hints_dict, logger=self.logger)
            else:
                if self.model_name == 'STC':
                    from importlib import reload
                    import molass_legacy.Selective.StochasticAdapter
                    reload(molass_legacy.Selective.StochasticAdapter)
                    from molass_legacy.Selective.StochasticAdapter import convert_to_stochastic_decomposition
                    decomp_result = convert_to_stochastic_decomposition(self.dialog)
                else:
                    # i.e., EDM 
                    from importlib import reload
                    import molass_legacy.Selective.AdvancedAdapter
                    reload(molass_legacy.Selective.AdvancedAdapter)
                    from molass_legacy.Selective.AdvancedAdapter import adapted_decompose_elution_simply
                    decomp_result = adapted_decompose_elution_simply(self)

        self.decomp_result = decomp_result
        if init_mode:
            self.decomp_result_init = decomp_result

        # TODO: simple_args=simple_args, use_emga=use_emga, dual_opt=dual_opt

        # task: remove this branch
        if self.model_name == 'STC':
            self.fx = decomp_result.x
            self.x = decomp_result.x
        else:
            self.fx = decomp_result.x
            self.x = self.xr_j0 + decomp_result.x

        self.y = decomp_result.y
        self.uv_y = decomp_result.uv_y
        self.opt_recs = decomp_result.opt_recs
        self.opt_recs_uv = decomp_result.opt_recs_uv
        self.num_eltns = len(decomp_result.opt_recs)

        # be aware that we are getting info from Xray fit recs
        # debug = self.model_name == 'STC'
        self.control_info = decomp_result.get_range_edit_info(logger=self.logger, debug=False)

        self.select_matrix  = self.control_info.select_matrix
        self.top_x_list     = self.control_info.top_x_list

        if self.is_memorized and False:
            # task: reconsider this case
            range_info = get_setting('decomp_editor_info')
            range_info.update(self.control_info)
            assert range_info is not None

            self.ignorable_flags = range_info.get_ignorable_flags()
            self.logger.info('ignorable_flags have been restored as ' + str(self.ignorable_flags))
            ranges = range_info.get_ranges()
            self.editor_ranges  = shift_editor_ranges(self.xr_j0, ranges)
            self.logger.info('editor_ranges have been restored as ' + str(self.editor_ranges))

            self.control_info.editor_ranges = ranges
            """
            the above is required since the following are done to or from self.control_info
                1) SpecPanel.range_list( == control_info.editor_ranges[k]) update
                2) self.make_range_info(), i.e., make_range_info_impl(...)
            """
        else:
            self.ignorable_flags = decomp_result.identify_ignorable_elements()
            self.logger.info('ignorable_flags have been initialized to %s', self.ignorable_flags)
            self.editor_ranges  = shift_editor_ranges(self.xr_j0, self.control_info.editor_ranges)
            self.logger.info('editor_ranges have been initialized to %s with xr_j0=%d', self.editor_ranges, self.xr_j0)

        self.specpanel_list = []

        if redraw:
            self.refresh_figs()
            self.refresh_spec_panels(restore=True, hints_dict=hints_dict)
            self.panel_frame.update()

    def get_decomp_score( self ):
        score = self.decomp_result.compute_synthetic_score(self.ignorable_flags, self.select_matrix)
        self.logger.info('score on %s is %g' % (self.model_name, score))
        return score

    def refresh_spec_panels(self, restore=False, hints_dict=None):
        if len(self.specpanel_list) > 0:
            for specpanel in self.specpanel_list:
                specpanel.destroy()
            self.specpanel_list = []

        active_peak_no = 0
        row_no_base = 0
        for k in range(self.num_eltns):
            # print('editor_ranges[%d]=' % k, self.editor_ranges[k])
            if not self.ignorable_flags[k] and len(self.editor_ranges[k]) > 0:
                active_peak_no += 1

            range_list = self.editor_ranges[k]
            panel = SpecPanel(self.ppframe,
                            editor=self,
                            j_min=self.xr_j0,
                            j_max=self.xr_j0 + len(self.x) - 1,
                            peak_no=k+1,
                            active_peak_no=None if self.ignorable_flags[k] else active_peak_no,
                            row_no_base=row_no_base,
                            col_widths=self.col_widths,
                            select_list=self.select_matrix[k],
                            range_list=range_list,
                            model=self.model,
                            # params_controllable=self.params_controllable,
                            params_controllable = False,    # controlable params are decprecated
                            opt_rec=self.opt_recs[k],
                            hints_dict=hints_dict,
                            )
            if not self.ignorable_flags[k]:
                row_no_base += len(range_list)
            if restore:
                panel.constraints_restore()
            panel.grid(row=k, column=0, sticky=Tk.N+Tk.S)
            self.specpanel_list.append(panel)

    def get_param_hints_dict(self):
        from collections import OrderedDict

        hints_dict = OrderedDict()
        for specpanel in self.specpanel_list:
            try:
                xkey = specpanel.peak.get_xkey()
                hints = specpanel.get_tau_hints()
                hints_dict[xkey] = hints
            except:
                # temporary fix: hints_dict will not be used anyway
                pass
        return hints_dict

    def log_info_for_reset(self, modification="last applied"):
        self.logger.info(DECOMP_EDITOR_LOG_HEADER)

        self.logger.info(modification + " model_name=" + self.model_name)

        self.logger.info("element cuves:")
        for k, panel in enumerate(self.specpanel_list):
            tau_hints = panel.get_tau_hints()
            xray_evaluator = self.opt_recs[k].evaluator
            param_text = xray_evaluator.get_all_params_string()
            xray_h = xray_evaluator.get_param_value(0)
            uv_evaluator = self.opt_recs_uv[k].evaluator
            uv_h = uv_evaluator.get_param_value(0)
            uv2xray_scale = uv_h/xray_h
            self.logger.info( str([k+1]) + ' ' + param_text + '; with tau constraints ' + str(tau_hints) + ' and uv/xray h ratio %g' % uv2xray_scale )

        self.logger.info("used ranges:")
        for k, panel in enumerate(self.specpanel_list):
            if self.ignorable_flags[k] or len(self.editor_ranges[k]) == 0:
                continue

            curves = []
            for m, flag in enumerate(self.select_matrix[k]):
                if flag:
                    curves.append(m+1)

            info = panel.get_info_for_reset()
            self.logger.info(str([k+1]) + ' ' + str(info[1]) + ' with element curves ' + str(curves) )

        self.logger.info(DECOMP_EDITOR_LOG_TRAILER)

    def on_mpl_button_press(self, event):
        if event.xdata is None:
            return

        if event.button == 3:
            from molass_legacy.KekLib.PopupMenuUtils import post_popup_menu
            self.create_popup_menu()
            post_popup_menu(self.popup_menu, self.mpl_canvas1_widget, event, mpl_event=True)
            return

    def create_popup_menu(self):
        if self.popup_menu is None:
            self.popup_menu = Tk.Menu( self, tearoff=0 )
            self.popup_menu.add_command( label='Save the curves', command=self.save_the_curves )

    def save_the_curves(self):
        print('save_the_curves')
        from molass_legacy.SerialAnalyzer.CurveSaverDialog import CurveSaverDialog

        curve_list = []
        for k, rec in enumerate(self.the_curves):
            label, x, y = rec
            curve = np.array([x, y]).T
            curve_list.append([label, curve, label + '.dat'])

        model_type = self.model_name.lower()
        data_type = 'xray' if self.elution_fig_type == 0 else 'uv'
        save_folder = '/'.join( [get_setting('analysis_folder'), model_type, data_type] )
        dialog = CurveSaverDialog(self.dialog, curve_list, save_folder)
        dialog.show()

    def update_range(self, pno, ad, f, t):
        print('update_range:', pno, ad, f, t, len(self.specpanel_list))
        range_ = self.editor_ranges[pno][ad]
        range_[0] = f
        range_[1] = t
        panel = self.specpanel_list[pno]
        panel.update_range(ad, f, t)
        # self.update_figs(pno)
        self.refresh_figs()

    def get_extrapolation_solver(self):
        """
        always construct a new solver to initialize simply
        """
        from molass_legacy.Extrapolation.ExtrapolationSolver import ExtrapolationSolver
        pdata, popts = self.dialog.preview_frame.get_preview_data()
        self.ex_solver = ExtrapolationSolver(pdata, popts)
        return self.ex_solver

    def compute_global_fitting(self):
        from molass_legacy.Optimizer.OptimalElution import OptimalEmg, OptimalEgh, get_h_vector, select_resc
        from molass_legacy.DataStructure.SvdDenoise import get_denoised_data

        self.logger.info('computing %s global fit', self.model_name)
        D, E, qvector, ecurve = self.dialog.serial_data.get_xr_data_separate_ly()
        h_vector, rank = get_h_vector(self.opt_recs)
        fit_recs = select_resc(h_vector, rank, self.opt_recs)
        M = get_denoised_data(D, rank=rank)

        if self.model_name[0:3] == "EMG":
            opimizer = OptimalEmg(M, ecurve, fit_recs, rank, error=E, debug=False)
        else:
            opimizer = OptimalEgh(M, ecurve, fit_recs, rank, error=E, debug=False)

        return UnifiedDecompResult(
                        xray_to_uv=None,
                        x_curve=ecurve, x=ecurve.x, y=ecurve.y,
                        opt_recs=opimizer.fit_recs,
                        max_y_xray = ecurve.max_y,
                        model_name=opimizer.model.get_name(),
                        decomposer=None,
                        uv_y=None,
                        opt_recs_uv=None,
                        max_y_uv= None,
                        global_flag=True,
                        )

    def compute_jsd(self):
        from scipy.spatial import distance

        x_curve = self.decomp_result.x_curve
        x = x_curve.x
        y = x_curve.y.copy()
        y[y < 0] = 0

        cy_list = []
        for rec in self.decomp_result.opt_recs:
            cy_list.append(rec.evaluator(x))

        ty = np.sum(cy_list, axis=0)
        return distance.jensenshannon(ty, y)

    def update_with_result(self, decomp_result, debug=False):
        self.recompute_decomposition(decomp_result=decomp_result, debug=debug)

    def reset(self):
        self.recompute_decomposition(decomp_result=self.decomp_result_init)
