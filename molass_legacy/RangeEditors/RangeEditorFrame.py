"""
    RangeEditorFrame.py

    Copyright (c) 2019-2024, SAXS Team, KEK-PF
"""
import os
import copy
import numpy                as np
import matplotlib.pyplot    as plt
from matplotlib.gridspec    import GridSpec
import matplotlib.patches   as mpl_patches      # 'as patches' does not work properly
from matplotlib import colors
import matplotlib
import seaborn as sns
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter             import Tk
from molass_legacy.KekLib.TkUtils                import split_geometry
from molass_legacy.KekLib.OurMatplotlib          import NavigationToolbar, get_color, get_hex_color
from molass_legacy._MOLASS.SerialSettings         import get_setting, set_setting
from DevSettings            import get_dev_setting
from .SuperEditorFrame import SuperEditorFrame
from molass_legacy.DataStructure.AnalysisRangeInfo      import AnalysisRangeInfo, convert_to_paired_ranges
from RangeInfo              import RangeEditorInfo, shift_editor_ranges
from PairedRangeLogger      import log_paired_ranges
from .RangeSpecFrame import RangeSpecFrame, RANGE_FRAME_HEIGHT

MIN_NUM_PANEL_SPACES    = 3
CANVAS_HIGHT    = 6
RANGE_EDITOR_LOG_HEADER  = "--- range editor log begin ---"
RANGE_EDITOR_LOG_TRAILER = "--- range editor log end ---"

def format_coord(x, y):
    return 'x=%.4g    y=%.4g' % (x, y)

class RangeEditorFrame(SuperEditorFrame):
    def __init__( self, parent, dialog, sd, mapper, corbase_info ):
        self.dialog = dialog
        self.logger = dialog.logger
        Tk.Frame.__init__(self, parent)

        self.mplt_ge_2_2 = matplotlib.__version__ >= '2.2'

        self.mapper = mapper
        self.xr_j0 = sd.xr_j0
        self.corbase_info = corbase_info

        self.elution_fig_type = 0   # 0 : Xray, 1:UV
        self.popup_menu = None
        self.ex_solver = None

        e_curve = mapper.x_curve
        self.xr_x = self.xr_j0 + e_curve.x
        self.xr_y = mapper.x_curve_y_adjusted
        self.uv_y = mapper.mapped_vector
        self.peaks = e_curve.get_emg_peaks()
        self.get_editor_ranges(e_curve, self.peaks)
        self.num_peaks = len(self.peaks)

        self.recompute_decomposition(redraw=False)

        self.params_controllable = False

        self.toggle = False
        title = "Range Specification for %s" % get_setting('in_folder')
        SuperEditorFrame.__init__(self, parent, title)

    def get_editor_ranges(self, curve, peaks):
        range_info = get_setting('range_editor_info')
        self.is_memorized = range_info is not None      # just temporary
        default_ranges = curve.get_default_editor_ranges()

        if range_info is None:
            ranges = default_ranges
            ignorable_flags = [0] * len(ranges)
        else:
            range_info.update(default_ranges)
            ranges = range_info.get_ranges()
            ignorable_flags = range_info.get_ignorable_flags()

        assert len(ranges) == len(peaks)

        self.editor_ranges = shift_editor_ranges(self.xr_j0, ranges)
        self.ignorable_flags = ignorable_flags

    def debug_plot(self):
        from molass_legacy.DataStructure.AnalysisRangeInfo import AnalysisRangeInfo
        from DecompEditorDebug import debug_plot_impl
        ranges = AnalysisRangeInfo(self.make_range_info()).get_ranges()
        debug_plot_impl(self.x, ranges, np.arange(253, 302))

    def get_figsize(self):
        fig_height = min( CANVAS_HIGHT, self.num_peaks*2 )
        return (3, fig_height)

    def update_error_label(self):
        # required to avoid accessing to self.fit_error
        pass

    def draw1(self):
        ax  = self.ax1
        ax.cla()

        x   = self.xr_x
        xr_y = self.xr_y
        uv_y = self.uv_y

        ax.plot(x, xr_y, color='orange', label='Adjusted Xray Elution')
        ax.plot(x, uv_y, ':', color='blue', label='Mapped UV Elution')

        ax.legend()
        self.fig1_range_parts = []
        self.add_range_patchs(ax, self.fig1_range_parts)

        # draw the difference
        ax1r = self.ax1r
        ax1r.cla()

        diff_y = uv_y - xr_y
        ax1r.bar( x, diff_y, color='purple', label='Difference (UV - Xray)', alpha=0.5 )
        ax1r.plot( x[[0, -1]], [0, 0], color='pink' )

        ymin, ymax = ax.get_ylim()
        yminr, ymaxr = ax1r.get_ylim()
        ymidr = (yminr + ymaxr)/2
        half_height_r = max((ymax - ymin)*self.height_ratio, ymaxr - yminr)/2
        ax1r.set_ylim(ymidr - half_height_r, ymidr + half_height_r)

        ax1r.legend(bbox_to_anchor=(1, 1), loc='lower right')
        self.fig1_range_parts_resid = []
        self.add_range_patchs(ax1r, self.fig1_range_parts_resid)

    def draw2(self, peak_no=None):

        if self.axes is not None:
            self.fig2.clf()
            figsize = self.get_figsize()
            self.fig2.set_size_inches(*figsize)

        x   = self.xr_x
        xr_y = self.xr_y
        uv_y = self.uv_y

        num_peaks = self.num_peaks
        gs = GridSpec( num_peaks, 1 )
        self.axes = [ self.fig2.add_subplot( gs[i] ) for i in range(num_peaks) ]

        self.fig2_range_parts_list = []
        for k, ax in enumerate(self.axes):
            if peak_no is not None:
                # do the following only for k == i
                if k != peak_no - 1:
                    # continue
                    pass

            ax.cla()
            # ax.set_axis_off()
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.plot(x, xr_y, color='orange')
            ax.plot(x, uv_y, ':', color='blue')

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

    def refresh_figs(self):
        self.draw1()
        self.draw2()
        self.canvas_draw1()
        self.canvas_draw2()

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
        pk = peakno - 1
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
        num_peaks = self.num_peaks

        canvas_width = int(self.mpl_canvas2_widget.cget('width')) + self.pcframe_padx*2
        self.col_widths  = [canvas_width, 60, 80, 60, 160, 60, 60, 160]
        label_texts = [ 'Figure',
                        'Peak No',
                        'Number of\nRanges',
                        'Range Id',
                        'Range(s)\nFrom       To',
                        'Ignore',
                        ]

        self.col_title_frames = []
        for k, text in enumerate(label_texts):
            frame = Tk.Frame(self.ptframe, width=self.col_widths[k], height=40)
            frame.grid(row=0, column=k, sticky=Tk.S)
            self.col_title_frames.append(frame)
            # https://stackoverflow.com/questions/16363292/label-width-in-tkinter
            frame.pack_propagate(0)
            label = Tk.Label(frame, text=text)
            label.pack(side=Tk.BOTTOM)

        for k in range(max(MIN_NUM_PANEL_SPACES, num_peaks)):
            self.ppframe.grid_rowconfigure(k, weight=1)

        self.refresh_spec_panels()

        for k in range( max(0, MIN_NUM_PANEL_SPACES - num_peaks) ):
            space = Tk.Frame(self.ppframe, height=RANGE_FRAME_HEIGHT)
            space.grid(row=num_peaks+k, column=0)

    def make_range_info(self, concentration_datatype=3):
        # concentration_datatype is not used here.
        # it is included in the above arguments to keep the inferface consistent
        # as seen from PreviewButtonFrame
        paired_ranges = []
        for k, panel in enumerate(self.specpanel_list):
            if self.ignorable_flags[k]:
                continue
            paired_ranges.append(panel.get_paired_range())
        return paired_ranges

    def make_restorable_ranges(self):
        paired_ranges = []
        for panel in self.specpanel_list:
            paired_ranges.append(panel.get_paired_range())
        return paired_ranges

    def apply( self ):  # overrides parent class method
        set_setting( 'use_elution_models', 0 )

        analysis_range_info = AnalysisRangeInfo( self.make_range_info(), editor='RangeEditor' )
        print('analysis_range_info=', analysis_range_info)
        set_setting( 'analysis_range_info', analysis_range_info )
        set_setting( 'range_type', 4 )
        paired_ranges = self.make_restorable_ranges()
        set_setting( 'range_editor_info', RangeEditorInfo(paired_ranges, self.ignorable_flags))

        ret = convert_to_paired_ranges(analysis_range_info.get_ranges())
        paired_ranges = ret[0]
        self.log_info_for_reset()
        log_paired_ranges(self.logger, paired_ranges)

    def set_data_label(self):
        pass

    def set_toggle_text(self):
        pass

    def toggle_show( self ):
        self.elution_fig_type = 1 - self.elution_fig_type
        self.refresh_frame()

    def refresh_frame(self, elution_fig_type=None):
        if elution_fig_type is not None:
            self.elution_fig_type = elution_fig_type
        self.set_data_label()
        self.set_toggle_text()
        self.refresh_figs()

    def reset_to_defaults( self ):
        pass

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

    def recompute_decomposition( self, redraw=True ):

        self.specpanel_list = []

        if redraw:
            self.refresh_figs()
            self.refresh_spec_panels(restore=True)
            self.panel_frame.update()

    def refresh_spec_panels(self, restore=False):
        if len(self.specpanel_list) > 0:
            for specpanel in self.specpanel_list:
                specpanel.destroy()
            self.specpanel_list = []

        active_peak_no = 0
        row_no_base = 0
        for k in range(self.num_peaks):
            # print('editor_ranges[%d]=' % k, self.editor_ranges[k])
            if not self.ignorable_flags[k] and len(self.editor_ranges[k]) > 0:
                active_peak_no += 1

            range_list = self.editor_ranges[k]
            panel = RangeSpecFrame(self.ppframe,
                            editor=self,
                            j_min=self.xr_j0,
                            j_max=self.xr_j0 + len(self.xr_x) - 1,
                            peak_no=k+1,
                            active_peak_no=None if self.ignorable_flags[k] else active_peak_no,
                            row_no_base=row_no_base,
                            col_widths=self.col_widths,
                            range_list=range_list,
                            peak=self.peaks[k],
                            )
            if not self.ignorable_flags[k]:
                row_no_base += len(range_list)
            if restore:
                panel.constraints_restore()
            panel.grid(row=k, column=0, sticky=Tk.N+Tk.S)
            self.specpanel_list.append(panel)

    def log_info_for_reset(self, modification="last applied"):
        self.logger.info(RANGE_EDITOR_LOG_HEADER)

        self.logger.info(modification + " model_name=None")
        self.logger.info("used ranges:")
        for k, panel in enumerate(self.specpanel_list):
            if self.ignorable_flags[k] or len(self.editor_ranges[k]) == 0:
                continue

            info = panel.get_info_for_reset()
            self.logger.info(str([k+1]) + ' ' + str(info[1]) )

        self.logger.info(RANGE_EDITOR_LOG_TRAILER)

    def on_mpl_button_press(self, event):
        pass

    def update_range(self, pno, ad, f, t):
        print('update_range:', pno, ad, f, t)
        range_ = self.editor_ranges[pno][ad]
        range_[0] = f
        range_[1] = t
        panel = self.specpanel_list[pno]
        panel.update_range(ad, f, t)
        # self.update_figs(pno)
        self.refresh_figs()

    def get_extrapolation_solver(self):
        if self.ex_solver is None:
            """
            self.ex_solver should be set to None
            whenever popts are changed.
            """
            from ExtrapolationSolver import ExtrapolationSolver
            pdata, popts = self.dialog.preview_frame.get_preview_data()
            self.ex_solver = ExtrapolationSolver(pdata, popts)
        return self.ex_solver
