"""
    SuperEditorFrame.py

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
from molass_legacy.DataStructure.AnalysisRangeInfo      import AnalysisRangeInfo, convert_to_paired_ranges
from .DecompSpecPanel import RANGE_FRAME_HEIGHT
from molass_legacy._MOLASS.Version import get_version_string

MIN_NUM_PANEL_SPACES    = 3
CANVAS_HIGHT    = 6

def format_coord(x, y):
    return 'x=%.4g    y=%.4g' % (x, y)

class SuperEditorFrame(Tk.Frame):
    def __init__( self, parent, title):
        Tk.Frame.__init__(self, parent)

        self.build_body(self, title)
        # self.create_popup_menu()  # deprecated?

    def close_figs(self):
        for fig in [self.fig1, self.fig2]:
            plt.close(fig)

    def build_body(self, body_frame, title):

        title_frame = Tk.Frame( body_frame )
        title_frame.pack()
        label = Tk.Label(title_frame, text=title, font=("", 20) )
        label.pack(pady=5)

        self.panel_frame = panel_frame = Tk.Frame( body_frame )
        panel_frame.pack()

        button_frame = Tk.Frame( body_frame )
        button_frame.pack(fill=Tk.X, expand=1)

        ctframe = Tk.Frame( panel_frame )
        ctframe.grid(row=0, column=0, sticky=Tk.W+Tk.E)
        self.data_label = Tk.Label(ctframe)
        self.set_data_label()
        self.data_label.pack(side=Tk.LEFT)
        toolbar_frame = Tk.Frame(ctframe)
        toolbar_frame.pack(side=Tk.LEFT, fill=Tk.X, expand=1, padx=10)

        ccframe = Tk.Frame( panel_frame )
        ccframe.grid(row=1, column=0)

        self.ptframe = Tk.Frame( panel_frame )
        self.ptframe.grid(row=0, column=1, sticky=Tk.W+Tk.E)
        p_frame = Tk.Frame( panel_frame )
        p_frame.grid(row=1, column=1, sticky=Tk.W+Tk.E+Tk.N+Tk.S)

        self.fig1 = fig1 = plt.figure( figsize=(7, CANVAS_HIGHT) )
        self.mpl_canvas1 = FigureCanvasTkAgg( fig1, ccframe )
        self.mpl_canvas1_widget = self.mpl_canvas1.get_tk_widget()
        self.mpl_canvas1_widget.pack( fill=Tk.BOTH, expand=1 )
        self.mpl_canvas1.mpl_connect('button_press_event', self.on_mpl_button_press)

        self.canvas_draw1()
        height_ratios = [5, 1]
        gs = GridSpec( 2, 1, height_ratios=height_ratios )
        self.height_ratio = height_ratios[1]/height_ratios[0]

        self.ax1 = fig1.add_subplot(gs[0,0])
        self.ax1.get_xaxis().set_visible(False)
        self.ax1r = fig1.add_subplot(gs[1,0], sharex=self.ax1)

        for ax in [self.ax1, self.ax1r]:
            ax.format_coord = format_coord    # override the default to avoid fluctuating

        self.draw1()
        fig1.tight_layout()
        # self.toolbar = NavigationToolbar( self.mpl_canvas1, toolbar_frame, show_mode=False )
        self.toolbar = NavigationToolbar( self.mpl_canvas1, toolbar_frame )
        self.toolbar.update()

        self.canvas1_height = int( self.mpl_canvas1_widget.cget( 'height' ) )
        pheight_adjust = Tk.Frame( p_frame, height=self.canvas1_height )
        pheight_adjust.grid(row=0, column=0 )

        self.pcframe_padx = 10
        self.pcframe = Tk.Frame( p_frame )
        self.pcframe.grid(row=0, column=0, padx=self.pcframe_padx, sticky=Tk.N)     # simply adding +Tk.S can be a problem when the figure canvas is resized
                                                                                    # see also adjust_the_tkframe_size  
        self.ppframe = Tk.Frame( p_frame )
        self.ppframe.grid(row=0, column=1, sticky=Tk.N+Tk.S)

        figsize = self.get_figsize()
        self.fig2 = fig2 = plt.figure( figsize=figsize )
        self.mpl_canvas2 = FigureCanvasTkAgg( fig2, self.pcframe )
        self.mpl_canvas2_widget = self.mpl_canvas2.get_tk_widget()
        self.mpl_canvas2_widget.pack( fill=Tk.BOTH, expand=1 )
        self.canvas_draw2()

        self.axes = None
        self.draw2()
        fig2.tight_layout()
        self.update()
        self.add_widgets()

        for j in range(2):
            button_frame.grid_columnconfigure(j, weight=1)

        button_frame_left = Tk.Frame(button_frame)
        button_frame_left.grid( row=0, column=0 )

        if self.toggle:
            self.toggle_btn = Tk.Button(button_frame_left, command=self.toggle_show )
            self.set_toggle_text()
            self.toggle_btn.pack( side=Tk.LEFT)

        self.error_label = Tk.Label(button_frame_left)
        self.error_label.pack( side=Tk.LEFT, padx=40 )
        self.update_error_label()

        button_frame_right = Tk.Frame(button_frame)
        button_frame_right.grid( row=0, column=1, sticky=Tk.E )

        # if get_version_string().find("dev") > 0:
        if False:
            if get_setting('enable_debug_plot'):
                debug_btn = Tk.Button(button_frame_right, text="Debug Plot", command=self.debug_plot )
                debug_btn.grid( row=0, column=0, padx=50 )

            reset_btn = Tk.Button(button_frame_right, text="Reset to Defaults", state=Tk.DISABLED, command=self.reset_to_defaults )
            reset_btn.grid( row=0, column=1, padx=50 )

            if  self.params_controllable:
                self.param_btn = Tk.Button(button_frame_right, text="Show Parameter Constraints", command=self.toggle_params_constraints )
                self.param_btn.grid( row=0, column=2, padx=5 )
                self.recompute_btn = Tk.Button(button_frame_right, text="Recompute", command=self.recompute_decomposition )
                self.recompute_btn.grid( row=0, column=3, padx=5 )
                self.recompute_btn.grid_forget()

    def debug_plot(self):
        # overrided in DecompEditorFrame or RangeEditorFrame
        from importlib import reload
        from molass_legacy.DataStructure.AnalysisRangeInfo import AnalysisRangeInfo
        import DecompEditorDebug
        reload(DecompEditorDebug)
        from DecompEditorDebug import debug_plot_impl
        ranges = AnalysisRangeInfo(self.make_range_info()).get_ranges()
        debug_plot_impl(self, self.dialog)

    def get_figsize(self):
        fig_height = min( CANVAS_HIGHT, self.num_eltns*2 )
        return (3, fig_height)

    def update_error_label(self):
        text='fit_error=%.3g' % self.fit_error
        self.error_label.config(text=text)

    def canvas_draw1(self):
        self.mpl_canvas1.draw()

    def canvas_draw2(self):
        self.mpl_canvas2.draw()

    def draw1(self):
        pass

    def draw2(self, peak_no=None):
        pass

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
        num_eltns = self.num_eltns

        canvas_width = int(self.mpl_canvas2_widget.cget('width')) + self.pcframe_padx*2
        self.col_widths  = [canvas_width, 60, 120, 80, 60, 140, 60, 60, 160]
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

    def set_data_label(self):
        data_type = "Xray" if self.elution_fig_type == 0 else "UV"
        self.data_label.config( text=data_type + " elution decomposition" )

    def set_toggle_text(self):
        # override
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
        self.update_error_label()

    def reset_to_defaults( self ):
        pass

    def get_fit_error( self ):
        return self.fit_error
