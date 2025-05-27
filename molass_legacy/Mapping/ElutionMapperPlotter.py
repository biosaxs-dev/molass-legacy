"""

    ElutionMapperPlotter.py

        recognition of peaks

    Copyright (c) 2018-2024, SAXS Team, KEK-PF

"""
import os
import numpy                as np
import copy
from scipy.interpolate      import UnivariateSpline
import matplotlib
import matplotlib.pyplot    as plt
import matplotlib.patches   as mpl_patches      # 'as patches' does not work properly
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter             import Tk
from molass_legacy.KekLib.OurMatplotlib          import NavigationToolbar
import OurMessageBox        as MessageBox
from molass_legacy.KekLib.TkUtils                import is_low_resolution, split_geometry
from molass_legacy.KekLib.BasicUtils             import get_caller_module
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable, BlinkingFrame
from molass_legacy.UV.AbsorbanceViewer import AbsorbanceViewer
from PlotAnnotation         import add_flow_change_annotation
from molass_legacy._MOLASS.SerialSettings         import get_setting, set_setting, get_xray_picking
from DevSettings            import get_dev_setting
from molass_legacy._MOLASS.SerialSettings         import get_setting, set_setting
from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker
from molass_legacy.SerialAnalyzer.ElutionCurve import MIN_HALF_WIDTH

DRAW_MAPPING_RANGES         = True
SUGGEST_DISCOMFORT_BY_TEXT  = True
FIGURE_WIDTH_POSITIONING    = 465
VALLEY_BOUNDARY_COLOR = "gray"
from .SingleComponent import SCI_BOUNDARIES

diff_btn_texts = [ 'Hide difference', 'Show difference', 'Show abs(difference)' ]

class ElutionMapperPlotter:
    def __init__( self, dialog, body_frame, sd, mapper=None, anim_mode=False ):

        self.mplt_ge_2_2 = matplotlib.__version__ >= '2.2'
        self.dialog = dialog
        self.logger = dialog.logger
        self.uv_j0 = sd.uv_j0
        self.xr_j0 = sd.xr_j0
        self.developer_mode     = False
        self.mapper             = mapper
        self.set_range_info(mapper)
        self.anim_mode          = anim_mode
        self.absorbance_picking = get_setting( 'absorbance_picking' )
        self.intensity_picking  = get_xray_picking()
        self.using_xray_conc = get_setting('use_xray_conc') == 1

        self.mapping_show_mode  = 'locally'
        self.in_range_adjustment = False    # should be changed depending on self.initial_std_diff
        self.popup_menu = None
        self.last_click_event = None
        uv_size = len(mapper.a_vector)
        self.x1 = self.uv_j0 + np.arange(uv_size)
        xray_size = len(mapper.x_vector)
        self.x2 = self.xr_j0 + np.arange(xray_size)
        self.x3 = self.x2

        self.hiresolution       =  get_dev_setting( 'hiresolution' )
        if is_low_resolution():
            figsize=( 16, 8 ) if self.developer_mode else ( 16, 4.4 )
        else:
            if self.hiresolution:
                figsize=( 23, 8 )
            else:
                figsize=( 18, 9 ) if self.developer_mode else ( 18, 5.5 )

        self.fig = fig = plt.figure( figsize=figsize )
        ax1 = self.fig.add_subplot( 131 )
        ax2 = self.fig.add_subplot( 132 )
        ax3 = self.fig.add_subplot( 133 )
        self.axes = [ ax1, ax2, ax3 ]

        cframe = Tk.Frame( body_frame )
        cframe.pack()
        tframe = Tk.Frame( body_frame )
        tframe.pack(fill=Tk.X)

        if self.hiresolution:
            for ax in self.axes:
                ax.tick_params( labelsize=16 )

        self.mpl_canvas = FigureCanvasTkAgg( self.fig, cframe )
        self.show()
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.mpl_canvas.mpl_connect('button_press_event', self.on_mpl_button_press)

        btn_frame = Tk.Frame( tframe )
        btn_frame.pack(side=Tk.LEFT, fill=Tk.X, expand=1)
        mpl_tb_frame = Tk.Frame(btn_frame)
        mpl_tb_frame.pack(side=Tk.LEFT, padx=10)
        self.toolbar = NavigationToolbar(self.mpl_canvas, mpl_tb_frame)
        self.toolbar.update()

        other_btn_frame = Tk.Frame(btn_frame)
        other_btn_frame.pack(side=Tk.RIGHT)
        space = Tk.Frame(other_btn_frame, width=70)
        space.pack(side=Tk.RIGHT)

        self.ax3t = None
        self.diff_visible = 0
        self.show_diff_btn = Tk.Button(other_btn_frame, command=self.show_hide_difference)
        self.show_diff_btn.pack(side=Tk.RIGHT, padx=10)
        self.other_btn_frame = other_btn_frame  # for self.dialog.show_button

    def close_fig(self):
        plt.close(self.fig)

    def set_range_info(self, mapper):
        """
        mapper.mapping_ranges
        mapper.uv_peak_eval_ranges
        mapper.peak_eval_ranges
        """

        def shift_ranges(j0, ranges):
            return [ [ j0 + r for r in range_ ] for range_ in ranges ]

        ranges_ = []
        for erec in mapper.x_curve.get_default_editor_ranges():
            if len(erec) == 1:
                # temporary fix for Kosugi3a
                left, right = erec[0]
                range_ = [left, left+MIN_HALF_WIDTH, right]
            else:
                range_ = [*erec[0], erec[1][1]]
            ranges_.append(range_)
        self.target_ranges = ranges_
        self.shifted_target_ranges = shift_ranges( self.xr_j0, self.target_ranges )
        self.uv_peak_eval_ranges = shift_ranges( self.uv_j0, mapper.uv_peak_eval_ranges )
        self.peak_eval_ranges = shift_ranges( self.xr_j0, mapper.peak_eval_ranges )

    def get_target_ranges(self):
        # for Tester
        return self.target_ranges

    def get_figsize( self ):
        return self.fig.get_size_inches()*self.fig.dpi

    def get_canvas_width( self ):
        return int( self.mpl_canvas_widget.cget( 'width' ) )

    def show( self ):
        if self.mplt_ge_2_2:
            self.mpl_canvas.draw()
        else:
            self.mpl_canvas.show()

    def get_draw_params(self):
        mapper = self.mapper
        base_color = self.get_baseline_color(mapper)
        cd_colors = self.dialog.get_cd_colors()
        opt_params = mapper.opt_params
        ub_adjust  = opt_params['uv_baseline_adjust'] == 1
        xb_adjust  = opt_params['xray_baseline_adjust'] == 1
        return mapper, ub_adjust, xb_adjust, base_color, cd_colors

    def draw( self, clear=False, restrict_info=None, animation_counter=None, animation_done=None ):
        ax1, ax2= self.axes[0:2]
        if clear:
            for ax in [ ax1, ax2 ]:
                ax.cla()

        mapper = self.mapper

        self.sci_list = mapper.get_sci_list()

        if self.anim_mode:
            self.anim_counter_text = '  i=(%d, %d)' % ( mapper.opt_phase, animation_counter)
        else:
            self.anim_counter_text = ''

        if self.hiresolution:
            self.title_fonsize   = 16
        else:
            self.title_fonsize   = 14

        in_folder = get_setting('in_folder')
        self.fig.suptitle( 'Mapping status between UV and Xray elutions from ' + in_folder, fontsize=self.title_fonsize )

        self.draw_fig1(ax1)
        self.draw_fig2(ax2)

        self.the_curves1 = [["standard uv elution", self.x1, mapper.a_vector]]
        self.the_curves2 = [["standard xray elution", self.x2, mapper.x_vector]]

        ymin1, ymax1 = ax1.get_ylim()
        ymin2, ymax2 = ax2.get_ylim()

        if restrict_info is None:
            self.draw_mapped(self.axes[2], clear=clear)
        else:
            print( 'draw: restrict_info=', restrict_info )
            for x in restrict_info:
                ax2.plot( [ x, x ], [ ymin2, ymax2 ], color='yellow' )

        if self.anim_mode:
            for ax in self.axes:
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                tx = xmin*0.8 + xmax*0.2
                ty = ymin*0.5 + ymax*0.5
                text = 'Complete' if animation_done else "Phase %d" % mapper.opt_phase
                ax.text( tx, ty, text, alpha=0.1, fontsize=60 )

        self.show()
        self.fig.tight_layout()
        self.fig.subplots_adjust( top=0.87, right=0.955 )

    def draw_fig1(self, ax, draw_ranges=DRAW_MAPPING_RANGES):
        mapper, ub_adjust, _, base_color, cd_colors = self.get_draw_params()

        uv_source = 'Microfluidic Control Info' if self.mapper.get_conc_type() == 2 else 'UV absorbance'
        ax.set_title( '(a)    Elution from %s at λ=%.4g%s' % ( uv_source, self.absorbance_picking, self.anim_counter_text ), fontsize=self.title_fonsize )

        ax.plot( self.x1, mapper.a_vector, color='blue' )

        if ub_adjust:
            ax.plot( self.x1, mapper.a_base, color='pink' )
            a_base_ = mapper.a_base + mapper.a_base_adjustment
            ax.plot( self.x1, a_base_, color=base_color, alpha=0.5 )
            self.add_adjust_polygon(ax, self.uv_j0, mapper.a_base, a_base_)
        else:
            assert mapper.a_base_adjustment == 0
            ax.plot( self.x1, mapper.a_base, color=base_color )

        for i, info in enumerate(mapper.a_curve.peak_info):
            peak    = info[1]
            ax.plot( self.uv_j0 + peak, mapper.a_spline(peak), 'o', color=cd_colors[i] )

        if mapper.inv_mapped_boundaries is not None:
            for boudary in mapper.inv_mapped_boundaries:
                ax.plot( self.uv_j0 + boudary, mapper.a_spline(boudary), 'o', color=VALLEY_BOUNDARY_COLOR )

        ymin1, ymax1 = ax.get_ylim()
        ax.set_ylim( ymin1, ymax1 )
        if self.mapper.get_conc_type() < 2:
            flow_changes = mapper.flow_changes
            add_flow_change_annotation( ax, flow_changes, mapper )
        else:
            # TODO: self.helper_info seems to contain incorrect info
            pass

        if self.anim_mode:
            anim_color = 'pink' if mapper.in_opt else None
        else:
            anim_color = None
        self.add_range_patches( ax, ymin1, ymax1, self.uv_peak_eval_ranges, color=anim_color )

        if draw_ranges:
            A, B = mapper.map_params
            for k, range_ in enumerate( self.target_ranges ):
                lower, _, upper = range_
                a_lower = A*lower + B
                a_upper = A*upper + B
                for x in [ a_lower, a_upper ]:
                    x_ = self.uv_j0 + x
                    ax.plot( [ x_, x_ ], [ ymin1, ymax1 ], ':', color='black', alpha=0.2 )

    def draw_fig2(self, ax, title=True, draw_ranges=DRAW_MAPPING_RANGES, peak_no=None):
        mapper, _, xb_adjust, base_color, cd_colors = self.get_draw_params()

        if title:
            ax.set_title( '(b)    Elution from Xray scattering around q=%.3g%s' % ( self.intensity_picking, self.anim_counter_text ), fontsize=self.title_fonsize )

        ax.plot( self.x2, mapper.x_vector, color='orange' )

        if xb_adjust:
            ax.plot( self.x2, mapper.x_base, color='pink' )
            x_base_ = mapper.x_base + mapper.x_base_adjustment
            ax.plot( self.x2, x_base_, color=base_color, alpha=0.5 )
            self.add_adjust_polygon(ax, self.xr_j0, mapper.x_base, x_base_)
        else:
            # assert mapper.x_base_adjustment == 0
            ax.plot(self.x2, mapper.x_base, color=base_color)

        for i, info in enumerate( mapper.x_curve.peak_info ):
                peak    = info[1]
                ax.plot(self.xr_j0 + peak, mapper.x_spline(peak), 'o', color=cd_colors[i])

        for boudary in mapper.x_curve.boundaries:
            ax.plot(self.xr_j0 + boudary, mapper.x_spline(boudary), 'o', color=VALLEY_BOUNDARY_COLOR)

        ymin2, ymax2 = ax.get_ylim()
        ax.set_ylim( ymin2, ymax2 )

        if SUGGEST_DISCOMFORT_BY_TEXT:
            if self.dialog is not None and self.dialog.three_d_guide:
                xmin2, xmax2 = ax.get_xlim()
                tx  = xmin2 * 0.8 + xmax2 * 0.2
                ty  = ymin2 * 0.5 + ymax2 * 0.5
                ax.text(tx, ty, "See drift in 3D", alpha=0.2, fontsize=36)

        if peak_no is None:
            self.add_range_patches(ax, ymin2, ymax2, self.peak_eval_ranges)

        if draw_ranges:
            self.draw_ranges_impl(ax, mapper, peak_no)

    def draw_ranges_impl(self, ax, mapper, peak_no):
        ymin, ymax = ax.get_ylim()
        for k, range_ in enumerate( self.shifted_target_ranges ):
            lower, _, upper = range_
            for x in [lower, upper]:
                ax.plot( [ x, x ], [ ymin, ymax ], ':', color='black', alpha=0.2 )

            if peak_no is not None and k == peak_no:
                f, _, t = self.peak_eval_ranges[k]
                p = mpl_patches.Rectangle(
                        (f, ymin),      # (x,y)
                        t - f,          # width
                        ymax - ymin,    # height
                        facecolor   = 'cyan',
                        alpha       = 0.2,
                    )
                ax.add_patch( p )

    def get_baseline_color( self, mapper ):
        if self.anim_mode:
            color = 'yellow' if mapper.in_adj else 'red'
        else:
            color = 'red'
        return color

    def add_adjust_polygon(self, ax, j0, base, adj_base):
        x_end_i  = len(base) - 1
        vertices = [ (j0, base[0]), (j0+x_end_i, base[x_end_i]), (j0+x_end_i, adj_base[x_end_i]), (j0, adj_base[0]) ]
        polygon = mpl_patches.Polygon( vertices, alpha=0.3, fc='pink' )
        ax.add_patch(polygon)

    def draw_mapped(self, ax3, clear=False, title=True, draw_ranges=DRAW_MAPPING_RANGES, peak_no=None):
        if clear:
            ax3.cla()

        mapper = self.mapper

        if title:
            ax3.set_title( '(c)    Mapped elutions (Xray scattering scale)' + self.anim_counter_text, fontsize=self.title_fonsize )

        self.the_curves3 = []

        if self.anim_mode:
            ax2 = self.axes[1]
            ax3.set_xlim( ax2.get_xlim() )
            ax3.set_ylim( ax2.get_ylim() )

        if mapper.x_curve_y_adjusted is not None:
            ax3.plot( self.x3, mapper.x_curve_y_adjusted, color='orange' )
            self.the_curves3.append(["adjusted xray elution", self.x3, mapper.x_curve_y_adjusted])

            # self.the_curves3.append([])

        locally_showing = self.mapping_show_mode == 'locally'

        if locally_showing:
            show_vector = mapper.mapped_vector
        else:
            show_vector = mapper.get_uniformly_scaled_vector()

        if show_vector is not None:
            ax3.plot( self.x3, show_vector, color='blue' )
            self.the_curves3.append(["mapped uv elution", self.x3, show_vector])

        self.diff_y = show_vector - mapper.x_curve_y_adjusted
        self.show_hide_difference( update_state=False )

        cd_colors = self.dialog.get_cd_colors()
        for i, info in enumerate(mapper.x_curve.peak_info):
            peak    = info[1]
            if locally_showing:
                if mapper.mapped_spline is None:
                    peak_y  = None
                else:
                    peak_y  = mapper.mapped_spline(peak)
            else:
                peak_y  = show_vector[ int(peak + 0.5) ]
            if peak_y is not None:
                ax3.plot( self.xr_j0 + peak, peak_y, 'o', color=cd_colors[i] )

        for boudary in mapper.x_curve.boundaries:
            if locally_showing:
                if mapper.mapped_spline is None:
                    valley_y    = None
                else:
                    valley_y    = mapper.mapped_spline(boudary)
            else:
                valley_y    = show_vector[boudary]
            if valley_y is not None:
                ax3.plot( self.xr_j0 + boudary, valley_y, 'o', color=VALLEY_BOUNDARY_COLOR )

        base_color = self.get_baseline_color( mapper )
        ax3.plot( self.x3, np.zeros( len(mapper.x_curve.y) ), color=base_color )

        ymin, ymax = ax3.get_ylim()
        ax3.set_ylim( ymin, ymax )

        if peak_no is None:
            sci_text = False if self.using_xray_conc else True
            self.add_range_patches( ax3, ymin, ymax, self.peak_eval_ranges, sci_text=sci_text )

        if self.in_range_adjustment:
            xmin, xmax = ax3.get_xlim()
            xoffset = ( xmax - xmin )*0.12
            yoffset = ( ymax - ymin )*0.2

            ann_texts = [ 'S(%d)', 'P(%d)', 'E(%d)' ]
            y_  = mapper.x_curve.y
            for j, row in enumerate( self.dialog.annotation_points ):
                for k, p in enumerate( row ):
                    ax_ = p
                    ay_ = y_[p]
                    xoffset_ = xoffset * ( k - 1 ) + 1e-5       # + 1e-5 : workaroud for bug #12820
                    yoffset_ = yoffset if k == 1 else 0
                    ax3.annotate( ann_texts[k] % (p), xy=(ax_, ay_),
                                    xytext=( ax_ + xoffset_, ay_ - yoffset_ ),
                                    ha='center', va='center',
                                    arrowprops=dict( headwidth=5, width=0.5, color='black', shrink=0.05),
                                    )

        if draw_ranges:
            self.draw_ranges_impl(ax3, mapper, peak_no)

    def add_range_patches( self, ax, ymin, ymax, ranges, color=None, sci_text=False ):
        if ranges is None:
            # this case may occur in animation
            return

        if len(ranges) != len(self.sci_list):
            # as in 20180605/Backsub3
            self.mapper.logger.warning("found bug: len(ranges) != len(self.sci_list) as %d != %d", len(ranges), len(self.sci_list))

        for i, row in enumerate(ranges):
            if color is None and i < len(self.sci_list):
                sci = self.sci_list[i]
                if sci >= SCI_BOUNDARIES[1]:
                    fc  = 'cyan'
                    tfc = 'blue'
                elif sci >= SCI_BOUNDARIES[0]:
                    fc  = 'yellow'
                    tfc = 'green'
                else:
                    fc  = 'pink'
                    tfc = 'red'
            else:
                fc  = color

            f = row[0]
            t = row[2]
            p = mpl_patches.Rectangle(
                    (f, ymin),      # (x,y)
                    t - f,          # width
                    ymax - ymin,    # height
                    facecolor   = fc,
                    alpha       = 0.2,
                )
            ax.add_patch( p )
            if sci_text:
                tx = (f + t)/2
                h = (ymax - ymin) * 0.03
                ax.text( tx, h, '%d' % int(sci), ha='center', fontsize=36, color=tfc, alpha=0.2 )

    def save_the_figure( self, folder, analysis_name ):
        # print( 'save_the_figure: ', folder, analysis_name )
        filename = analysis_name.replace( 'analysis', 'figure' )
        path = os.path.join( folder, filename )
        self.fig.savefig( path )

    def show_hide_difference( self, update_state=True ):

        if update_state:
            self.diff_visible = (self.diff_visible + 1) % 3

        self.plot_difference(draw_base=True)
        self.show()

        text = diff_btn_texts[ (self.diff_visible + 1) % 3]
        self.show_diff_btn.config( text=text )

    def update_show_diff_btn_state(self):
        state = Tk.NORMAL if self.mapping_show_mode == 'locally' else Tk.DISABLED
        self.show_diff_btn.config(state=state)

    def plot_difference(self, draw_base=False, debug=False):
        if self.ax3t is not None:
            self.ax3t.remove()
            self.ax3t = None

        if debug:
            self.logger.info("plot_difference: diff_visible=%d", self.diff_visible)

        if self.diff_visible == 0:
            return

        from matplotlib.ticker import FormatStrFormatter

        fc_xe = self.mapper.get_mapped_flow_changes()

        slice_ = slice(*fc_xe)

        self.ax3t = self.axes[-1].twinx()
        amaxy = np.max(self.diff_y) * 1.1
        self.ax3t.set_ylim( -amaxy, amaxy )
        self.ax3t.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

        x = self.xr_j0 + self.mapper.x_curve.x
        y = self.diff_y if self.diff_visible == 1 else abs(self.diff_y)
        x_ = x[slice_]
        y_ = y[slice_]
        self.ax3t.bar( x_, y_, color='purple', alpha=0.3 )
        self.ax3t.plot( x[[0, -1]], [0, 0], color='pink' )

        diff_rec = ["difference", x, y]
        if len(self.the_curves3) < 3:
            self.the_curves3.append(diff_rec)
        else:
            self.the_curves3[-1] = diff_rec

        if draw_base:
            base_color = self.get_baseline_color( self.mapper )
            self.axes[-1].plot( x, np.zeros( len(x) ), color=base_color )

    def on_mpl_button_press(self, event):
        if event.xdata is None:
            return

        if event.button == 3:
            self.create_popup_menu(event)
            dialog = self.dialog
            rootx = dialog.winfo_rootx()
            rooty = dialog.winfo_rooty()
            w, h, x, y = split_geometry(self.mpl_canvas_widget.winfo_geometry())
            self.popup_menu.post(rootx + int(event.x), rooty + h - int(event.y))
            return

    def create_popup_menu(self, event):
        if self.popup_menu is None:
            self.popup_menu = Tk.Menu( self.dialog, tearoff=0 )
            self.popup_menu.add_command( label='Conc. Dependency', command=self.show_cdi_dialog )
            self.popup_menu.add_command( label='Save the curves', command=self.save_the_curves )

        i = self.axes.index(event.inaxes)
        state = Tk.DISABLED if i == 0 else Tk.NORMAL
        self.popup_menu.entryconfigure(0, state=state)
        self.last_click_event = event

    def show_cdi_dialog(self, pno=None, show_later=False):
        from Conc.CdInspection import CdInspectionDailog

        if pno is None:
            event = self.last_click_event
            i = self.axes.index(event.inaxes)
            assert i > 0
            x = event.xdata - self.xr_j0
            min_dist = None
            pno = None
            range_ = None
            for k, rec in enumerate(self.mapper.x_curve.peak_info):
                dist = abs(rec[1] - x)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    pno = k
                    range_ = (rec[0], rec[2])
        else:
            i = 1
            rec = self.mapper.x_curve.peak_info[pno]
            range_ = (rec[0], rec[2])

        sd = self.dialog.mapper.baseline_corrected_copy
        if sd is None:
            sd = self.dialog.serial_data
        else:
            print('using corrected sd')
        q = sd.qvector
        f, t = range_
        eslice = slice(f, t+1)
        M = sd.intensity_array[eslice,:,1].T
        E = sd.intensity_array[eslice,:,2].T
        c = self.mapper.x_curve.y[eslice]
        c_ = c/np.max(c)
        C = np.array([c_])
        xray_scale = sd.get_xray_scale()
        self.cdi_dialog = CdInspectionDailog(self.dialog, self.dialog, M, E, C, q, slice(f+self.xr_j0, t+self.xr_j0),
                                                xray_scale=xray_scale, plotter_info=[self, i, pno])
        if show_later:
            pass
        else:
            self.cdi_dialog.show()

    def save_the_curves(self):
        from CurveSaverDialog import CurveSaverDialog

        the_curves = None
        event = self.last_click_event
        k = int(event.x/FIGURE_WIDTH_POSITIONING)
        print("save_the_curves", k, event.x)
        the_curves = [self.the_curves1, self.the_curves2, self.the_curves3][k]

        curve_list = []
        for k, rec in enumerate(the_curves):
            label, x, y = rec
            curve = np.array([x, y]).T
            curve_list.append([label, curve, label + '.dat'])

        save_folder = '/'.join( [get_setting('analysis_folder'), 'mapping'] )

        dialog = CurveSaverDialog(self.dialog, curve_list, save_folder)
        w, h, x, y = split_geometry(self.dialog.geometry())
        x_ = x + 200 + FIGURE_WIDTH_POSITIONING * k
        dialog.geometry("+%d+%d" % (x_,  y+400))
        dialog.show()
