"""

    OutlineFigure.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF

"""
import os
import sys
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
from molass_legacy.KekLib.OurTkinter import Tk
from molass_legacy.KekLib.TkUtils import split_geometry
from molass_legacy._MOLASS.SerialSettings import get_setting, get_xray_picking

ADD_TOOLBAR         = False
LEGEND_FONTSIZE     = 10
LEGEND_FONTSIZE_LARGE = 14

class OutlineFigure(Tk.Frame):
    def __init__(self, parent, dialog, *args, **kwargs):
        self.parent = parent
        self.dialog = dialog
        self.high_dpi = dialog.high_dpi
        self.text_font = 22 if self.high_dpi else 16
        self.use_average = kwargs.pop('use_average', False)
        self.file_info_table = kwargs.pop('file_info_table', None)
        figsize = kwargs.pop('figsize', (4,5))
        self.enable_usable_limit = get_setting('enable_usable_limit')
        self.absorbance_picking = get_setting('absorbance_picking')
        self.averaged_elution = None
        self.Rg = None
        self.xray_elution_restrict = None
        self.xray_angle_restrict = None
        self.uv_elution_restrict = None

        self.logger = logging.getLogger( __name__ )

        Tk.Frame.__init__(self, parent)

        self.fig = plt.figure( figsize=figsize )
        self.mpl_canvas = FigureCanvasTkAgg( self.fig, self )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.mpl_canvas.mpl_connect('button_press_event', self.on_mpl_button_press)

        if ADD_TOOLBAR:
            self.min_toolbar = Tk.Label(self)
            self.min_toolbar.pack()
            self.mpl_canvas.mpl_connect('motion_notify_event', self.on_mpl_motion_notify)

        gs = gridspec.GridSpec(2, 1)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax1.set_axis_off()
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax2.set_axis_off()

        self.fig.tight_layout(pad=0, w_pad=0, h_pad=0, rect=[-0.05, 0, 1, 1])
        self.popup_menu = None

    def create_popup_menu( self ):
        # TODO: unify this menu and Tools menu
        from molass_legacy._MOLASS.Version import is_developing_version
        if self.popup_menu is None:
            self.popup_menu = Tk.Menu( self, tearoff=0 )
            self.popup_menu.add_command( label='Data Range Trimming', command=self.show_datarange_dialog )
            self.popup_menu.add_command( label="Remove Abnormal Points", command=self.show_abnomality_check_dialog, state=Tk.DISABLED )
            if is_developing_version():
                self.popup_menu.add_command(label='Save the upper figure', command=self.save_the_upper_figure)
                self.popup_menu.add_command(label='Save the XR elution curve', command=lambda: self.save_the_elution_curve(0))
                self.popup_menu.add_command(label='Save the UV elution curve', command=lambda: self.save_the_elution_curve(1))

    def clear_figures(self):
        for ax in [self.ax1, self.ax2]:
            ax.cla()
            ax.set_axis_off()

        self.mpl_canvas.draw()

    def draw_guide_message( self, message ):
        ax1 = self.ax1
        ax1.cla()
        ax1.set_axis_off()
        ax1.text(0.5, 0.5, message, ha='center', fontsize=self.text_font, alpha=0.2)
        self.mpl_canvas.draw()

    def set_data(self, pre_recog):
        self.pre_recog = pre_recog
        serial_data = pre_recog.get_pre_recog_copy()
        self.serial_data = serial_data
        self.similarity = pre_recog.cs
        self.averaged_elution = serial_data.ivector
        self.elution_size = len(serial_data.ivector)
        exact_index = ( serial_data.xray_slice.start + serial_data.xray_slice.stop ) //2
        self.exact_elution = serial_data.intensity_array[:,exact_index,1]
        self.q_ = serial_data.qvector
        self.default_eno = serial_data.xray_curve.primary_peak_i
        self.selected = self.default_eno
        self.xray_array = serial_data.intensity_array
        self.angle_size = self.xray_array.shape[1]
        self.usable_limit = serial_data.get_usable_q_limit()
        self.pre_rg = pre_recog.pre_rg
        use_xray_conc  = get_setting('use_xray_conc')
        use_mtd_conc  = get_setting('use_mtd_conc')
        if use_xray_conc == 0 and use_mtd_conc == 0:
            self.mapped_x = self.similarity.get_extended_x()
            self.mapped_y = self.similarity.get_uniformly_mapped_a_curve(x=self.mapped_x)
        else:
            self.mapped_x = None
            self.mapped_y = None

    def update_elution_curve(self):
        from bisect import bisect_right
        serial_data = self.serial_data
        # print(__name__, 'serial_data=', id(serial_data))
        q = get_xray_picking()
        exact_index = bisect_right(serial_data.qvector, q)
        self.exact_elution = serial_data.intensity_array[:,exact_index,1]
        self.draw_figure1()
        self.mpl_canvas.draw()

    def draw_figure(self, selected=None):
        # print('draw_figure', selected)
        if selected is None:
            selected = self.default_eno

        self.selected = selected
        self.get_restrict_info()
        self.draw_figure1()
        self.draw_figure2()
        self.draw_rg_text()
        self.mpl_canvas.draw()

    def get_restrict_info(self):
        uv_restrict_list = get_setting('uv_restrict_list')
        if uv_restrict_list is None:
            self.uv_elution_restrict = None
        else:
            rec = uv_restrict_list[0]
            self.uv_elution_restrict = rec if rec is None else rec.get_safe_object()

        xr_restrict_list = get_setting('xr_restrict_list')
        if xr_restrict_list is None:
            self.xray_elution_restrict = None
            self.xray_angle_restrict = None
        else:
            rec = xr_restrict_list[0]
            self.xray_elution_restrict = rec if rec is None else rec.get_safe_object()
            rec = xr_restrict_list[1]
            self.xray_angle_restrict = rec if rec is None else rec.get_safe_object()

    def draw_figure1(self):
        if self.averaged_elution is None:
            # before set_data?
            return

        self.draw_outline_elution(self.ax1)

    def draw_outline_elution(self, ax1):
        ax1.cla()
        # ax1.set_title("Xray elution curve")
        ax1.axes.get_yaxis().set_ticks([])

        if self.mapped_y is not None:
            ax1.plot( self.mapped_x, self.mapped_y, ':', color='blue', label='UV at wavelength=%.3g' % self.absorbance_picking )

        if self.use_average:
            label = 'average around Q=%.2g' % get_xray_picking()
            elution_y = self.averaged_elution
        else:
            label = 'Xray at Q=%.2g' % get_xray_picking()
            elution_y = self.exact_elution
        ax1.plot( elution_y, color='orange', label=label )
        ax1.plot( self.selected, elution_y[self.selected], 'o', color='yellow', markersize=5 )
        xmin, xmax = ax1.get_xlim()
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax)

        if self.uv_elution_restrict is not None:
            flag, lower, stop, _ = self.uv_elution_restrict
            uv_lower, uv_stop = [ self.similarity.inverse_int_value(j) for j in [lower, stop] ]

            if self.xray_elution_restrict is None:
                color = 'purple'
                alpha = 0.5
                lower, stop = uv_lower, uv_stop
            else:
                flag, xr_lower, xr_stop, _ = self.xray_elution_restrict
                color = 'black'
                alpha = 1.0
                # to avoid misunderstanding that the range is not appropriate
                # (may be afraid of being too wide as in 20230304)
                lower = max(uv_lower, xr_lower)
                stop = min(uv_stop, xr_stop)

            for x_ in [lower, stop-1 ]:
                ax1.plot( [x_, x_], [ymin, ymax], ':', color=color, alpha=alpha )

        fontsize = LEGEND_FONTSIZE_LARGE if self.high_dpi else LEGEND_FONTSIZE
        ax1.legend(fontsize=fontsize)

    def draw_figure2(self):
        ax2 = self.ax2
        ax2.cla()
        ax2.axes.get_yaxis().set_ticks([])
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
        if self.use_average:
            label = 'Log average near elution No. %d' % self.selected
            scattering_y = self.pre_rg.get_scattering_y(self.selected, slice(None, None))
        else:
            label = 'Log $I$ at elution No. %d' % self.selected
            scattering_y = self.xray_array[self.selected, :, 1]

        ax2.plot(self.q_, np.log10(scattering_y), label=label)

        if self.xray_angle_restrict is not None:
            ymin, ymax = ax2.get_ylim()
            ax2.set_ylim(ymin, ymax)
            flag, start, stop, _ = self.xray_angle_restrict
            for i in [start, stop - 1]:
                if i is None:
                    continue
                x_ = self.q_[i]
                ax2.plot( [x_, x_], [ymin, ymax], ':', color='black' )

        fontsize = LEGEND_FONTSIZE_LARGE if self.high_dpi else LEGEND_FONTSIZE
        ax2.legend(loc='upper right', fontsize=fontsize)

    def format_func(self, value, tick_number):
        return '%.2g' % (value)

    def on_mpl_button_press(self, event):
        if event.xdata is None or self.averaged_elution is None:
            return

        if event.button == 3:
            self.create_popup_menu()
            w, h, x, y = split_geometry(self.dialog.geometry())
            self.popup_menu.post(x + event.x + 30, y + h - event.y)
            return

        if event.inaxes != self.ax1:
            return

        min_d = None
        min_x = None
        for x, y in enumerate(self.averaged_elution):
            d = (x - event.xdata)**2 + (y - event.ydata)**2
            if min_d is None or d < min_d:
                min_d = d
                min_x = x
        if min_x is not None:
            # print('min_x=', min_x)
            if self.file_info_table is not None:
                # as in the case of unit test
                self.file_info_table.select_row(min_x)
            self.draw_figure(selected=min_x)

    def on_mpl_motion_notify(self, event):
        if event.xdata is None:
            text = ''
        else:
            text = 'x=%d' % event.xdata
        self.min_toolbar.config(text=text)

    def compute_rg(self):
        return self.pre_rg.compute_rg(self.selected)

    def draw_rg_text(self, recompute=True):
        ax2 = self.ax2
        xmin, xmax = ax2.get_xlim()
        ymin, ymax = ax2.get_ylim()

        if recompute:
            Rg = self.Rg = self.compute_rg()
        else:
            Rg = self.Rg
        Rg_ = 'None' if Rg is None else '%.1f' % Rg
        qmax_i = len(self.q_)-1 if self.xray_angle_restrict is None else self.xray_angle_restrict.stop - 1
        qmRg = 'None' if Rg is None else '%.1f' % (Rg * self.q_[qmax_i])
        tx = xmin * 0.95 + xmax * 0.05
        ty = ymin * 0.95 + ymax * 0.05
        ax2.text(tx, ty, '$R_g$=%s\n$Q_{max} \\times R_g$=%s' % (Rg_, qmRg), alpha=0.2, fontsize=self.text_font )

        cx, cy = self.q_[qmax_i], ymin
        xoffset = (xmax - xmin) * 0.05
        yoffset = (ymax - ymin) * 0.15

        if qmax_i/len(self.q_) < 0.75:
            xoffset_ = xoffset
            ha = 'left'
        else:
            xoffset_ = -xoffset
            ha = 'right'

        ax2.annotate( '$Q_{max}$', xy=(cx, cy ),
                    xytext=( cx + xoffset_, cy + yoffset ), alpha=0.5,
                    arrowprops=dict( headwidth=3, headlength=8, width=0.5, color='black', alpha=0.5),
                    ha=ha, va='center', fontsize=self.text_font
                    )

    def show_datarange_dialog(self):
        print('show_datarange_dialog')
        from molass_legacy.Trimming import DataRangeDialog
        dialog = DataRangeDialog(self.dialog, self.pre_recog)
        dialog.show()
        self.dialog.draw_figure()

    def show_abnomality_check_dialog(self):
        pass

    def save_the_figure( self, folder, analysis_name ):
        from DataUtils import cut_upper_folders
        in_folder = cut_upper_folders(get_setting('in_folder'))
        self.fig.suptitle(in_folder)
        self.fig.subplots_adjust(top=0.9)

        # print( 'save_the_figure: ', folder, analysis_name )
        filename = analysis_name.replace( 'analysis', 'figure' )
        path = os.path.join( folder, filename )
        self.fig.savefig( path )

    def save_the_upper_figure(self):
        from importlib import reload
        import Trimming.OutlineFigureSaver
        reload(Trimming.OutlineFigureSaver)
        from molass_legacy.Trimming.OutlineFigureSaver import save_the_upper_figure_impl
        save_the_upper_figure_impl(self)

    def save_the_elution_curve(self, type):
        from importlib import reload
        import Trimming.OutlineFigureSaver
        reload(Trimming.OutlineFigureSaver)
        from molass_legacy.Trimming.OutlineFigureSaver import save_the_elution_curve_impl
        save_the_elution_curve_impl(self, type)
