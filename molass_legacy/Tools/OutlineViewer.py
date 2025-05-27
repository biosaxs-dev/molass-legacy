"""
    OutlineViewer.py

    Copyright (c) 2018-2023, SAXS Team, KEK-PF
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, SpanSelector
from matplotlib.patches import Polygon
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import get_color, NavigationToolbar
from molass_legacy.KekLib.TkUtils import is_low_resolution
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

USE_PLOT_SURFACE = False

def format_coord(x, y):
    return 'x=%.4f    y=%.4f' % (x, y)

def using_normal_conc():
    return get_setting('use_xray_conc') == 0 and get_setting('use_mtd_conc') == 0

class OutlineViewer3dFrame(Tk.Frame):
    def __init__(self, parent, parent_obj, figsize, title, x, y, data, curve_info):
        self.in_folder = get_setting('in_folder')
        self.showing_xray = title.find('Xray') >= 0
        self.using_normal_conc = using_normal_conc()
        Tk.Frame.__init__(self, parent)
        cframe = Tk.Frame(self)
        cframe.pack()
        self.fig = plt.figure( figsize=figsize )
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

        if self.showing_xray or self.using_normal_conc:
            self.ax = self.fig.add_subplot(111, projection='3d')

            if title is not None:
                self.ax.set_title(title, y=1.1)
            if data is None:
                x = np.linspace(0, 1, 10)
                self.ax.plot(x, x, x)
            else:
                self.draw_3d(x, y, data, curve_info)
        else:
            self.ax = self.fig.gca()
            self.ax.text(0.5, 0.5, "Not available", ha='center', fontsize=40, alpha=0.3)
            self.ax.set_axis_off()
        self.mpl_canvas.draw()

    def close(self):
        plt.close(self.fig)

    def draw_3d(self, x, y, data, curve_info):
        self.fig.suptitle(self.in_folder)
        color = get_color(0)

        if USE_PLOT_SURFACE:
            from MatrixData import simple_plot_3d
            simple_plot_3d(self.ax, data)
        else:
            skip_cycle = data.shape[0]//100
            for i in range(data.shape[0]):
                if i % skip_cycle != 0:
                    continue
                y_   = y
                x_   = np.ones(len(y))*x[i]
                z_   = data[i,:]
                self.ax.plot(x_, y_, z_, color=color, alpha=0.3)

        if curve_info is not None:
            for k, info in enumerate(curve_info):
                pos, values, c = info[0:3]
                linestyle = info[3] if len(info) > 3 else None
                force_eview = info[4] if len(info) > 4 else False   # too tricky, better refactor
                if k % 2 == 0 or force_eview:
                    x_ = np.ones(len(values))*x[pos]
                    y_ = y
                    z_ = values
                else:
                    x_ = x
                    y_ = np.ones(len(values))*y[pos]
                    z_ = values
                self.ax.plot(x_, y_, z_, color=c, linestyle=linestyle)

        pos = plt.axes([0.8, 0.05, 0.1, 0.04])
        self.reset_view_btn = Button(pos, 'Reset View')
        self.reset_view_btn.on_clicked(self.reset_view)

    def reset_view(self, event):
        self.ax.view_init()
        self.mpl_canvas.draw()

class OutlineViewer2dFrame(Tk.Frame):
    def __init__(self, parent, parent_obj, figsize, title, x, y, data, color, opos, ocolor, restrict_info,
                sub_curve_info=None, pre_recog=None, ecurve=None, opos_sub=None, disable_selection=False,
                changeable_yscale=False, bridge_slice=None, toolbar=False, val_label=False):
        self.title = title
        self.x = x
        self.y = y
        self.data = data
        self.color = color
        self.opos = opos
        self.ocolor = ocolor
        self.restrict_info = restrict_info
        self.sub_curve_info = sub_curve_info
        self.pre_recog = pre_recog
        self.ecurve = ecurve
        self.showing_uv = pre_recog is not None
        self.opos_sub = opos_sub
        self.disable_selection = disable_selection
        self.selection_guide_text = "Drag to suggest where to select"
        self.changeable_yscale = changeable_yscale
        self.bridge_slice = bridge_slice
        self.toolbar_flag = toolbar
        self.val_label_flag = val_label
        self.max_value = len(y) - 1
        Tk.Frame.__init__(self, parent)

        self.normal_plot = title.find('Xray') >= 0 or using_normal_conc() or title.find('Elution') >= 0
        self.cbvar = None

        cframe = Tk.Frame(self)
        cframe.pack()
        self.build_cframe(cframe, figsize)

        bframe = Tk.Frame(self)
        bframe.pack(fill=Tk.X, expand=1)

        if self.toolbar_flag:
            bframe_ = self.add_toolbar(bframe)
        else:
            bframe_ = bframe

        self.build_bframe(bframe_)

    def add_toolbar(self, bframe):
        tbframe = Tk.Frame(bframe)
        tbframe.pack(side=Tk.LEFT)
        self.ax.format_coord = format_coord     # override the default to avoid fluctuating
        # self.toolbar = NavigationToolbar( self.mpl_canvas, tbframe, show_mode=False )
        self.toolbar = NavigationToolbar( self.mpl_canvas, tbframe )
        self.toolbar.update()
        bframe_ = Tk.Frame(bframe)
        bframe_.pack(side=Tk.RIGHT)
        return bframe_

    def build_bframe(self, bframe_):
        pass

    def build_cframe(self, cframe, figsize):
        self.fig, self.ax = plt.subplots( figsize=figsize )
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

        if self.normal_plot:
            if self.changeable_yscale:
                self.log_scale = 1
                self.button_ax = plt.axes([0.9, 0.9, 0.09, 0.08])
            else:
                self.log_scale = None
                self.button_ax = None

            if not self.disable_selection:
                self.span = SpanSelector(self.ax, self.onselect, 'horizontal', useblit=True,
                            props=dict(alpha=0.5))
            self.draw()
        else:
            self.ax.text(0.5, 0.5, "Not available", ha='center', fontsize=40, alpha=0.3)
            self.ax.set_axis_off()
            self.mpl_canvas.draw()

    def close(self):
        plt.close(self.fig)

    def draw(self, canvas_draw=True, restrict_info=None):
        ax = self.ax
        ax.cla()
        if self.changeable_yscale:
            self.button_ax.cla()
            if self.log_scale:
                scale = 'log'
                btn_text = 'To Linear'
            else:
                scale = 'linear'
                btn_text = 'To Log'
            ax.set_yscale(scale)
            self.toggle_scale_btn = Button(self.button_ax, btn_text)
            self.toggle_scale_btn.on_clicked(self.toggle_scale)
        else:
            pass

        ax.set_title(self.title)
        ax.plot(self.x, self.y, color=self.color, linewidth=3)
        if self.sub_curve_info is not None:
            i_sub, y_sub, c_sub, s_sub = self.sub_curve_info
            ax.plot(self.x, y_sub, color=c_sub, linestyle=s_sub, linewidth=1)

        ax.set_ylim(ax.get_ylim())

        if self.bridge_slice is None:
            ox = self.x[self.opos]
            oy = self.y[self.opos]
            ax.plot([ox, ox], [0, oy], color=self.ocolor, linewidth=3, alpha=0.5)
        else:
            start   = self.bridge_slice.start
            stop    = self.bridge_slice.stop + 1  # include stop
            slice_  = slice( start, stop )
            x_  = self.x[slice_]
            y_  = self.y[slice_]
            ymin, ymax = ax.get_ylim()
            verts = [ (x_[0], ymin) ] + list(zip(x_, y_)) + [(x_[-1], ymin)]
            poly = Polygon(verts, facecolor=self.ocolor, alpha=0.5 )
            ax.add_patch(poly)

        if self.opos_sub is not None:
            ox = self.x[self.opos_sub]
            oy = self.y[self.opos_sub]
            ax.plot([ox, ox], [0, oy], color=self.ocolor, linestyle=':', alpha=0.5)

        if not self.disable_selection:
            self.draw_selection(ax, restrict_info)

        if self.showing_uv:
            fc_list = self.pre_recog.flowchange.get_real_flow_changes()
            to_plot = []
            for fc in fc_list:
                if fc is not None:
                    to_plot.append(fc)
            ax.plot(to_plot, self.ecurve.spline(to_plot), 'o', color='cyan')
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xoffset = (xmax - xmin)*0.1
            yoffset = (ymax - ymin)*0.15
            for k, fc in enumerate(fc_list):
                if fc is None:
                    continue

                fx = fc
                fy = self.ecurve.spline(fx)
                signed_scale = -1.5 if k == 0 else 0.5
                ax.annotate( "flow change", xy=( fx, fy ),
                    xytext=( fx + signed_scale*xoffset, fy + yoffset ), alpha=1,
                    arrowprops=dict( headwidth=5, width=1, color='black', shrink=0 ),
                    )

        self.fig.tight_layout()
        if canvas_draw:
            self.mpl_canvas.draw()

    def onselect(self, xmin, xmax):
        pass

    def draw_selection(ax, restrict_info):
        pass

    def toggle_scale(self, event):
        self.log_scale = 1 - self.log_scale
        self.toggle_scale_btn = None
        self.draw()

class OutlineViewerFrame(Tk.Frame):
    def __init__(self, parent, dialog, titles, x, y, data, epos, ecurve, el_color, restrict_list,
                    toggle_info, disable_x_selection=False,
                    averaged_ridge=False,
                    changeable_yscale=False, sub_curve_info=None, pre_recog=None,
                    x_axis_ends=None,
                    bridge_slice=None,
                    sub_curve_info_list=None,   # overriding sub_curve_info
                    ):
        # print('x.shape=', x.shape, 'y.shape=', y.shape, 'data.shape=', data.shape)
        self.dialog = dialog
        self.data = data
        self.x  = x
        self.y  = y

        self.epos = epos
        self.ecurve = ecurve
        self.el_color = el_color
        self.restrict_list = restrict_list
        self.toggle_info = toggle_info
        self.using_xray_conc = get_setting('use_xray_conc') == 1
        self.using_normal_conc = using_normal_conc()

        Tk.Frame.__init__(self, parent)
        self.build_body(titles, disable_x_selection, averaged_ridge, changeable_yscale, sub_curve_info, pre_recog, bridge_slice, sub_curve_info_list)

    def get_frame_classes(self):
        return OutlineViewer3dFrame, OutlineViewer2dFrame

    def build_body(self, titles, disable_x_selection, averaged_ridge, changeable_yscale, sub_curve_info, pre_recog, bridge_slice, sub_curve_info_list=None):
        class_3d, class_2d = self.get_frame_classes()

        body_frame = Tk.Frame(self)
        body_frame.pack(padx=10)

        W_frame = Tk.Frame(body_frame)
        W_frame.pack(side=Tk.LEFT, padx=5)
        self.W_frame = W_frame
        Wupper_frame = Tk.Frame(W_frame)
        Wupper_frame.pack()
        Wlower_frame = Tk.Frame(W_frame)
        Wlower_frame.pack()

        state = Tk.DISABLED if self.using_xray_conc else Tk.NORMAL
        self.toggle_btn = Tk.Button(Wlower_frame, text=self.toggle_info[0], command=self.toggle_info[1], state=state)
        self.toggle_btn.pack(side=Tk.RIGHT, padx=10)

        E_frame = Tk.Frame(body_frame)
        E_frame.pack(side=Tk.RIGHT, padx=5)
        Eupper_frame = Tk.Frame(E_frame)
        Eupper_frame.pack()
        Elower_frame = Tk.Frame(E_frame)
        Elower_frame.pack()

        if is_low_resolution():
            figsize1 = ( 8, 7 )
            figsize2 = ( 8, 3 )
        else:
            figsize1 = ( 10, 8 )
            figsize2 = ( 10, 4 )

        ppi = self.ecurve.primary_peak_i
        if averaged_ridge:
            pp_ridge_y = self.get_averaged_ridge(ppi)
        else:
            pp_ridge_y = self.data[:,ppi]

        ridge_color = 'green'
        curve_info = [(self.epos, self.ecurve.y, self.el_color), (ppi, pp_ridge_y, ridge_color)]

        if sub_curve_info_list is None:
            if sub_curve_info is not None:
                sub_curve_info_list = [sub_curve_info]      # assure backward compatibility

        if sub_curve_info_list is not None:
            curve_info += sub_curve_info_list

        canvas1 = class_3d(Wupper_frame, self, figsize1, titles[0], self.x, self.y, self.data, curve_info)
        canvas1.pack()

        e_restrict = None if self.restrict_list is None else self.restrict_list[0]
        canvas2 = class_2d(Eupper_frame, self, figsize2, titles[1], self.y, self.ecurve.y, self.data, self.el_color, ppi, ridge_color, e_restrict,
                                        sub_curve_info=sub_curve_info, pre_recog=pre_recog, ecurve=self.ecurve,
                                        toolbar=False)  # TODO: toolbar=True
        canvas2.pack()

        opos_sub = None if sub_curve_info is None else sub_curve_info[0]
        r_restrict = None if self.restrict_list is None else self.restrict_list[1]
        canvas3 = class_2d(Elower_frame, self, figsize2, titles[2], self.x, pp_ridge_y, self.data, ridge_color, self.epos, self.el_color, r_restrict,
                                opos_sub=opos_sub,
                                disable_selection=disable_x_selection,
                                changeable_yscale=changeable_yscale,
                                bridge_slice=bridge_slice,
                                toolbar=True, val_label=True)
        canvas3.pack()
        self.canvases = []
        for k, c in enumerate([canvas1, canvas2, canvas3]):
            if self.using_normal_conc or k == 1:
                c.mpl_canvas.draw()
            self.canvases.append(c)

    def close(self):
        for canvas in self.canvases:
            canvas.close()

    def get_averaged_ridge(self, ppi):
        start = max(0, ppi - 2)
        stop  = min(self.data.shape[1], ppi + 2)
        y = np.average(self.data[:, start:stop], axis=1)
        return y

    def get_info_list(self):
        info_list = []
        for c in self.canvases[1:]:
            info_list.append(c.get_restrict_info())
        return info_list

    def forget_left_frame(self):
        self.W_frame.pack_forget()

    def pack_left_frame(self):
        self.W_frame.pack()

    def disable_toggle_button(self):
        self.toggle_btn.config(state=Tk.DISABLED)

class OutlineViewerDialog(Dialog):
    def __init__(self, parent, serial_data):
        self.logger = logging.getLogger( __name__ )
        self.serial_data = serial_data
        self.using_normal_conc = using_normal_conc()
        self.using_xray_conc = get_setting('use_xray_conc') == 1
        self.xray_only = False
        Dialog.__init__(self, parent, "Outline Viewer", visible=False)

    def get_frame_class(self):
        return OutlineViewerFrame

    def get_bridge_slice(self):
        return self.serial_data.xray_slice

    def body(self, frame):
        frame_class = self.get_frame_class()

        serial_data = self.serial_data
        self.frames = []

        titles = ["Xray Scattering Data", "Elution Curve in Xray", "Xray Scattering Curve"]
        x   = serial_data.qvector
        y   = serial_data.jvector
        xray_slice = self.serial_data.xray_slice
        epos = (xray_slice.start + xray_slice.stop)//2
        ecurve  = serial_data.get_xray_curve()
        data = serial_data.intensity_array[:, :, 1].T
        el_color = 'orange'
        xr_restrict_list = get_setting('xr_restrict_list')
        bridge_slice = self.get_bridge_slice()

        f = frame_class(frame, self, titles, x, y, data, epos, ecurve, el_color, xr_restrict_list,
                            ["Show UV", self.toggle], changeable_yscale=True, averaged_ridge=False,
                            bridge_slice=bridge_slice)
        f.pack()
        self.frames.append(f)

        if self.xray_only:
            f = Tk.Frame(frame)     # dummy (never be apparent)
        else:
            titles = ["UV Absorbance Data", "Elution Curve in UV", "UV Absorbance Curve"]
            absorbance = serial_data.absorbance
            data = absorbance.data
            x   = absorbance.wl_vector
            y   = np.arange(data.shape[1])
            absorbance = serial_data.absorbance
            ecurve  = absorbance.a_curve
            epos    = absorbance.index    # this line must be executed after get_uv_curve()
            el_color = 'blue'
            uv_restrict_list = get_setting('uv_restrict_list')

            sub_curve_info = None
            sub_curve_info_list = None
            if self.using_normal_conc:
                # task: make it easier to give these specifications
                from molass_legacy.UV.PlainCurveUtils import get_flat_info
                w, i, ev = get_flat_info(self.serial_data)
                sub_curve_info_list = [ (absorbance.index_sub, absorbance.a_vector_sub, 'blue', ':'),
                                        (i, ev, 'cyan', ':', True),
                                        ]

            x_axis_ends=serial_data.absorbance.get_wave_len_ends()
            print('x_axis_ends=', x_axis_ends)
            f = frame_class(frame, self, titles, x, y, data, epos, ecurve, el_color, uv_restrict_list,
                                ["Show Xray", self.toggle], disable_x_selection=True,
                                sub_curve_info=sub_curve_info,
                                pre_recog=serial_data.pre_recog,
                                x_axis_ends=x_axis_ends,
                                sub_curve_info_list=sub_curve_info_list)
        f.pack()
        self.frames.append(f)

        self.current = 0
        self.frames[1 - self.current].pack_forget()

    def get_current_frame(self):
        return self.frames[self.current]

    def get_opposite_frame(self):
        return self.frames[1 - self.current]

    def show(self):
        self._show()

    def toggle(self):
        self.current = 1 - self.current
        self.frames[1 - self.current].pack_forget()
        self.frames[self.current].pack()

    def cancel(self):
        for frame in self.frames:
            frame.close()

        Dialog.cancel(self)
