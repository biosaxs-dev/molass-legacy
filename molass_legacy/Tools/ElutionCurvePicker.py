# coding: utf-8
"""
    ElutionCurvePicker.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from bisect import bisect_right
from idlelib.tooltip import Hovertip
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from .OutlineViewer import (format_coord, using_normal_conc,
                OutlineViewerDialog,
                OutlineViewerFrame, OutlineViewer3dFrame, OutlineViewer2dFrame)
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

toggle_btn_texts = ["Show 3D View", "Hide 3D View"]
width_btn_texts = ["◀", "▶"]

def float_cut(v):
    return float('%.3f' % v)

class ElutionCurvePicker(OutlineViewerDialog):
    def __init__(self, parent, pre_recog):
        self.logger = logging.getLogger( __name__ )
        self.pre_recog = pre_recog
        self.serial_data = pre_recog.get_pre_recog_copy()
        self.using_normal_conc = using_normal_conc()
        self.using_xray_conc = get_setting('use_xray_conc') == 1
        self.showing_3d_view = 0
        self.xray_only = False
        self.applied = False
        Dialog.__init__(self, parent, "Elution Curve Picker", visible=False)

    def buttonbox(self):
        bframe = Tk.Frame(self)
        bframe.pack(fill=Tk.X)
        box1 = Tk.Frame(bframe)
        box1.pack(side=Tk.LEFT, anchor=Tk.W)

        self.show_3d_btn = Tk.Button(box1, text=toggle_btn_texts[self.showing_3d_view], command=self.toggle_3d_view)
        self.show_3d_btn.pack(padx=20)

        box2 = Tk.Frame(bframe)
        box2.pack(side=Tk.RIGHT, padx=300)
        super().buttonbox(box2)

    def get_frame_class(self):
        return ElutionCurvePickerFrame

    def get_bridge_slice(self):
        return self.serial_data.xray_slice

    def toggle_3d_view(self):
        for frame in self.frames:
            if self.showing_3d_view:
                frame.forget_left_frame()
            else:
                frame.pack_left_frame()
        self.showing_3d_view = 1 - self.showing_3d_view
        self.show_3d_btn.config(text=toggle_btn_texts[self.showing_3d_view])

    def apply(self):
        self.frames[0].canvases[2].apply()
        self.logger.info('elution curve has been re-picked to updage serial_data object(%s)' % str(id(self.serial_data)))
        self.applied = True

class ElutionCurvePickerFrame(OutlineViewerFrame):
    def __init__(self, parent, dialog, titles, x, y, data, epos, ecurve, el_color, restrict_list,
                    toggle_info, disable_x_selection=False,
                    averaged_ridge=False,
                    changeable_yscale=False, sub_curve_info=None, pre_recog=None,
                    x_axis_ends=None,
                    bridge_slice=None,
                    sub_curve_info_list=None):
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

        self.build_title()
        self.build_body(titles, disable_x_selection, averaged_ridge, changeable_yscale, sub_curve_info, pre_recog, bridge_slice)
        self.forget_left_frame()
        self.disable_toggle_button()

    def build_title(self):
        label = Tk.Label(self, text="Elution Curve Picking for " + get_setting('in_folder'), font=('', 18))
        label.pack(pady=20)

    def get_frame_classes(self):
        return OutlineViewer3dFrame, ElutionCurvePicker2dFrame

    def forget_3d_canvas(self):
        self.canvases[0].pack_forget()

class ElutionCurvePicker2dFrame(OutlineViewer2dFrame):
    def __init__(self, parent, parent_obj, figsize, title, x, y, data, color, opos, ocolor, restrict_info,
                sub_curve_info=None, pre_recog=None, ecurve=None, opos_sub=None, disable_selection=False,
                changeable_yscale=False,
                bridge_slice=None,
                toolbar=False, val_label=False):
        self.parent = parent
        self.parent_obj = parent_obj
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
        self.changeable_yscale = changeable_yscale
        self.bridge_slice = bridge_slice
        self.toolbar_flag = toolbar
        self.val_label_flag = val_label
        self.max_value = len(y) - 1
        Tk.Frame.__init__(self, parent)

        self.showing_elution = title.find('Elution') >= 0
        self.showing_xray = title.find('Xray') >= 0
        self.normal_plot = self.showing_xray or using_normal_conc() or self.showing_elution

        if self.showing_elution:
            self.disable_selection = True
        else:
            self.disable_selection = disable_selection

        self.selection_guide_text = "Drag to suggest where to pick"

        self.showing_small_angle = 0
        self.init_xlim = None
        self.set_small_xlim()
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

    def set_small_xlim(self):
        xmin = -0.01
        xmax = self.x[len(self.x)//4]
        self.small_xlim = (xmin, xmax)

    def get_toggle_xlim(self):
        if self.init_xlim is None:
            self.init_xlim = self.ax.get_xlim()
        return self.small_xlim if self.showing_small_angle else self.init_xlim

    def toggle_show_width(self):
        self.showing_small_angle = 1 - self.showing_small_angle
        self.set_width_adj_btn_text()
        self.draw()

    def set_width_adj_btn_text(self):
        text = width_btn_texts[self.showing_small_angle]
        self.width_adj_btn.config(text=text)

    def build_bframe(self, bframe):
        if self.showing_elution or not self.showing_xray:
            label = Tk.Label(bframe, text="")
            label.pack()
            return

        self.width_adj_btn = Tk.Button(bframe, command=self.toggle_show_width)
        self.set_width_adj_btn_text()
        self.width_adj_btn.grid(row=0, column=0, padx=5)
        Hovertip( self.width_adj_btn, 'Change Small/Wide View' )

        pick_frame = Tk.Frame(bframe)
        pick_frame.grid(row=0, column=1)

        for j, t in enumerate(["Angle(Q) Ragne", "Num Points", "Method"]):
            label = Tk.Label(pick_frame, text=t)
            label.grid(row=0, column=j)

        range_frame = Tk.Frame(pick_frame)
        range_frame.grid(row=1, column=0, padx=5)

        bridge_slice = self.bridge_slice
        start, stop = bridge_slice.start, bridge_slice.stop
        self.from_  = Tk.DoubleVar()
        self.from_.set(self.x[start])
        self.to_  = Tk.DoubleVar()
        self.to_.set(self.x[stop])
        self.dvars  = [ self.from_, self.to_ ]

        k = -1
        for i, t in enumerate(["from ", "to "]):
            if i == 1:
                k += 1
                space = Tk.Label(range_frame, width=1)
                space.grid(row=0, column=k)

            k += 1
            label = Tk.Label(range_frame, text=t)
            label.grid(row=0, column=k)

            k += 1
            entry   = Tk.Spinbox( range_frame, textvariable=self.dvars[i],
                        from_=0, to=1, increment=0.001, 
                        justify=Tk.CENTER, width=6 )
            entry.grid(row=0, column=k)

        self.num_points = Tk.IntVar()
        self.num_points.set(stop - start)
        label = Tk.Label(pick_frame, textvariable=self.num_points)
        label.grid(row=1, column=1)

        method_frame = Tk.Frame(pick_frame)
        method_frame.grid(row=1, column=2, padx=5)

        self.pick_method = Tk.IntVar()
        self.pick_method.set(get_setting('x_ecurve_pickmethod'))
        for j, t in enumerate(["Average", "Integral"]):
            rb = Tk.Radiobutton( method_frame, text=t, variable=self.pick_method, value=j)
            rb.grid(row=0, column=j)

        self.pick_method.trace('w', self.pick_method_tracer)

        self.spinbox_stack = []
        self.spinbox_stop_trace = False
        for k, dvar in enumerate(self.dvars):
            dvar.trace('w', lambda *args, k_=k:self.spinbox_tracer(k_))

    def draw(self):
        super().draw(canvas_draw=False)

        if not self.showing_elution:
            self.ax.set_xlim(self.get_toggle_xlim())

        self.mpl_canvas.draw()

    def draw_selection(self, ax, restrict_info):
        self.draw_selection_guide(ax)

    def draw_selection_guide(self, ax):
        """
        overriding the super method to adjust to the narrowed range (-0.01, 0.4)
        """
        xmin, xmax = self.get_toggle_xlim()
        ymin, ymax = ax.get_ylim()
        tx = (xmin + xmax)/2
        if self.changeable_yscale and self.log_scale:
            # ty = np.power(10, (-2 + np.log10(ymax))/2 )
            ty = np.power(10, np.log10(ymin)*0.1 + np.log10(ymax)*0.9)
        else:
            ty = (ymin + ymax)/2
        ax.text(tx, ty, self.selection_guide_text, ha="center", va="center", alpha=0.2, fontsize=30)

    def onselect(self, xmin, xmax):
        print('onselect', (xmin, xmax))
        if xmax - xmin < 1e-6:
            return

        self.spinbox_stop_trace = True
        try:
            max_i = len(self.x)-1
            start = min(max_i, bisect_right(self.x, xmin))
            j = min(max_i, bisect_right(self.x, xmax))
            stop = j+1
            self.bridge_slice = slice(start, stop)
            self.from_.set(float_cut(self.x[start]))
            self.to_.set(float_cut(self.x[j]))
            self.num_points.set(stop - start)
            self.update()
            self.draw()
        except:
            pass
        self.spinbox_stop_trace = False
        self.redraw_the_elution_canvas()

    def spinbox_tracer(self, k):
        if self.spinbox_stop_trace:
            return

        values = []
        try:
            for dvar in self.dvars:
                values.append(dvar.get())
        except:
            return

        if values[0] > values[1] - 1e-6:
            if k == 0:
                v = float_cut(values[1] - 0.001)
            else:
                v = float_cut(values[0] + 0.001)
            self.dvars[k].set(v)
            return

        self.spinbox_stack.append(values)
        self.after(500, self.delayed_select)
        self.update()

    def pick_method_tracer(self, *args):
        self.redraw_the_elution_canvas()

    def delayed_select(self):
        if len(self.spinbox_stack) == 0:
            return

        values = self.spinbox_stack.pop(-1) # get the latest
        self.spinbox_stack.clear()          # and discard other values
        self.onselect(*values)

    def redraw_the_elution_canvas(self):
        y = np.average(self.data[self.bridge_slice,:], axis=0)

        method = self.pick_method.get()
        if method == 1:
            from_ = self.from_.get()
            to_ = self.to_.get()
            span = to_ - from_
            y *= span

        elution_canvas = self.parent_obj.canvases[1]
        elution_canvas.y = y
        elution_canvas.draw()

    def apply(self):
        serial_data = self.parent_obj.dialog.serial_data
        serial_data.update_xray_elution_info(self.y, self.bridge_slice)

        set_setting('x_ecurve_pickmethod', self.pick_method.get())
        set_setting('x_ecurve_pickslice', self.bridge_slice)
        i = (self.bridge_slice.start + self.bridge_slice.stop)//2
        set_setting('x_ecurve_picking_q', self.x[i])
