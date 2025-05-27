# coding: utf-8
"""
    BackgroundMaker.py

    Copyright (c) 2019-2020, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from bisect import bisect_right
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.Tools.OutlineViewer import (format_coord, using_normal_conc,
                OutlineViewerDialog,
                OutlineViewerFrame, OutlineViewer3dFrame, OutlineViewer2dFrame)
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

toggle_btn_texts = ["Show 3D View", "Hide 3D View"]

class BackgroundMakerDialog(OutlineViewerDialog):
    def __init__(self, parent, pre_recog):
        self.logger = logging.getLogger( __name__ )

        self.pre_recog = pre_recog
        self.serial_data = pre_recog.pre_recog_copy
        self.using_normal_conc = using_normal_conc()
        self.using_xray_conc = get_setting('use_xray_conc') == 1
        self.showing_3d_view = 0
        self.applied = False
        self.xray_only = True
        Dialog.__init__(self, parent, "Background Maker", visible=False)
        self.tell_xray_frame_redraw()

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

    def tell_xray_frame_redraw(self):
        """
        note that elution_canvas.redraw_the_angular_canvas
        must not be called until the construction of this dialog is complete
        when all the xray_frame.canvases have been built.
        """
        xray_frame = self.frames[0]
        elution_canvas = xray_frame.canvases[1]
        self.after(100, elution_canvas.redraw_the_angular_canvas)

    def get_frame_class(self):
        return BackgroundMakerFrame

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
        # apply Xray elution canvas
        self.frames[0].canvases[1].apply()
        self.applied = True

    def get_the_background_info(self):
        elution_canvas = self.frames[0].canvases[1]
        angular_canvas = self.frames[0].canvases[2]
        qvector = angular_canvas.x
        intensity = angular_canvas.y
        error_matrix = self.serial_data.intensity_array[angular_canvas.bridge_slice,:,2].T
        N = error_matrix.shape[1]
        error = np.sqrt(np.sum((error_matrix/N)**2, axis=1))    # error propagation rule
        return (np.vstack([qvector, intensity, error]).T, elution_canvas.bridge_slice)

class BackgroundMakerFrame(OutlineViewerFrame):
    def __init__(self, parent, dialog, titles, x, y, data, epos, ecurve, el_color, restrict_list,
                    toggle_info, disable_x_selection=False,
                    averaged_ridge=False,
                    changeable_yscale=False, sub_curve_info=None, pre_recog=None,
                    x_axis_ends=None,
                    bridge_slice=None):
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
        label = Tk.Label(self, text="Background for " + get_setting('in_folder'), font=('', 18))
        label.pack(pady=20)

    def get_frame_classes(self):
        return OutlineViewer3dFrame, BackgroundMaker2dFrame

    def forget_3d_canvas(self):
        self.canvases[0].pack_forget()

class BackgroundMaker2dFrame(OutlineViewer2dFrame):
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
        self.serial_data = parent_obj.dialog.serial_data
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
            self.disable_selection = False
        else:
            self.disable_selection = True

        self.selection_guide_text = "Drag to suggest where to average"

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

    def get_default_elution_slice(self, width=10):
        mtd_elution = self.serial_data.mtd_elution
        if mtd_elution is None:
            start, stop = 0, 10
        else:
            start, stop = mtd_elution.propose_background_range(len(self.x), xy=(self.x, self.y))
        return slice(start, stop)

    def build_bframe(self, bframe):
        if not self.showing_elution:
            label = Tk.Label(bframe, text="")
            label.pack()
            return

        center_frame = Tk.Frame(bframe)
        center_frame.pack(pady=5)

        pick_frame = Tk.Frame(center_frame)
        pick_frame.grid(row=0, column=1)

        for j, t in enumerate(["Elution Ragne", "Num Points"]):
            label = Tk.Label(pick_frame, text=t)
            label.grid(row=0, column=j)

        range_frame = Tk.Frame(pick_frame)
        range_frame.grid(row=1, column=0, padx=5)

        elution_slice = self.get_default_elution_slice()
        self.bridge_slice = elution_slice
        start, stop = elution_slice.start, elution_slice.stop
        self.from_  = Tk.IntVar()
        self.from_.set(self.x[start])
        self.to_  = Tk.IntVar()
        self.to_.set(self.x[stop])
        self.ivars  = [ self.from_, self.to_ ]

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
            entry   = Tk.Spinbox( range_frame, textvariable=self.ivars[i],
                        from_=0, to=len(self.x)-1, increment=1, 
                        justify=Tk.CENTER, width=6 )
            entry.grid(row=0, column=k)

        self.num_points = Tk.IntVar()
        self.num_points.set(stop - start)
        label = Tk.Label(pick_frame, textvariable=self.num_points)
        label.grid(row=1, column=1)

        self.spinbox_stop_trace = False
        for k, dvar in enumerate(self.ivars):
            dvar.trace('w', lambda *args, k_=k:self.spinbox_tracer(k_))

        self.draw()
        # self.redraw_the_angular_canvas()

    def draw(self, canvas_draw=True):
        super().draw(canvas_draw=False)

        if not self.showing_elution:
            self.ax.set_xlim(self.get_toggle_xlim())

        if canvas_draw:
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
            self.from_.set(self.x[start])
            self.to_.set(self.x[j])
            self.num_points.set(stop - start)
            self.update()
            self.draw()
        except:
            if False:
                from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
                etb = ExceptionTracebacker()
                print(etb.last_lines())
        self.spinbox_stop_trace = False
        self.redraw_the_angular_canvas()

    def spinbox_tracer(self, k):
        if self.spinbox_stop_trace:
            return

        values = []
        try:
            for ivar in self.ivars:
                values.append(ivar.get())
        except:
            return

        if values[0] > values[1]:
            print('values=', values)
            if k == 0:
                v = values[1]
            else:
                v = values[0]
            self.spinbox_stop_trace = True
            self.ivars[k].set(v)
            values[k] = v
            self.update()
            self.spinbox_stop_trace = False

        start = values[0]
        stop = values[1]+1
        self.bridge_slice = slice(start, stop)
        self.num_points.set(stop - start)
        self.draw()
        self.update()

    def redraw_the_angular_canvas(self):
        y = np.average(self.data[:, self.bridge_slice], axis=1)

        angular_canvas = self.parent_obj.canvases[2]
        angular_canvas.y = y
        angular_canvas.draw(canvas_draw=False)
        angular_canvas.draw_the_bridge_suggested_curves(self.bridge_slice)
        angular_canvas.mpl_canvas.draw()

    def apply(self):
        # set_setting('xray_bg_pickslice', self.bridge_slice)
        pass

    def draw_the_bridge_suggested_curves(self, bridge_slice):
        ax = self.ax
        x = self.x
        for k in range(bridge_slice.start, bridge_slice.stop):
            y = self.data[:, k]
            ax.plot(x, y, color=self.color, alpha=0.5)
