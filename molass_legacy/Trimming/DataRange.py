"""
    DataRange.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from bisect import bisect_right
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.Tools.OutlineViewer import (format_coord, using_normal_conc,
                OutlineViewerDialog,
                OutlineViewerFrame, OutlineViewer3dFrame, OutlineViewer2dFrame)
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting, reset_setting
from .TrimmingInfo import TrimmingInfo

class DataRangeDialog(OutlineViewerDialog):
    def __init__(self, parent, pre_recog):
        self.parent = parent
        self.pre_recog = pre_recog
        self.logger = logging.getLogger( __name__ )
        self.serial_data = pre_recog.get_pre_recog_copy()
        self.pre_rg = pre_recog.pre_rg
        self.cs = pre_recog.cs
        self.using_normal_conc = using_normal_conc()
        self.using_xray_conc = get_setting('use_xray_conc') == 1
        self.xray_only = False
        Dialog.__init__(self, parent, "Data Range Trimming", visible=False)

    def get_frame_class(self):
        return DataRestrictorFrame

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack(fill=Tk.X, padx=50, pady=5)
        for k in range(3):
            box.columnconfigure(k, weight=1)

        self.at_a_glance_btn = Tk.Button(box, text="At a Glance", command=self.show_it_at_a_glance)
        self.at_a_glance_btn.grid(row=0, column=0)

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.grid(row=0, column=1)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.grid(row=0, column=2)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def show_it_at_a_glance(self):
        from .TrimmingResult import TrimmingResultDialog
        dialog = TrimmingResultDialog(self, self.pre_recog)
        dialog.show()

    def apply(self):
        xr_restrict_list  = self.frames[0].get_info_list()
        if self.using_xray_conc:
            uv_restrict_list    = xr_restrict_list
        else:
            uv_restrict_list    = self.frames[1].get_info_list()
        self.logger.info('restriction info for xray has been set as ' + str(xr_restrict_list) )
        self.logger.info('restriction info for uv has been set as ' + str(uv_restrict_list) )
        set_setting('xr_restrict_list' , xr_restrict_list )
        set_setting('uv_restrict_list' , uv_restrict_list )
        set_setting('manually_trimmed' , True )

        reset_setting('zx_penalty_matrix')
        reset_setting('ignore_all_bqs')
        reset_setting('ignore_bq_list')
        reset_setting('range_editor_info')
        reset_setting('editor_ranges')
        reset_setting('editor_model')
        reset_setting('decomp_editor_info')

    def exec_callback(self, callback, path):
        callback(self, path)

class DataRestrictorFrame(OutlineViewerFrame):
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
        # self.build_body(titles, disable_x_selection, averaged_ridge, changeable_yscale, sub_curve_info, pre_recog, bridge_slice, sub_curve_info_list)
        # why self.disable_selection?
        self.build_body(titles, False, averaged_ridge, changeable_yscale, sub_curve_info, pre_recog, bridge_slice, sub_curve_info_list)

    def get_frame_classes(self):
        return OutlineViewer3dFrame, DataRestrictor2dFrame

class DataRestrictor2dFrame(OutlineViewer2dFrame):
    """
        Note that this class is used in the following four ways.
        UV elution          X elution
        UV spectral         X angular
    """
    def __init__(self, parent, parent_obj, figsize, title, x, y, data, color, opos, ocolor, restrict_info,
                sub_curve_info=None, pre_recog=None, ecurve=None, opos_sub=None, disable_selection=False,
                changeable_yscale=False,
                bridge_slice=None,
                toolbar=False, val_label=False):
        self.parent_obj = parent_obj
        self.logger = parent_obj.dialog.logger
        self.sd = parent_obj.dialog.serial_data
        self.title = title
        self.drawing_uv = title.find("UV") >= 0         # unify with showing_uv?
        self.drawing_elution = title.find("Elution") >= 0
        self.x = x
        self.y = y
        self.data = data
        self.color = color
        self.opos = opos
        self.ocolor = ocolor
        self.restrict_info = restrict_info
        self.sub_curve_info = sub_curve_info
        self.showing_X = title.find("X") >= 0 
        self.pre_recog = pre_recog
        self.ecurve = ecurve
        self.showing_uv = pre_recog is not None
        self.opos_sub = opos_sub
        self.disable_selection = disable_selection
        self.selection_guide_text = "Drag to suggest where to trim."
        self.changeable_yscale = changeable_yscale
        self.bridge_slice = bridge_slice
        self.toolbar_flag = toolbar
        self.val_label_flag = val_label
        self.max_value = len(y) - 1
        self.popup_menu = None
        self.inspector = None
        Tk.Frame.__init__(self, parent)

        self.normal_plot = title.find('Xray') >= 0 or using_normal_conc() or title.find('Elution') >= 0
        self.cbvar = None

        cframe = Tk.Frame(self)
        cframe.pack()
        self.build_cframe(cframe, figsize)
        self.add_popup_menu_bind()

        bframe = Tk.Frame(self)
        bframe.pack(fill=Tk.X, expand=1)

        if self.toolbar_flag:
            bframe_ = self.add_toolbar(bframe)
        else:
            bframe_ = bframe

        self.build_bframe(bframe_)

        if self.drawing_uv and not self.drawing_elution:
           self.draw_flat_wavelength()

        self.gra_dialog = None

    def onselect(self, xmin, xmax):
        width = xmax - xmin
        if width < self.span_ack_min_width:
            # should better be handled in SpanSelector
            # print("this is not a span selection. ignore width=", width)
            return

        i = min(len(self.x)-1, bisect_right(self.x, xmin))
        j = min(len(self.x)-1, bisect_right(self.x, xmax))

        prev_lower = self.lower.get()
        prev_upper = self.upper.get()

        self.lower.set( i )
        self.upper.set( j )
        self.cbvar.set( 1 )     # this line must be executed after the above lines execution

        self.logger.info("range changed from (%d, %d) to (%d, %d)", prev_lower, prev_upper, i, j)

    def draw_selection(self, ax, restrict_info):
        self.draw_selection_guide(ax)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
        if restrict_info is None:
            if self.cbvar is not None and self.cbvar.get() == 1:
                try:
                    restrict_info = [ var.get() for var in self.ivars  ]
                except:
                    # ?
                    restrict_info = None
        if restrict_info is not None:
            for i in restrict_info:
                x_ = self.x[i]
                ax.plot( [x_, x_], [ymin, ymax], color='yellow' )

    def draw_selection_guide(self, ax):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        tx = (xmin + xmax)/2
        if self.changeable_yscale and self.log_scale:
            # ty = np.power(10, (-2 + np.log10(ymax))/2 )
            ty = np.power(10, np.log10(ymin)*0.1 + np.log10(ymax)*0.9)
        else:
            ty = (ymin + ymax)/2
        ax.text(tx, ty, self.selection_guide_text, ha="center", va="center", alpha=0.2, fontsize=30)

    def build_bframe(self, bframe):

        btn_frame = Tk.Frame(bframe)
        # btn_frame.pack(side=Tk.RIGHT, padx=50 if self.val_label_flag else 110)
        btn_frame.pack(side=Tk.RIGHT, padx=50)

        cb_state = Tk.DISABLED if self.disable_selection or not self.normal_plot else Tk.NORMAL

        if self.restrict_info is None:
            cbvar_init = 0
            start   = None
            stop    = None
        else:
            cbvar_init = self.restrict_info.flag
            start   = self.restrict_info.start
            stop    = self.restrict_info.stop
        lower_init = 0 if start is None else start
        upper_init = self.max_value if stop is None else stop - 1

        self.logger.info("debug info: len(x)=%d, lower_init=%d, upper_init=%d", len(self.x), lower_init, upper_init)
        if upper_init > self.max_value:
            # as in 20170209/OAGIwyatt_01
            # the reason of which remains to be investigated
            self.logger.warning("forced upper_init(%d) to be equal to max_value(%d)", upper_init, self.max_value)
            upper_init = self.max_value

        self.cbvar  = Tk.IntVar()
        self.cbvar.set( cbvar_init )
        cb  = Tk.Checkbutton( btn_frame, text="trim",
                                variable=self.cbvar, state=cb_state )
        cb.grid( row=0, column=0 )

        self.cbvar.trace( 'w', lambda *args: self.cb_tracer() )

        self.cb_depentents = []
        self.lower  = Tk.IntVar()
        self.lower.set( lower_init )
        self.upper  = Tk.IntVar()
        self.upper.set( upper_init )
        self.ivars  = [ self.lower, self.upper ]

        if self.val_label_flag:
            self.val_vars = []

        colbase = 1
        num_parts = 3 if self.val_label_flag else 2
        space_count = 0
        for i, rec in enumerate( [ [ 'start', lower_init], [ 'end' , upper_init ] ] ):
            text, value = rec

            item_base = i*num_parts
            space = Tk.Frame( btn_frame, width=10 )
            space.grid( row=0, column=colbase + space_count + item_base, sticky=Tk.W )
            space_count += 1

            btn_label   = Tk.Label( btn_frame, text=text )
            btn_label.grid( row=0, column=colbase + item_base + space_count, sticky=Tk.W )

            self.cb_depentents.append( btn_label )
            entry   = Tk.Spinbox( btn_frame, textvariable=self.ivars[i],
                        from_=0, to=self.max_value, increment=1, 
                        justify=Tk.CENTER, width=6 )
            entry.grid( row=0, column=colbase + item_base + 1 + space_count, sticky=Tk.W )
            self.cb_depentents.append( entry )

            if self.val_label_flag:
                val_var = Tk.StringVar()
                val_var.set( '%.3g' % self.x[value])
                self.val_vars.append( val_var )
                val_label   = Tk.Label( btn_frame, textvariable=val_var )
                val_label.grid( row=0, column=colbase + item_base + 2 + space_count, sticky=Tk.W )
                self.cb_depentents.append( val_label )

            self.ivars[i].trace( 'w', self.spinbox_tracer )

        self.cb_tracer()    # for intial state

        if self.ecurve is not None:
            last_column = colbase + num_parts*2 + space_count
            data_name = "UV" if self.showing_X else "Xray"
            self.from_opposite_side_btn = Tk.Button(btn_frame, text="Set from " + data_name, command=self.set_from_opposite_data_info)
            self.from_opposite_side_btn.grid(row=0, column=last_column + 1, padx=10)

    def cb_tracer(self):
        cb_val = self.cbvar.get()
        state   = Tk.NORMAL if cb_val== 1 else Tk.DISABLED
        for j, w in enumerate(self.cb_depentents ):
            w.config( state=state )
            if cb_val == 1 and j == 1:
                w.focus_force()     # set focus to the spinbox
        if self.normal_plot:
            self.draw_restrict_lines( state == Tk.NORMAL )

    def set_from_opposite_data_info(self):
        dialog = self.parent_obj.dialog

        cs = dialog.cs
        A, B = cs.slope, cs.intercept

        if self.showing_X:
            A_, B_ = 1/A, -B/A
        else:
            A_, B_ = A, B

        def convert(x):
            return A_*x + B_

        oframe = dialog.get_opposite_frame()
        ocanvas = oframe.canvases[1]
        r_info = ocanvas.get_restrict_info(fill=True)

        xmin = convert(r_info.start)
        xmax = convert(r_info.stop - 1)
        self.onselect(xmin, xmax)

    def draw_restrict_lines(self, restrict):
        if restrict:
            try:
                restrict_info   = [ var.get() for var in self.ivars ]
                if self.val_label_flag:
                    for k, i in enumerate(restrict_info):
                        self.val_vars[k].set( '%.3g' % self.x[i] )
            except:
                # when entry value is deleted
                return
        else:
            restrict_info   = None

        self.draw(restrict_info=restrict_info)

    def spinbox_tracer( self, *args ):
        try:
            self.draw_restrict_lines( True )
        except:
            # when the Spinbox entry text is deleted
            pass

    def get_restrict_info(self, fill=False):
        cbval   = self.cbvar.get()
        if cbval:
            lower   = self.lower.get()
            upper   = self.upper.get()
            if lower == 0 and upper == self.max_value and not fill:
                info = None
            else:
                info = TrimmingInfo(1, lower, upper+1, len(self.x))
        else:
            if fill:
                info = TrimmingInfo(1, 0, self.max_value+1, len(self.x))
            else:
                info = None
        return info

    def add_popup_menu_bind(self):
        xmin, xmax = self.ax.get_xlim()
        self.span_ack_min_width = (xmax - xmin) * 0.05
        self.mpl_canvas.mpl_connect('button_press_event', self.on_figure_click)

    def on_figure_click(self, event):
        if event.button == 1:
            return
        elif event.button == 3:
            if get_setting('enable_gr_analyzer') and event.inaxes == self.ax:
                if self.showing_X:
                    if self.ecurve is None:
                        menu_specs = [  ("Guinier Region Inspector", self.show_guinier_inspector),
                                        ("Gradual Region Analysis", self.show_gradual_analysis),
                                        ]
                    else:
                        menu_specs = [("Show Extra Info", self.show_extra_info)]
                    self.show_popup_menu(event, menu_specs)
                return

    def show_popup_menu(self, event, menu_specs):
        from molass_legacy.KekLib.TkUtils import split_geometry
        self.create_popup_menu(event, menu_specs)
        canvas = self.mpl_canvas_widget
        cx = canvas.winfo_rootx()
        cy = canvas.winfo_rooty()
        w, h, x, y = split_geometry(canvas.winfo_geometry())
        self.popup_menu.post(cx + int(event.x), cy + h - int(event.y))

    def create_popup_menu(self, event, menu_specs):
        if self.popup_menu is None:
            self.popup_menu = Tk.Menu(self, tearoff=0 )
            for spec in menu_specs:
                self.popup_menu.add_command(label=spec[0], command=spec[1])

    def show_extra_info(self):
        from .TrimmingInspection import ElutionTrimmingDialog
        parent_frame = self.parent_obj
        dialog = parent_frame.dialog
        data = parent_frame.data
        self.inspector = ElutionTrimmingDialog(dialog, dialog, data, self.ecurve, self.restrict_info)
        self.inspector.show()

    def show_guinier_inspector(self):
        from .GuinierRegion import GuinierRegionInspector
        parent_frame = self.parent_obj
        dialog = parent_frame.dialog
        data = parent_frame.data
        i_smp = self.opos       # i_smp : index of SMP - Standard Mapping Plane
        trimming_info = dialog.frames[0].get_info_list()[0]
        self.inspector = GuinierRegionInspector(dialog, dialog, data, i_smp, trimming_info.get_slice())
        self.inspector.show()

    def create_gradual_analysis(self, debug=True):
        if debug:
            from importlib import reload
            import Trimming.GradualRegionAnalysis
            reload(Trimming.GradualRegionAnalysis)
        from .GradualRegionAnalysis import GradualRegionAnalysis
        parent_frame = self.parent_obj
        dialog = parent_frame.dialog
        dialog.config(cursor='wait')
        dialog.update()
        # note that self.pre_recog can be None unlike dialog.pre_recog
        # it must be dialog.pre_recog here is stead of self.pre_recog
        trimming_info = dialog.frames[0].get_info_list()[1]     # angular slice
        gra_dialog = GradualRegionAnalysis(dialog, dialog, self.sd, trimming_info, dialog.pre_recog)
        dialog.config(cursor='')
        dialog.update()
        return gra_dialog

    def show_gradual_analysis(self):
        # separate constructor to make it possible to manipulate from the tester
        self.gra_dialog = self.create_gradual_analysis()

        self.gra_dialog.show()

    def has_gra_dialog(self):
        return self.gra_dialog is not None

    def draw_flat_wavelength(self):
        from molass_legacy.UV.PlainCurveUtils import get_flat_wavelength
        # assuming the ylim has been already set
        ax = self.ax

        ymin, ymax = ax.get_ylim()
        w = get_flat_wavelength(self.sd.lvector)
        ax.plot([w, w], [ymin, ymax], color="cyan")
