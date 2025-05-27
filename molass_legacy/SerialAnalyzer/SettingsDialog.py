"""
    SettingsDialog.py

    Copyright (c) 2016-2024, SAXS Team, KEK-PF
"""
import os
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk, Font, FileDialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
import molass_legacy.KekLib.OurMessageBox as MessageBox
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
from molass_legacy.ATSAS.AutoRg import set_exe_array, get_autorg_exe_paths
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting, ITEM_DEFAULTS, ALTERNATIVE_WEIGHTS, load_settings_dict, save_settings
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
from molass_legacy.KekLib.ListingFrame import ListingFrame
import AutorgKek.Settings   # necessary?
from molass_legacy._MOLASS.Version import get_version_string

label_gb_color_var = 'white smoke'
relief_var = Tk.GROOVE

SHOW_QUALITY_FACTOR_WEIGHTS = False
USE_SPINBOX_FOR_NUM_POINTS  = True
ENABLE_CHART_AXES_OPTIONS   = False
LATER_THAN_V2 = get_version_string().find("_MOLASS 2") >= 0

class SettingsDialog(Dialog):

    def __init__( self, parent, title ):
        self.grab = 'local'     # used in grab_set
        self.parent             = parent
        self.applied            = False
        self.temp_absorbance    = None
        self.absorbance_has_been_changed    = False
        self.linsting = None

        Dialog.__init__( self, parent, title )

    def body( self, body_frame ):   # overrides parent class method

        tk_set_icon_portable( self )

        iframe = Tk.Frame( body_frame );
        iframe.pack( expand=1, fill=Tk.BOTH, padx=40, pady=10 )

        text_entry_width    = 12
        state_show_sizable_width    = 163
        state_show_sizable_height   = 20

        grid_row = -1

        # ATSAS autorg paths ---------------------------------------------------
        grid_row += 1
        atsas_autorg_paths_label    = Tk.Label( iframe, text= 'ATSAS Version Selection' )
        atsas_autorg_paths_label.grid( row=grid_row, column=0, sticky=Tk.NW ) 

        self.linting_grid_row = grid_row
        self.iframe = iframe
        atsas_exe_paths = get_setting( 'atsas_exe_paths' )

        if len(atsas_exe_paths) == 0:
            atsas_exe_paths = get_autorg_exe_paths()

        self.refresh_atsas_exe_path_listing(atsas_exe_paths)
        grid_row += 1
        label = Tk.Label(iframe, text="Drag one of the above entries to change the reference order.")
        label.grid(row=grid_row, column=1)

        grid_row += 1
        path_update_frame = Tk.Frame(iframe)
        path_update_frame.grid(row=grid_row, column=1)
        label = Tk.Label(path_update_frame, text="Press the button right to update the above paths.")
        label.grid(row=0, column=0)
        button = Tk.Button(path_update_frame, text="Update", command=self.update_atsas_exe_paths)
        button.grid(row=0, column=1)

        # Absorbance-Intensity Mapping  ----------------------------------------
        grid_row += 1
        spacing_frame = Tk.Frame( iframe, height=20 )
        spacing_frame.grid( row=grid_row, column=0 )

        grid_row += 1
        absorbance_intensity_mapping_label = Tk.Label( iframe, text= 'Curves on the Standard Mapping Plane' )
        absorbance_intensity_mapping_label.grid( row=grid_row, column=0, sticky=Tk.W )

        grid_row += 1
        picking_label_frame = Tk.Frame( iframe )
        picking_label_frame.grid( row=grid_row, column=0, sticky=Tk.E )
        picking_value_frame = Tk.Frame( iframe )
        picking_value_frame.grid( row=grid_row, column=1, sticky=Tk.W )
        picking_row = 0

        # Picking Absorbance Wave Lenghth  -------------------------------------
        absorbance_picking_label    = Tk.Label( picking_label_frame, text= 'UV Absorbance Elution Curve:   ')
        absorbance_picking_label.grid( row=picking_row, column=0, sticky=Tk.E )

        num_points_label    = Tk.Label( picking_value_frame, text= 'averaging')
        num_points_label.grid( row=picking_row, column=0, sticky=Tk.W )

        self.num_points_abs  = Tk.IntVar()
        self.num_points_abs.set( get_setting('num_points_absorbance') )
        if USE_SPINBOX_FOR_NUM_POINTS:
            num_points_abs_entry    = Tk.Spinbox( picking_value_frame, textvariable=self.num_points_abs,  width=3, justify=Tk.CENTER,
                                        from_=1, to=15, increment=2 )
        else:
            num_points_abs_entry    = Tk.Entry( picking_value_frame, textvariable=self.num_points_abs, width=3, justify=Tk.CENTER )
        num_points_abs_entry.grid( row=picking_row, column=1, sticky=Tk.W )
        self.num_points_abs.trace( 'w', self.num_points_abs_tracer )

        around_label = Tk.Label( picking_value_frame, text= 'points around')
        around_label.grid( row=picking_row, column=2, sticky=Tk.W )

        wavelength_label= Tk.Label( picking_value_frame, text= 'picking  wave length λ₁=')
        wavelength_label.grid( row=picking_row, column=3, sticky=Tk.E )
        self.absorbance_picking     = Tk.DoubleVar()
        self.absorbance_picking.set( get_setting('absorbance_picking') )
        absorbance_picking_entry    = Tk.Entry( picking_value_frame, textvariable=self.absorbance_picking, width=8, justify=Tk.CENTER )
        absorbance_picking_entry.grid( row=picking_row, column=4, sticky=Tk.W )
        useage_label = Tk.Label( picking_value_frame, text= '(used in the analysis)' )
        useage_label.grid( row=picking_row, column=5, sticky=Tk.W, padx=5 )

        for i in range(3):
            picking_row += 1
            spacing_label = Tk.Label( picking_label_frame, text=" " )
            spacing_label.grid(row=picking_row, column=0)

        self.zero_absorbance = Tk.DoubleVar()
        self.zero_absorbance.set( get_setting('zero_absorbance') )
        wavelength_label= Tk.Label( picking_value_frame, text= 'baseline wave length λ₂=')
        wavelength_label.grid( row=picking_row, column=3, sticky=Tk.E )
        zero_absorbance_entry    = Tk.Entry( picking_value_frame, textvariable=self.zero_absorbance, width=8, justify=Tk.CENTER )
        zero_absorbance_entry.grid( row=picking_row, column=4, sticky=Tk.W )
        useage_label = Tk.Label( picking_value_frame, text= '(used in the analysis)' )
        useage_label.grid( row=picking_row, column=5, sticky=Tk.W, padx=5  )

        picking_row += 1
        self.zero_absorbance_auto = Tk.IntVar()
        self.zero_absorbance_auto.set(get_setting("zero_absorbance_auto"))
        w = Tk.Checkbutton(picking_value_frame, text="baseline wave length auto correction", variable=self.zero_absorbance_auto)
        w.grid(row=picking_row, column=3, columnspan=3, sticky=Tk.W)

        picking_row += 1
        self.absorbance_picking_sub     = Tk.DoubleVar()
        self.absorbance_picking_sub.set( get_setting('absorbance_picking_sub') )
        wavelength_label= Tk.Label( picking_value_frame, text= 'observe wave length λ₃=')
        wavelength_label.grid( row=picking_row, column=3, sticky=Tk.E )
        absorbance_picking_entry    = Tk.Entry( picking_value_frame, textvariable=self.absorbance_picking_sub, width=8, justify=Tk.CENTER )
        absorbance_picking_entry.grid( row=picking_row, column=4, sticky=Tk.W )
        useage_label = Tk.Label( picking_value_frame, text= '(for observation only)' )
        useage_label.grid( row=picking_row, column=5, sticky=Tk.W, padx=5  )

        # btn_state = Tk.DISABLED if self.parent.serial_data is None else Tk.NORMAL
        if False:
            btn_state = Tk.DISABLED
            show_fig_button = Tk.Button( iframe, text='Figure', command=self.show_absorbance_figure, state=btn_state )
            show_fig_button.grid( row=grid_row, column=1, sticky=Tk.E )

        self.absorbance_picking.trace( 'w', self.absorbance_update_tracer )
        self.zero_absorbance.trace( 'w', self.absorbance_update_tracer )
        self.zero_absorbance_auto.trace( 'w', self.absorbance_update_tracer )

        # Picking Intensity Sacttering Vector  ---------------------------------
        grid_row += 1
        picking_row += 1
        intensity_picking_label   = Tk.Label( picking_label_frame, text= 'X-ray Scattering Elution Curve:  ')
        intensity_picking_label.grid( row=picking_row, column=0, sticky=Tk.E )

        num_points_label    = Tk.Label( picking_value_frame, text= 'averaging')
        num_points_label.grid( row=picking_row, column=0, sticky=Tk.W )

        self.num_points_int  = Tk.IntVar()
        self.num_points_int.set( get_setting('num_points_intensity') )

        if USE_SPINBOX_FOR_NUM_POINTS:
            num_points_int_entry    = Tk.Spinbox( picking_value_frame, textvariable=self.num_points_int, width=3, justify=Tk.CENTER,
                                        from_=1, to=15, increment=2 )
        else:
            num_points_int_entry    = Tk.Entry( picking_value_frame, textvariable=self.num_points_int, width=3, justify=Tk.CENTER )
        num_points_int_entry.grid( row=picking_row, column=1, sticky=Tk.W )
        self.num_points_int.trace('w', self.num_points_int_tracer)

        around_label    = Tk.Label( picking_value_frame, text= 'points around')
        around_label.grid( row=picking_row, column=2, sticky=Tk.W )

        qlabel= Tk.Label( picking_value_frame, text= 'scattering vector q=')
        qlabel.grid( row=picking_row, column=3, sticky=Tk.E )

        self.intensity_picking    = Tk.DoubleVar()
        self.intensity_picking.set( get_setting('intensity_picking') )
        intensity_picking_entry   = Tk.Entry( picking_value_frame, textvariable=self.intensity_picking, width=8, justify=Tk.CENTER )
        intensity_picking_entry.grid( row=picking_row, column=4, sticky=Tk.W )

        # Default Angle Range Start -----------------------------------------
        grid_row += 1
        spacing_frame = Tk.Frame( iframe, height=20 )
        spacing_frame.grid( row=grid_row, column=0 )

        grid_row += 1
        angular_range_start_label = Tk.Label(iframe, text= 'Guinier Limit in Small Angle Data Trimming')
        angular_range_start_label.grid( row=grid_row, column=0, columnspan=2, sticky=Tk.W )

        grid_row += 1
        self.cut_before_guinier = Tk.IntVar()
        self.cut_before_guinier.set( get_setting('cut_before_guinier') )
        texts = [   '(1) Minimum of the measured Q range',
                    '(2) Minimum of the Guinier Q range',
                    '(3) Extended Limit beyond the Guinier Q range',
                ]
        for k, text in enumerate(texts):
            grid_row += 1
            rb_frame = Tk.Frame( iframe )
            rb_frame.grid( row=grid_row, column=0, columnspan=2, sticky=Tk.W, padx=30 )
            rb = Tk.Radiobutton( rb_frame, text=text, variable=self.cut_before_guinier, value=k )
            rb.grid( row=0, column=0, sticky=Tk.W + Tk.N )
            if k == 2:
                limit_prob_frame = Tk.Frame(rb_frame)
                limit_prob_frame.grid( row=0, column=1, padx=30)
                limit_prob_label = Tk.Label(limit_prob_frame, text="Acceptable Rg-consistency: ")
                limit_prob_label.grid( row=0, column=0 )
                self.acceptable_rg_consist = Tk.DoubleVar()
                self.acceptable_rg_consist.set( get_setting('acceptable_rg_consist') )
                limit_prob_entry = Tk.Spinbox(limit_prob_frame, textvariable=self.acceptable_rg_consist, width=6, justify=Tk.CENTER,
                                        from_=0, to=1, increment=0.1)
                limit_prob_entry.grid( row=0, column=1, sticky=Tk.W )
                guide_label_min = Tk.Label(limit_prob_frame, text="Setting 0 here implies the above option (1)")
                guide_label_min.grid( row=1, column=1, padx=5 )
                guide_label_max = Tk.Label(limit_prob_frame, text="Setting 1 here implies the above option (2)")
                guide_label_max.grid( row=2, column=1, padx=5 )

        # Keeping Temporary Folders
        grid_row += 1
        spacing_frame = Tk.Frame(iframe, height=20)
        spacing_frame.grid(row=grid_row, column=0)

        grid_row += 1
        saving_temporary_folders_label = Tk.Label(iframe, text="Keeping Temporary Folders")
        saving_temporary_folders_label.grid(row=grid_row, column=0, columnspan=2, sticky=Tk.W)

        grid_row += 1
        cb_frame = Tk.Frame( iframe )
        cb_frame.grid(row=grid_row, column=0, columnspan=2, sticky=Tk.W, padx=30)

        self.keep_tempfolder_averaged = Tk.IntVar()
        self.keep_tempfolder_averaged.set(get_setting("keep_tempfolder_averaged"))

        w = Tk.Checkbutton(cb_frame, text='keep "averaged" folder used to run ALMERGE', variable=self.keep_tempfolder_averaged)
        w.grid(row=0, column=0, sticky=Tk.W)

        # Default Font of Result Excel Books
        grid_row += 1
        spacing_frame = Tk.Frame(iframe, height=20)
        spacing_frame.grid(row=grid_row, column=0)

        grid_row += 1
        label = Tk.Label(iframe, text="Default Font of Result Excel Books")
        label.grid(row=grid_row, column=0, columnspan=2, sticky=Tk.W)

        grid_row += 1
        self.report_default_font = Tk.StringVar()
        self.report_default_font.set(get_setting("report_default_font"))
        self.font_box = ttk.Combobox(master=iframe, values=["Arial", "Calibri", "Times New Roman", "ＭＳ Ｐゴシック"],
                                        textvariable=self.report_default_font, width=16)
        self.font_box.grid(row=grid_row, column=0, sticky=Tk.W, padx=30)

        # Quality Factor Weights
        factor_names = []
        if SHOW_QUALITY_FACTOR_WEIGHTS:
            # Quality Factor Weighting ---------------------------------------------
            grid_row += 1
            spacing_frame = Tk.Frame( iframe, height=20 )
            spacing_frame.grid( row=grid_row, column=0 )

            grid_row += 1
            quality_factor_weights_label = Tk.Label( iframe, text= 'Quality Factor Weighting' )
            quality_factor_weights_label.grid( row=grid_row, column=0, sticky=Tk.W )

            factor_names += [ 'Basic Qualilty', 'Positive Score', 'Rg Stdev Score', 'Left q*Rg Score', 'End Consistency' ]
            # quality_weighting = Settings.get_setting( 'quality_weighting' )
            quality_weighting = get_setting( 'quality_weighting' )
            self.quality_factors = []
            for i, name in enumerate( factor_names ):
                grid_row += 1
                factor_label = Tk.Label( iframe, text=name )
                factor_label.grid( row=grid_row, column=0, sticky=Tk.E, padx=10 )
                quality_factor = Tk.DoubleVar()
                self.quality_factors.append( quality_factor )
                quality_factor.set( quality_weighting[i] )
                if i == 2:
                    quality_factor_entry = Tk.Frame( iframe )
                    entry_widget = Tk.Entry( quality_factor_entry, textvariable=quality_factor, width=8, justify=Tk.CENTER, state=Tk.DISABLED )
                    entry_widget.grid( row=0, column=0, sticky=Tk.W )
                    default_weights_button = Tk.Button( quality_factor_entry, text="Default Weights", command=self.default_weights, state=Tk.DISABLED )
                    default_weights_button.grid( row=0, column=1, sticky=Tk.W, padx=80 )
                else:
                    quality_factor_entry = Tk.Entry( iframe, textvariable=quality_factor, width=8, justify=Tk.CENTER, state=Tk.DISABLED )
                quality_factor_entry.grid( row=grid_row, column=1, sticky=Tk.W )

        # Chart axis direction for descending side --------------------------
        if ENABLE_CHART_AXES_OPTIONS:
            grid_row += 1
            spacing_frame = Tk.Frame( iframe, height=20 )
            spacing_frame.grid( row=grid_row, column=0 )

            grid_row += 1
            axis_direction_desc_label = Tk.Label( iframe, text= 'Chart axes direction for descending sides' )
            axis_direction_desc_label.grid( row=grid_row, column=0, sticky=Tk.W )

            self.axis_direction_desc = Tk.IntVar()
            self.axis_direction_desc.set( get_setting( 'axis_direction_desc' ) )
            grid_row += 1

            tail_text = ' order of concentration'
            for i, t in enumerate( [ 'Descending'+ tail_text, 'Ascending' + tail_text ] ):
                b = Tk.Radiobutton( iframe, text=t,
                            variable=self.axis_direction_desc, value=i )
                b.grid( row=grid_row+i, column=0, sticky=Tk.W,  padx=30  )

        # Reset to defaults button
        grid_row += len(factor_names) + 4
        spacing_frame = Tk.Frame( iframe, height=20 )
        spacing_frame.grid( row=grid_row, column=0 )

        grid_row += 1
        reset_button = Tk.Button( iframe, text="Reset to Defaults", command=self.reset_to_defaults )
        reset_button.grid( row=grid_row, column=1, sticky=Tk.E )

        # global grab cannot be set befor windows is 'viewable'
        # and this happen in mainloop after this function returns
        # Thus, it is needed to delay grab setting of an interval
        # long enough to make sure that the window has been made
        # 'viewable'
        if self.grab == 'global':
            self.after(100, self.grab_set_global )
        else:
            pass # local grab is set by parent class constructor

        # self.resizable(width=False, height=False)

    def absorbance_update_tracer( self, *args ):
        print( 'absorbance updated' )
        self.absorbance_has_been_changed = True

    def spinbox_tracer(self, var):
        try:
            n = var.get()
        except:
            return

        if n % 2 == 0:
            if n > 5:
                n_ = n - 1
            else:
                n_ = n + 1
            var.set(n_)

    def num_points_abs_tracer( self, *args ):
        self.spinbox_tracer(self.num_points_abs)

    def num_points_int_tracer( self, *args ):
        self.spinbox_tracer(self.num_points_int)

    def reset_to_defaults( self ):
        self.num_points_abs.set(            ITEM_DEFAULTS['num_points_absorbance'] )
        self.absorbance_picking.set(        ITEM_DEFAULTS['absorbance_picking'] )
        self.zero_absorbance.set(           ITEM_DEFAULTS['zero_absorbance'] )
        self.zero_absorbance_auto.set(      ITEM_DEFAULTS['zero_absorbance_auto'] )
        self.absorbance_picking_sub.set(    ITEM_DEFAULTS['absorbance_picking_sub'] )
        self.num_points_int.set(            ITEM_DEFAULTS['num_points_intensity'] )
        self.intensity_picking.set(         ITEM_DEFAULTS['intensity_picking'] )
        if ENABLE_CHART_AXES_OPTIONS:
            self.axis_direction_desc.set(   ITEM_DEFAULTS['axis_direction_desc'] )
        self.cut_before_guinier.set(        ITEM_DEFAULTS['cut_before_guinier'] )
        self.acceptable_rg_consist.set(     ITEM_DEFAULTS['acceptable_rg_consist'] )
        self.keep_tempfolder_averaged.set(  ITEM_DEFAULTS['keep_tempfolder_averaged'] )
        self.report_default_font.set(       ITEM_DEFAULTS['report_default_font'] )

    def buttonbox(self):
        '''
        override standard buttonbox.
        add "Default" button
        '''

        box = Tk.Frame(self)

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        """
        w = Tk.Button(box, text="Default", width=10, command=self.reset_to_default)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        """

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack( pady=10 )

    def refresh_atsas_exe_path_listing(self, atsas_exe_paths):
        if self.linsting is not None:
            self.linsting.destroy()

        grid_row = self.linting_grid_row
        self.linsting = ListingFrame(self.iframe, numbering=True)
        self.linsting.grid(row=grid_row, column=1)

        self.path_var_array = []
        num_path_entries = max( 2, len( atsas_exe_paths ) )
        for i in range(num_path_entries):
            if i < len( atsas_exe_paths ):
                path = atsas_exe_paths[i]
            else:
                path = ''
            var = Tk.StringVar()
            var.set(path.replace("\\autorg.exe", ""))
            self.path_var_array.append( var )
            folder_entry = FolderEntry(self.linsting, textvariable=var, slimbutton=True, width=80)
            self.linsting.insert(var, folder_entry)

        self.linsting.update_bind_geometry()

    def update_atsas_exe_paths(self):
        atsas_exe_paths = get_autorg_exe_paths()
        self.refresh_atsas_exe_path_listing(atsas_exe_paths)

    def show_absorbance_figure( self ):
        from molass_legacy.UV.AbsorbancePlot import show_absorbance_figure_util
        show_absorbance_figure_util( self )

    def apply( self ):  # overrides parent class method
        print( "ok. apply" )

        atsas_exe_paths = []
        for var in self.linsting.get_variables():
            path = var.get()
            atsas_exe_paths.append( path + r'\autorg.exe' )

        ok_ = True

        try:
            absorbance_picking  = self.absorbance_picking.get()
            zero_absorbance     = self.zero_absorbance.get()
            zero_absorbance_auto    = self.zero_absorbance_auto.get()
            absorbance_picking_sub  = self.absorbance_picking_sub.get()
            intensity_picking   = self.intensity_picking.get()
            num_points_abs      = self.num_points_abs.get()
            num_points_int      = self.num_points_int.get()
            if SHOW_QUALITY_FACTOR_WEIGHTS:
                weights = []
                for factor_var in self.quality_factors:
                    weights.append( factor_var.get() )

        except:
            ok_ = False
            error_msg_param = [ 'Value Format Error', 'There exits an ill-formatted value.' ]

        if ok_ and num_points_abs > 0 and num_points_int > 0:
            pass
        else:
            ok_ = False
            error_msg_param = [ 'Value Range Error', 'Number of points must be positive.' ]

        if ok_ and absorbance_picking > 0 and intensity_picking > 0:
            pass
        else:
            ok_ = False
            error_msg_param = [ 'Value Range Error', 'Picking parameter must be positive.' ]

        if SHOW_QUALITY_FACTOR_WEIGHTS:
            if ok_:
                sum_w = np.sum( weights )
                if abs( sum_w - 1 ) < 1e-10:
                    pass
                else:
                    ok_ = False
                    error_msg_param = [ 'Value Range Error', 'Sum of the quality factors must be 1.' ]

        if not ok_:
            MessageBox.showerror(
                error_msg_param[0],
                error_msg_param[1],
                parent=self,
                )
            self.applied    = None
            # setting this to None causes retry in the parent
            return

        set_exe_array(paths=atsas_exe_paths)
        set_setting( 'absorbance_picking',      absorbance_picking )
        set_setting( 'zero_absorbance',         zero_absorbance )
        set_setting( 'zero_absorbance_auto',    zero_absorbance_auto )
        set_setting( 'absorbance_picking_sub',  absorbance_picking_sub )
        set_setting( 'intensity_picking',       intensity_picking )
        set_setting( 'num_points_absorbance',   num_points_abs )
        set_setting( 'num_points_intensity',    num_points_int )
        set_setting( 'cut_before_guinier',      self.cut_before_guinier.get() )
        set_setting( 'acceptable_rg_consist',   self.acceptable_rg_consist.get() )
        set_setting( 'keep_tempfolder_averaged',   self.keep_tempfolder_averaged.get() )
        set_setting( 'report_default_font',     self.report_default_font.get() )

        if SHOW_QUALITY_FACTOR_WEIGHTS:
            Settings.set_setting( 'quality_weighting',  weights )

        if ENABLE_CHART_AXES_OPTIONS:
            set_setting( 'axis_direction_desc',     self.axis_direction_desc.get() )

        if self.absorbance_has_been_changed and self.temp_absorbance is not None:
            self.parent.serial_data.absorbance = self.temp_absorbance

        save_settings()
        self.applied    = True

    def select_autorg_path( self, i ):
        dir_ = 'C:/'
        f = FileDialog.askopenfilename(
                        initialdir=dir_,
                        filetypes = [
                            ( 'Executalbe file', '*.exe' ),
                            ],
                        parent=self,
          )
        if not f:
            return

        self.path_var_array[i].set( f )

class RestoreSettingDialog( Dialog ):
    def __init__( self, parent, title ):
        self.grab = 'local'     # used in grab_set
        self.parent             = parent
        self.applied            = False

        Dialog.__init__( self, parent, title )

    def body( self, body_frame ):   # overrides parent class method

        tk_set_icon_portable( self )

        grid_row = 0

        result_folder_label  = Tk.Label( body_frame, text= 'Result Folder: ' )
        result_folder_label.grid( row=grid_row, column=0, sticky=Tk.E )

        self.result_folder = Tk.StringVar()
        result_folder = self.parent.an_folder.get() + '/' + self.parent.analysis_name.get()
        self.result_folder.set( result_folder )
        self.result_folder_entry = FolderEntry( body_frame, textvariable=self.result_folder, width=70,
                                            on_entry_cb=self.on_entry_in_folder )
        self.result_folder_entry.grid( row=grid_row, column=1, sticky=Tk.W )

        if self.grab == 'global':
            self.after(100, self.grab_set_global )
        else:
            pass # local grab is set by parent class constructor

    def on_entry_in_folder( self ):
        pass

    def validate( self ):
        pickle_file = self.result_folder.get() + '/.save/serial_settings.dump'
        if os.path.exists( pickle_file ):
            self.result_folder_entry.config( fg='black' )
            yn = MessageBox.askyesno( 'Comfirmation',
                            'This action restores settings from the previous results,'
                            '\nand your current input settings will be lost.'
                            '\nAre you sure to proceed?',
                            parent=self
                            )
            if yn:
                self.restore_setting( pickle_file )
                return 1
            else:
                return 0
        else:
            self.result_folder_entry.config( fg='red' )
            MessageBox.showerror( 'Input Error',
                            pickle_file + '\ndoes not exist.' + '\nEnter a folder which really contains results.',
                            parent=self )
            return 0

    def apply( self ):  # overrides parent class method
        print( "ok. apply" )
        self.applied = True

    def restore_setting( self, pickle_file ):
        dict_ = load_settings_dict( pickle_file )
        parent = self.parent
        parent.in_folder.set( dict_.get( 'in_folder' ) )
        parent.uv_folder.set( dict_.get( 'uv_folder' ) )
        parent.uv_file.set( dict_.get( 'uv_file' ) )
        parent.an_folder.set( dict_.get( 'an_folder' ) )
        # TODO: what to restore
        # move this method to parent
