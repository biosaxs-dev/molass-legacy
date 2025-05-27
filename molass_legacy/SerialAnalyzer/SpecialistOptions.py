"""

    ファイル名：   SpecialistOptions.py

    処理内容：

        専門家者用の設定変更ダイアログ

    Copyright (c) 2017-2023, SAXS Team, KEK-PF

"""

import os
import re
import warnings
import time

from molass_legacy.KekLib.BasicUtils             import get_filename_extension
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog, ttk, Font, FileDialog, is_empty_val, checkFolder
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
import OurMessageBox        as MessageBox
from molass_legacy._MOLASS.SerialSettings         import get_setting, set_setting, ITEM_DEFAULTS
from DevSettings            import get_dev_setting, set_dev_setting, ITEM_DEFAULTS as DEV_ITEM_DEFAULTS
from molass_legacy.KekLib.TkCustomWidgets        import FolderEntry

ENABLE_Rayleigh_Scatter_OPTION  = False
ENABLE_XRAY_PROP_CONC_SMOOTHING = False
ENABLE_REGRESSION_LOOKING_BACK  = False
ENABLE_Mapping_Animation        = False
ENABLE_DEPRECATED_OPTIONS       = False

class SpecialistOptionsDialog( Dialog ):
    def __init__( self, parent, title, grand_parent=None ):
        self.grab = 'local'     # used in grab_set
        self.parent             = parent
        self.grand_parent       = grand_parent
        self.title_             = title
        self.applied            = False
        self.temp_absorbance    = None
        self.absorbance_has_been_changed    = False

    def show( self ):
        self.parent.config( cursor='wait' )
        self.parent.update()

        Dialog.__init__(self, self.parent, self.title_ )

        self.parent.config( cursor='' )

    def body( self, body_frame ):   # overrides parent class method

        tk_set_icon_portable( self )

        secondary_padx  = 20

        iframe = Tk.Frame( body_frame );
        iframe.pack( expand=1, fill=Tk.BOTH, padx=10, pady=20 )

        grid_row = 0
        spacing_frame = Tk.Frame( iframe, width=80 )
        spacing_frame.grid( row=grid_row, column=0 )

        # ----------------------------------------------------------------------
        # Abnormal Data Handling
        # ----------------------------------------------------------------------
        grid_row += 1
        abnormal_data_label = Tk.Label( iframe, text='Abnormal Data Handling' )
        abnormal_data_label.grid( row=grid_row, column=0, sticky=Tk.W )

        # enable mapping animation
        grid_row += 1
        self.data_exclusion = Tk.IntVar()
        self.data_exclusion.set( get_setting( 'data_exclusion' ) )

        cb = Tk.Checkbutton( iframe, text="enable data exclusion indication in the Data File Table",
                                variable=self.data_exclusion  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx   )

        grid_row += 1
        spacing = Tk.Label( iframe, text='' )
        spacing.grid( row=grid_row, column=0 )

        # ----------------------------------------------------------------------
        # Concentration Mapping Option
        # ----------------------------------------------------------------------
        if ENABLE_Mapping_Animation:
            grid_row += 1
            mapping_options_label = Tk.Label( iframe, text='Concentration Mapping Option' )
            mapping_options_label.grid( row=grid_row, column=0, sticky=Tk.W )

            # enable mapping animation
            grid_row += 1
            self.enable_mapping_anim = Tk.IntVar()
            self.enable_mapping_anim.set( get_setting( 'enable_mapping_anim' ) )

            cb = Tk.Checkbutton( iframe, text="enable Mapping Animation",
                                    variable=self.enable_mapping_anim  )
            cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx   )

            grid_row += 1
            spacing = Tk.Label( iframe, text='' )
            spacing.grid( row=grid_row, column=0 )

        # ----------------------------------------------------------------------
        # Data Correction Options
        # ----------------------------------------------------------------------
        grid_row += 1
        correction_options_label = Tk.Label( iframe, text='Data Correction Options' )
        correction_options_label.grid( row=grid_row, column=0, sticky=Tk.W )

        # enable mapping animation
        grid_row += 1
        self.enable_xb_save = Tk.IntVar()
        self.enable_xb_save.set( get_dev_setting( 'enable_xb_save' ) )

        cb = Tk.Checkbutton( iframe, text="enable X-ray Scattering Baseline Save",
                                variable=self.enable_xb_save  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx   )

        self.xb_folder = Tk.StringVar()
        # analysis_folder = get_setting( 'analysis_folder' )
        _, analysis_folder = self.parent.make_analysis_folder()
        if analysis_folder is not None:
            folder_path = os.path.join( analysis_folder, 'xray_scattering_base' ).replace( '\\', '/' )
            self.xb_folder.set( folder_path )
        folder_entry = FolderEntry( iframe, textvariable=self.xb_folder, width=70,
                                            on_entry_cb=self.on_entry_xb_folder )
        folder_entry.grid( row=grid_row, column=1, sticky=Tk.W )

        grid_row += 1
        if len(self.parent.file_info_table.datafiles) > 0:
            self.filename_example   = os.path.split( self.parent.file_info_table.datafiles[0] )[-1]
            self.filename_extention = '.' + get_filename_extension( self.filename_example )
        else:
            self.filename_example   = None
            self.filename_extention = ".dat"

        postfix_frame = Tk.Frame( iframe )
        postfix_frame.grid( row=grid_row, column=1, sticky=Tk.W )
        postfix_label = Tk.Label( postfix_frame, text="filename postfix" )
        postfix_label.grid( row=0, column=0 )
        self.base_file_postfix = Tk.StringVar()
        self.base_file_postfix.set( "_base" )

        self.base_file_postfix_entry = Tk.Entry( postfix_frame, textvariable=self.base_file_postfix, width=10 )
        self.base_file_postfix_entry.grid( row=0, column=1, padx=5 )

        as_in_label = Tk.Label( postfix_frame, text="as in" )
        as_in_label.grid( row=0, column=2 )

        self.postfix_eg = Tk.StringVar()
        self.postfix_eg_update()
        eg_label = Tk.Label( postfix_frame, textvariable=self.postfix_eg )
        eg_label.grid( row=0, column=3, padx=3 )

        grid_row += 1
        spacing = Tk.Label( iframe, text='' )
        spacing.grid( row=grid_row, column=0 )

        # ----------------------------------------------------------------------
        # Elution Mapping Options
        # ----------------------------------------------------------------------
        grid_row += 1
        mapping_options_label = Tk.Label( iframe, text='Elution Mapping Options' )
        mapping_options_label.grid( row=grid_row, column=0, sticky=Tk.W )

        # Peak Mapping Only
        grid_row += 1
        self.peak_mapping_only = Tk.IntVar()
        self.peak_mapping_only.set( get_setting('peak_mapping_only') )

        cb = Tk.Checkbutton( iframe, text="do mapping with peak tops and end points only",
                                variable=self.peak_mapping_only  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx   )

        # enable LPM variations (linear, quadratic, spline)
        grid_row += 1
        self.lpm_variations = Tk.IntVar()
        self.lpm_variations.set( get_setting('lpm_variations') )

        cb = Tk.Checkbutton( iframe, text="enable LPM variations: linear, quadratic, spline",
                                variable=self.lpm_variations  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx )

        # allow angular slope in MF-baseplane
        grid_row += 1
        self.allow_angular_slope_in_mf = Tk.IntVar()
        self.allow_angular_slope_in_mf.set( get_setting('allow_angular_slope_in_mf') )

        cb = Tk.Checkbutton( iframe, text="allow angular slope in MF-baseplane  ( i.e., z = a*x + b*y + c, a != 0 )",
                                variable=self.allow_angular_slope_in_mf  )
        cb.grid( row=grid_row, column=0, columnspan=2, sticky=Tk.W, padx=secondary_padx )

        grid_row += 1
        spacing = Tk.Label( iframe, text='' )
        spacing.grid( row=grid_row, column=0 )

        # ----------------------------------------------------------------------
        # Elution Curve Modeling and Elution Decomposer Options
        # ----------------------------------------------------------------------
        grid_row += 1
        guinier_options_label = Tk.Label( iframe, text='Elution Curve Modeling and Elution Decomposer Options' )
        guinier_options_label.grid( row=grid_row, column=0, sticky=Tk.W )

        # enable Affine Transformation
        grid_row += 1
        self.enable_affine_tran = Tk.IntVar()
        self.enable_affine_tran.set( get_setting( 'enable_affine_tran' ) )

        cb = Tk.Checkbutton( iframe, text="enable Affine Transformation",
                                variable=self.enable_affine_tran  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx   )

        # enable Decomposition from molass_legacy.UV-Xray separation
        grid_row += 1
        self.decomp_from_separation = Tk.IntVar()
        self.decomp_from_separation.set( get_setting( 'decomp_from_separation' ) )

        cb = Tk.Checkbutton( iframe, text="enable Decomposition from molass_legacy.UV-Xray separation",
                                variable=self.decomp_from_separation  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx   )

        grid_row += 1
        spacing = Tk.Label( iframe, text='' )
        spacing.grid( row=grid_row, column=0 )

        # ----------------------------------------------------------------------
        # Guinier Analysis Options
        # ----------------------------------------------------------------------
        dataset_is_ready = self.parent.dataset_is_ready
        self.serial_data = self.parent.serial_data
        ranges = [ rec for rec in self.serial_data.xray_curve.peak_info ] if dataset_is_ready else []
        self.init_ranges = [ [ 1, tuple(range_) ]  for range_ in ranges ]
        self.num_peaks = len(ranges)

        grid_row += 1
        guinier_options_label = Tk.Label( iframe, text='Guinier Analysis Options' )
        guinier_options_label.grid( row=grid_row, column=0, sticky=Tk.W )

        grid_row += 1
        self.fixed_guinier_start = Tk.IntVar()
        self.fixed_guinier_start.set( get_setting( 'fixed_guinier_start' ) )

        state = Tk.NORMAL if dataset_is_ready else Tk.DISABLED
        cb = Tk.Checkbutton( iframe, text="Fix Guinier interval start point to the",
                                variable=self.fixed_guinier_start, state=state )
        cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx )

        self.guinier_start_dependents = []

        frame_ = Tk.Frame( iframe )
        frame_.grid( row=grid_row, column=1, sticky=Tk.W )

        self.guinier_start_point = Tk.IntVar()
        self.guinier_start_point.set( 0 )
        entry_ = Tk.Spinbox( frame_, textvariable=self.guinier_start_point,
                                            from_=0, to=50, increment=1, 
                                            justify=Tk.CENTER, width=6, state=Tk.NORMAL )
        entry_.pack( side=Tk.LEFT  )
        self.guinier_start_dependents.append( entry_ )

        label_ = Tk.Label( frame_, text=' -th point; looking at the scattering curve on the' )
        label_.pack( side=Tk.LEFT )
        self.guinier_start_dependents.append( label_ )

        self.guinier_fig_peak_no = Tk.IntVar()
        self.guinier_fig_peak_no.set( 0 )

        entry_ = Tk.Spinbox( frame_, textvariable=self.guinier_fig_peak_no,
                                            from_=0, to=max(1, self.num_peaks-1), increment=1, 
                                            justify=Tk.CENTER, width=6, state=Tk.NORMAL )
        entry_.pack( side=Tk.LEFT  )
        self.guinier_start_dependents.append( entry_ )

        label_ = Tk.Label( frame_, text=' -th peak' )
        label_.pack( side=Tk.LEFT )
        self.guinier_start_dependents.append( label_ )

        self.guinier_btn = Tk.Button( iframe, text="Guinier plot", command=self.guinier_plot_dialog )
        self.guinier_btn.grid( row=grid_row, column=2, sticky=Tk.W )
        self.guinier_start_dependents.append( self.guinier_btn )

        self.fixed_guinier_start_tracer()   # set initial state
        self.fixed_guinier_start.trace( 'w', self.fixed_guinier_start_tracer )

        # ----------------------------------------------------------------------
        # Extrapolation Options
        # ----------------------------------------------------------------------
        grid_row += 1
        extrapolation_options_label = Tk.Label( iframe, text='Extrapolation Options' )
        extrapolation_options_label.grid( row=grid_row, column=0, sticky=Tk.W )

        enable_new_features = get_setting('enable_new_features')

        # enable Concentration Datatype Changes
        grid_row += 1
        self.enable_conctype_change = Tk.IntVar()
        self.enable_conctype_change.set(get_setting('enable_conctype_change'))
        if enable_new_features:
            cb = Tk.Checkbutton( iframe, text="enable Concentration Datatype Changes",
                                    variable=self.enable_conctype_change  )
            cb.grid( row=grid_row, column=0, columnspan=2, sticky=Tk.W, padx=secondary_padx  )

        # Extended Concentration Dependency
        grid_row += 1
        self.extended_conc_dep = Tk.IntVar()
        self.extended_conc_dep.set( get_setting( 'extended_conc_dep' ) )
        cb = Tk.Checkbutton( iframe, text="enable Extended Concentration Dependency  ( i.e., I(q) = A(q)*c + B(q)*c² + Z(q)*c³ )",
                                variable=self.extended_conc_dep  )
        cb.grid( row=grid_row, column=0, columnspan=2, sticky=Tk.W, padx=secondary_padx )

        # Rank Variation
        grid_row += 1
        self.allow_rank_variation = Tk.IntVar()
        self.allow_rank_variation.set( get_setting( 'allow_rank_variation' ) )
        cb = Tk.Checkbutton( iframe, text="allow Rank Variation  ( i.e., no SVD denoise, additional rank allowance )",
                                variable=self.allow_rank_variation  )
        cb.grid( row=grid_row, column=0, columnspan=2, sticky=Tk.W, padx=secondary_padx  )

        # use xray-transformed concentration
        if False:
            grid_row += 1
            self.use_xray_conc = Tk.IntVar()
            self.use_xray_conc.set( get_setting( 'use_xray_conc' ) )
            self.use_xray_conc.trace( 'w', self.use_xray_conc_tracer )

            column_ = 0
            cb = Tk.Checkbutton( iframe, text="use Xray-proportional concentration",
                                    variable=self.use_xray_conc  )
            cb.grid( row=grid_row, column=column_, sticky=Tk.W, padx=secondary_padx  )

            if ENABLE_XRAY_PROP_CONC_SMOOTHING:
                column_ += 1
                self.smoothed_xray_conc = Tk.IntVar()
                self.smoothed_xray_conc.set( get_dev_setting( 'smoothed_xray_conc' ) )

                cb = Tk.Checkbutton( iframe, text="apply smoothing to the left-mentioned concentration",
                                        variable=self.smoothed_xray_conc  )
                cb.grid( row=grid_row, column=column_, sticky=Tk.W  )

            self.apply_backsub = Tk.IntVar()
            self.apply_backsub.set( get_setting( 'apply_backsub' ) )

            column_ += 1
            self.apply_backsub_cb = Tk.Checkbutton( iframe, text="apply backsub by LPM",
                                    variable=self.apply_backsub  )
            self.apply_backsub_cb.grid( row=grid_row, column=column_, sticky=Tk.W  )
            self.use_xray_conc_tracer()

        # Recompute Regression Boundary by looking back
        if ENABLE_REGRESSION_LOOKING_BACK:
            grid_row += 1
            self.recompute_regboundary = Tk.IntVar()
            self.recompute_regboundary.set( get_dev_setting( 'recompute_regboundary' ) )

            cb = Tk.Checkbutton( iframe, text="recompute Regression Boundary by looking back",
                                    variable=self.recompute_regboundary  )
            cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx  )

        # almerge analyzer
        grid_row += 1
        self.almerge_analyzer = Tk.IntVar()
        self.almerge_analyzer.set( get_setting( 'almerge_analyzer' ) )

        cb = Tk.Checkbutton( iframe, text="enable Almerge Analyzer",
                                variable=self.almerge_analyzer, state=Tk.DISABLED  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx  )

        # space
        grid_row += 1
        spacing = Tk.Label( iframe, text='' )
        spacing.grid( row=grid_row, column=0 )

        if ENABLE_DEPRECATED_OPTIONS:
            # ----------------------------------------------------------------------
            # Deprecated Extrapolation Options
            # ----------------------------------------------------------------------
            self.fixed_font = Font.Font( family="Courier", size=9 )
            extrapolation_state = Tk.NORMAL

            grid_row += 1
            extrapolation_options_label = Tk.Label( iframe, text='Deprecated Extrapolation Options' )
            extrapolation_options_label.grid( row=grid_row, column=0, sticky=Tk.W )

            # Width of Regression in Q-axis
            grid_row += 1
            angle_boundary_label = Tk.Label( iframe, text='Width of Regression in the direction of Q-axis', state=extrapolation_state )
            angle_boundary_label.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx  )

            regression_method_frame = Tk.Frame( iframe )
            regression_method_frame.grid( row=grid_row, column=1, sticky=Tk.W )

            self.regression_method_buttons = []
            self.regression_method = Tk.IntVar()
            self.regression_method.set( get_setting( 'zx_num_q_points' ) )

            b = Tk.Radiobutton( regression_method_frame, text='Five Points',
                        font=self.fixed_font,
                        variable=self.regression_method, value=5,
                        state=extrapolation_state,
                        )
            b.grid( row=0, column=0, sticky=Tk.E )
            self.regression_method_buttons.append( b )

            b = Tk.Radiobutton( regression_method_frame, text='Three Points',
                        font=self.fixed_font,
                        variable=self.regression_method, value=3,
                        state=extrapolation_state,
                        )
            b.grid( row=0, column=1, sticky=Tk.E, padx=27 )
            self.regression_method_buttons.append( b )

            b = Tk.Radiobutton( regression_method_frame, text='One Point',
                        font=self.fixed_font,
                        variable=self.regression_method, value=1,
                        state=extrapolation_state,
                        )
            b.grid( row=0, column=2, sticky=Tk.E )
            self.regression_method_buttons.append( b )

            # Regression Boundary in A(q)
            grid_row += 1
            zx_boundary_label = Tk.Label( iframe, text='Regression Boundary', state=extrapolation_state )
            zx_boundary_label.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx  )

            zx_boundary_frame = Tk.Frame( iframe )
            zx_boundary_frame.grid( row=grid_row, column=1, sticky=Tk.W )

            self.zx_boundary_method_buttons = []
            self.zx_boundary_method = Tk.StringVar()
            self.zx_boundary_method.set( get_setting( 'zx_boundary_method' ) )

            b = Tk.Radiobutton( zx_boundary_frame, text='Automatic   ',
                        font=self.fixed_font,
                        variable=self.zx_boundary_method, value='AUTO',
                        state=extrapolation_state,
                        )
            b.grid( row=0, column=0, sticky=Tk.E )
            self.zx_boundary_method_buttons.append( b )

            b = Tk.Radiobutton( zx_boundary_frame, text='Fixed at',
                        font=self.fixed_font,
                        variable=self.zx_boundary_method, value='FIXED',
                        state=extrapolation_state,
                        )
            b.grid( row=0, column=1, sticky=Tk.E, padx=20 )
            self.zx_boundary_method_buttons.append( b )

            self.zx_boundary = Tk.DoubleVar()
            self.zx_boundary.set( get_setting( 'zx_boundary' ) )
            self.zx_boundary_entry = Tk.Entry( zx_boundary_frame, textvariable=self.zx_boundary, justify=Tk.CENTER, width=6 )
            self.zx_boundary_entry.grid( row=0, column=2, sticky=Tk.W )

            b = Tk.Radiobutton( zx_boundary_frame, text='No Boundary',
                        font=self.fixed_font,
                        variable=self.zx_boundary_method, value='NO',
                        state=extrapolation_state,
                        )
            b.grid( row=0, column=3, sticky=Tk.E, padx=30 )
            self.zx_boundary_method_buttons.append( b )
            self.zx_boundary_method.trace( 'w', self.zx_boundary_method_tracer )

            # Construction Method in Wide Angle Region of A(q)
            grid_row += 1
            wide_angle_region_label = Tk.Label( iframe, text='A(q) Curve Construction in B(q)-Extinguished Region', state=extrapolation_state )
            wide_angle_region_label.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx )

            wide_angle_region_frame = Tk.Frame( iframe )
            wide_angle_region_frame.grid( row=grid_row, column=1, sticky=Tk.W )

            self.zx_build_method_buttons = []
            self.zx_build_method = Tk.StringVar()
            self.zx_build_method.set( get_setting( 'zx_build_method' ) )

            b = Tk.Radiobutton( wide_angle_region_frame, text='Use max conc. only',
                        variable=self.zx_build_method, value='MAX',
                        state=extrapolation_state,
                        )
            b.grid( row=0, column=0, sticky=Tk.E )
            self.zx_build_method_buttons.append( b )

            b = Tk.Radiobutton( wide_angle_region_frame, text='Conc-weighted regression',
                        variable=self.zx_build_method, value='REG',
                        state=extrapolation_state,
                        )
            b.grid( row=0, column=1, sticky=Tk.E, padx=2 )
            self.zx_build_method_buttons.append( b )

            # space
            grid_row += 1
            spacing = Tk.Label( iframe, text='' )
            spacing.grid( row=grid_row, column=0 )

        # ----------------------------------------------------------------------
        # Software Availability
        # ----------------------------------------------------------------------
        grid_row += 1
        label = Tk.Label( iframe, text='Software Availability' )
        label.grid( row=grid_row, column=0, sticky=Tk.W )

        grid_row += 1
        self.revoke_excel = Tk.IntVar()
        self.revoke_excel_init = get_setting('revoke_excel')
        self.revoke_excel.set(self.revoke_excel_init)

        cb = Tk.Checkbutton( iframe,
                                        text='revoke Excel Availability',
                                        variable=self.revoke_excel,
                                        state=Tk.NORMAL )
        cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx )

        grid_row += 1
        self.revoke_atsas = Tk.IntVar()
        self.revoke_atsas_init = get_setting('revoke_atsas')
        self.revoke_atsas.set(self.revoke_atsas_init)

        cb = Tk.Checkbutton( iframe,
                                        text='revoke ATSAS Availability',
                                        variable=self.revoke_atsas,
                                        state=Tk.NORMAL )
        cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx )

        grid_row += 1
        self.revoke_cuda = Tk.IntVar()
        self.revoke_cuda_init = get_setting('revoke_cuda')
        self.revoke_cuda.set(self.revoke_cuda_init)

        cb = Tk.Checkbutton( iframe,
                                        text='revoke CUDA Availability',
                                        variable=self.revoke_cuda,
                                        state=Tk.NORMAL )
        cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx )

        # space
        grid_row += 1
        spacing = Tk.Label( iframe, text='' )
        spacing.grid( row=grid_row, column=0 )

        # ----------------------------------------------------------------------
        # Devolopment/Maintenance
        # ----------------------------------------------------------------------
        grid_row += 1
        maintenance = Tk.Label( iframe, text='Maintenance' )
        maintenance.grid( row=grid_row, column=0, sticky=Tk.W )

        grid_row += 1
        self.maintenance_mode = Tk.IntVar()
        self.maintenance_mode.set( get_setting( 'maintenance_mode' ) )

        maintenance_mode_cb = Tk.Checkbutton( iframe,
                                        text='enable Maintenance Mode',
                                        variable=self.maintenance_mode,
                                        state=Tk.NORMAL )
        maintenance_mode_cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=secondary_padx )

        # ----------------------------------------------------------------------
        # Reset to Defaults button
        # ----------------------------------------------------------------------
        grid_row += 1
        reset_button = Tk.Button( iframe, text="Reset to Defaults", command=self.reset_to_defaults )
        reset_button.grid( row=grid_row, column=2, sticky=Tk.E )

        # global grab cannot be set befor windows is 'viewable'
        # and this happen in mainloop after this function returns
        # Thus, it is needed to delay grab setting of an interval
        # long enough to make sure that the window has been made
        # 'viewable'
        if self.grab == 'global':
            self.after(100, self.grab_set_global )
        else:
            pass # local grab is set by parent class constructor

        self.update()

    def buttonbox( self ):
        box = Tk.Frame( self )
        box.pack( pady=10 )
        Dialog.buttonbox( self, frame=box )

    def on_entry_xb_folder( self ):
        pass

    def postfix_eg_update( self ):
        try:
            filename = self.filename_example.replace( self.filename_extention, self.base_file_postfix.get() + self.filename_extention )
            self.postfix_eg.set( filename )
        except:
            pass

    def use_xray_conc_tracer( self, *args ):
        if self.use_xray_conc.get() == 0:
            state = Tk.DISABLED
        else:
            state = Tk.NORMAL
        self.apply_backsub_cb.config( state=state )

    def show_absorbance_figure( self ):
        if ENABLE_Rayleigh_Scatter_OPTION:
            set_setting( 'consider_scatter',    self.consider_scatter.get() )
        set_setting( 'scatter_picking',     self.zero_absorbance.get() )

        from molass_legacy.UV.AbsorbancePlot import show_absorbance_figure_util
        show_absorbance_figure_util( self, make_temp=True )

    def reset_to_defaults( self ):
        self.data_exclusion.set(        ITEM_DEFAULTS['data_exclusion'] )
        self.enable_affine_tran.set(    ITEM_DEFAULTS['enable_affine_tran'] )
        self.decomp_from_separation.set(ITEM_DEFAULTS['decomp_from_separation'] )
        if ENABLE_Rayleigh_Scatter_OPTION:
            self.consider_scatter.set(  ITEM_DEFAULTS['consider_scatter'] )
        self.almerge_analyzer.set(      ITEM_DEFAULTS['almerge_analyzer'] )
        self.peak_mapping_only.set(     ITEM_DEFAULTS['peak_mapping_only'] )
        self.lpm_variations.set(        ITEM_DEFAULTS['lpm_variations'] )
        self.allow_angular_slope_in_mf.set( ITEM_DEFAULTS['allow_angular_slope_in_mf'] )

        if ENABLE_DEPRECATED_OPTIONS:
            self.regression_method.set(         ITEM_DEFAULTS['zx_num_q_points' ] )
            self.zx_build_method.set(           ITEM_DEFAULTS['zx_build_method' ] )
            self.zx_boundary_method.set(        ITEM_DEFAULTS['zx_boundary_method' ] )
            self.zx_boundary.set(               ITEM_DEFAULTS['zx_boundary' ] )

        self.revoke_excel.set(          ITEM_DEFAULTS['revoke_excel'] )
        self.revoke_atsas.set(          ITEM_DEFAULTS['revoke_atsas'] )
        self.revoke_cuda.set(           ITEM_DEFAULTS['revoke_cuda'] )
        self.maintenance_mode.set(      ITEM_DEFAULTS['maintenance_mode'] )

        if ENABLE_Mapping_Animation:
            self.enable_mapping_anim.set(   ITEM_DEFAULTS['enable_mapping_anim' ] )

        self.allow_rank_variation.set(  ITEM_DEFAULTS['allow_rank_variation' ] )
        self.enable_conctype_change.set(    ITEM_DEFAULTS['enable_conctype_change' ] )
        self.extended_conc_dep.set(     ITEM_DEFAULTS['extended_conc_dep' ] )
        if False:
            self.use_xray_conc.set(         ITEM_DEFAULTS['use_xray_conc' ] )
            self.apply_backsub.set(         ITEM_DEFAULTS['apply_backsub'] )
            if ENABLE_XRAY_PROP_CONC_SMOOTHING:
                self.smoothed_xray_conc.set(    DEV_ITEM_DEFAULTS['smoothed_xray_conc' ] )
        # self.zx_add_constant.set(       DEV_ITEM_DEFAULTS['zx_add_constant' ] )

        if ENABLE_REGRESSION_LOOKING_BACK:
            self.recompute_regboundary.set( DEV_ITEM_DEFAULTS['recompute_regboundary' ] )


    def apply( self ):  # overrides parent class method

        ok_ = True

        if not ok_:
            MessageBox.showerror(
                error_msg_param[0],
                error_msg_param[1],
                parent=self,
                )
            self.applied    = None
            # setting this to None causes retry in the parent
            return

        if self.absorbance_has_been_changed and self.temp_absorbance is not None:
            # TODO: parent.serial_data does not exist!
            self.parent.serial_data.absorbance = self.temp_absorbance

        set_setting( 'data_exclusion',          self.data_exclusion.get() )
        set_setting( 'enable_affine_tran',      self.enable_affine_tran.get() )
        set_setting( 'decomp_from_separation',  self.decomp_from_separation.get() )
        if ENABLE_Rayleigh_Scatter_OPTION:
            set_setting( 'consider_scatter',    self.consider_scatter.get() )
        set_setting( 'almerge_analyzer',        self.almerge_analyzer.get() )
        set_setting( 'peak_mapping_only',       self.peak_mapping_only.get() )
        set_setting( 'lpm_variations',          self.lpm_variations.get() )
        set_setting( 'allow_angular_slope_in_mf', self.allow_angular_slope_in_mf.get() )

        revoke_excel = self.revoke_excel.get()
        set_setting( 'revoke_excel', revoke_excel )
        revoke_atsas = self.revoke_atsas.get()
        set_setting( 'revoke_atsas', revoke_atsas )
        if revoke_excel != self.revoke_excel_init or revoke_atsas != self.revoke_atsas_init:
            self.parent.check_environment()
        set_setting( 'revoke_cuda',             self.revoke_cuda.get() )

        set_setting( 'maintenance_mode',        self.maintenance_mode.get() )

        set_setting( 'fixed_guinier_start',     self.fixed_guinier_start.get() )
        set_setting( 'guinier_start_point',     self.guinier_start_point.get() )

        if ENABLE_DEPRECATED_OPTIONS:
            set_setting( 'zx_num_q_points',         self.regression_method.get() )
            set_setting( 'zx_build_method',         self.zx_build_method.get() )
            set_setting( 'zx_boundary_method',      self.zx_boundary_method.get() )
            set_setting( 'zx_boundary',             self.zx_boundary.get() )

        if ENABLE_Mapping_Animation:
            set_setting( 'enable_mapping_anim',     self.enable_mapping_anim.get() )

        set_setting( 'allow_rank_variation',    self.allow_rank_variation.get() )
        set_setting( 'enable_conctype_change',      self.enable_conctype_change.get() )
        set_setting( 'extended_conc_dep',       self.extended_conc_dep.get() )
        if False:
            set_setting( 'use_xray_conc',           self.use_xray_conc.get() )
            set_setting( 'apply_backsub',           self.apply_backsub.get() )

        set_dev_setting( 'enable_xb_save',      self.enable_xb_save.get() )
        set_dev_setting( 'xb_folder',           self.xb_folder.get() )
        set_dev_setting( 'base_file_postfix',   self.base_file_postfix.get() )

        if False:
            if ENABLE_XRAY_PROP_CONC_SMOOTHING:
                set_dev_setting( 'smoothed_xray_conc',  self.smoothed_xray_conc.get() )

        if ENABLE_REGRESSION_LOOKING_BACK:
            set_dev_setting( 'recompute_regboundary',   self.recompute_regboundary.get() )

        self.parent.update_plot_button_state()

        self.applied    = True

    def fixed_guinier_start_tracer( self, *args ):
        cb_value = self.fixed_guinier_start.get()
        state = Tk.NORMAL if cb_value == 1 else Tk.DISABLED
        for w in self.guinier_start_dependents:
            w.config( state=state )

    def guinier_plot_dialog( self ):
        from GuinierStartSelector   import GuinierStartSelector
        dialog = GuinierStartSelector( self, self.serial_data, self.init_ranges  )
        dialog.show()

    def zx_boundary_method_tracer( self, *args ):
        method = self.zx_boundary_method.get()
        state   = Tk.DISABLED if method == 'NO' else Tk.NORMAL
        for rb in self.zx_build_method_buttons:
            rb.config( state=state )

        state   = Tk.NORMAL if method == 'FIXED' else Tk.DISABLED
        self.zx_boundary_entry.config( state=state )
