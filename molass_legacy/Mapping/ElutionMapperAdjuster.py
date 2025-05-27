"""

    ElutionMapperAdjuster.py

    custom widget for mapping adjustment interface

    Copyright (c) 2018-2024, SAXS Team, KEK-PF

"""

import logging
from molass_legacy.KekLib.OurTkinter import Tk, Font
from molass_legacy.KekLib.TkSupplements import BlinkingFrame
from molass_legacy.KekLib.ReadOnlyText import CopyableLabel
from CanvasFrame import CanvasFrame
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from .MappingParams import MappingParams

UV_BASELINE_METHOD_LPM  = 1
UV_LPM_OPTIONS = ['2D', '3D']
XR_BPA_OPTIONS = ['SCD 1', 'SCD 2']

class ElutionMapperAdjuster(Tk.Frame):
    def __init__( self, parent, canvas ):
        Tk.Frame.__init__( self, parent )

        self.parent     = parent
        self.canvas     = canvas
        self.mapper     = canvas.mapper
        self.logger     = logging.getLogger( __name__ )
        self.fixed_font = Font.Font( family="Courier", size=9 )
        self.doing_range_adj    = False
        self.use_xray_conc  = get_setting('use_xray_conc')
        self.use_mtd_conc  = get_setting('use_mtd_conc')
        self.enable_lrf_baseline = get_setting('enable_lrf_baseline')
        self.conc_type = self.mapper.get_conc_type()
        self.uv_lpm_menu = None
        self.xr_bpa_menu = None

        self.initial_std_diff   = canvas.initial_std_diff
        self.current_std_diff   = canvas.current_std_diff

        # --- Frame Settings ----------------------------------------------------
        self.i_frame = i_frame = Tk.Frame( self )
        i_frame.pack()

        iframe0 = Tk.Frame( i_frame )
        iframe0.pack( fill=Tk.X )

        iframe0.rowconfigure( 0, weight=1 )
        for j in range(3):
            iframe0.columnconfigure( j, weight=1 )

        iframe1_ = Tk.Frame( i_frame )
        iframe1_.pack( fill=Tk.X, pady=5 )
        for i in range(3):
            iframe1_.columnconfigure( i, weight=1 )

        self.iframe1 = iframe1 = Tk.Frame( iframe1_ )
        iframe1.grid( row=0, column=0 )
        iframe2 = Tk.Frame( iframe1_ )
        iframe2.grid( row=0, column=1, sticky=Tk.E )

        canvas_width = canvas.get_canvas_width()
        base_frame_width = ( canvas_width - 100 )//3

        m_frame0 = Tk.LabelFrame( iframe0, text="UV Absorbance Baseline Adjustment", labelanchor=Tk.N )
        m_frame0.grid( row=0, column=0, sticky=Tk.N + Tk.S + Tk.E + Tk.W, ipadx=5, ipady=10 )
        m_frame0_adjust = Tk.Frame( m_frame0, width=base_frame_width, height=1 )
        m_frame0_adjust.grid( row=0, column=0 )

        m_frame1 = Tk.LabelFrame( iframe0, text="Xray Scattering Baseline Adjustment", labelanchor=Tk.N )
        m_frame1.grid( row=0, column=1, sticky=Tk.N + Tk.S + Tk.E + Tk.W, padx=10, ipadx=5, ipady=10 )
        m_frame1_adjust = Tk.Frame( m_frame1, width=base_frame_width, height=1 )
        m_frame1_adjust.grid( row=0, column=0 )

        m_frame2 = Tk.LabelFrame( iframe0, text="Mapping Precision Measure", labelanchor=Tk.N )
        m_frame2.grid( row=0, column=2, sticky=Tk.N + Tk.S + Tk.E + Tk.W )
        m_frame2.columnconfigure(0, weight=1)
        m_frame2_adjust = Tk.Frame( m_frame2, width=base_frame_width, height=1 )
        m_frame2_adjust.grid( row=0, column=0, ipadx=5,  ipady=10 )

        self.blink_ready = False

        self.penalty_widgets = []
        self.allowance_vars = []
        self.trace_suppress = False

        # --- Absorbance Baseline Adjustment -----------------------------------
        if self.use_xray_conc or self.use_mtd_conc:
            # create a dummy varriable so that it can be manipulated in the same way as in the normal cases
            self.build_uv_absorbance_option_widgets_dummy()
        else:
            self.build_uv_absorbance_option_widgets( m_frame0 )

        # --- Scattering Baseline Adjustment -----------------------------------
        self.build_xray_scattering_option_widgets( m_frame1 )

        self.blink_ready = True

        # --- Mapping Precision Measure ----------------------------------------
        self.build_mapping_precision_measure_widgets( m_frame2_adjust )

        # --- Optimize Button --------------------------------------------------
        self.build_optimize_button( iframe1 )

        # --- Other Buttons ------------------------------------------------
        self.build_other_buttons( iframe2 )

        self.update_mapping_precision_measure( self.current_std_diff )

    def close_fig(self):
        self.cframe.close_fig()

    def get_conc_type( self ):
        return self.conc_type

    def start_blink( self ):
        if self.blink_ready:
            self.optimize_btn_blink.start()
            self.canvas.change_depedent_states( Tk.DISABLED )

    def stop_blink( self ):
        self.optimize_btn_blink.stop()
        self.canvas.change_depedent_states( Tk.NORMAL )

    def blink_start_tracer(self, *args):
        self.start_blink()

    def build_uv_absorbance_option_widgets( self, m_frame0 ):
        uv_baseline_frame = Tk.Frame( m_frame0 )
        uv_baseline_frame.grid( row=0, column=0, sticky=Tk.W )

        self.uv_baseline_opt = Tk.IntVar()
        self.uv_baseline_opt.set( get_setting( 'uv_baseline_opt' ) )
        self.uv_baseline_type = Tk.IntVar()
        self.uv_baseline_type.set( get_setting( 'uv_baseline_type' ) )

        lrf_base_label = "LRF" if self.enable_lrf_baseline else "shited"
        radio_button_params = [(1, "linear"), (4, lrf_base_label), (5, "integral")]

        self.uv_radio_buttons = []
        self.uv_baseline_opt_dependents = []
        grid_row = -1
        for i, t in enumerate( [ "No correction to input data", "Correction to input data" ] ):
            grid_row += 1
            state = Tk.NORMAL
            b = Tk.Radiobutton( uv_baseline_frame, text=t, variable=self.uv_baseline_opt, value=i, state=state )
            b.grid( row=grid_row, column=0, sticky=Tk.W )
            self.uv_radio_buttons.append( b )
            if i == 1:
                grid_row += 1
                frame = Tk.Frame( uv_baseline_frame )
                frame.grid( row=grid_row, column=0, sticky=Tk.W )
                space = Tk.Frame( frame, width=40 )
                space.grid( row=0, column=0 )
                label = Tk.Label( frame, text="using LPM {" )
                label.grid( row=0, column=1 )
                self.uv_baseline_opt_dependents.append( label )
                state_ = self.get_uv_base_const_state()
                for j, t in radio_button_params:
                    b = Tk.Radiobutton( frame, text=t, variable=self.uv_baseline_type, value=j, state=state_ )
                    b.grid( row=0, column=j+2 )
                    self.uv_baseline_opt_dependents.append( b )

                label = Tk.Label( frame, text="} baseline" )
                label.grid( row=0, column=j+3 )
                self.uv_baseline_with_bpa = Tk.IntVar()
                self.uv_baseline_with_bpa.set(get_setting('uv_baseline_with_bpa'))
                cb = Tk.Checkbutton( frame, text="with BPA", variable=self.uv_baseline_with_bpa, state=state )
                cb.grid( row=0, column=j+4, sticky=Tk.W )
                self.uv_baseline_with_bpa_cb = cb
                self.uv_baseline_with_bpa.trace("w", self.blink_start_tracer)
                self.uv_baseline_opt_dependents.append( cb )

        grid_row += 1
        frame = Tk.Frame( uv_baseline_frame )
        frame.grid( row=grid_row, column=0, sticky=Tk.W )

        self.uv_baseline_adjust = Tk.IntVar()
        self.uv_baseline_adjust.set( get_setting( 'uv_baseline_adjust' )  )
        cb = Tk.Checkbutton( frame, text="Linear adjutment by iterated optimization",
                                variable=self.uv_baseline_adjust )
        cb.grid( row=0, column=0, sticky=Tk.W )
        self.uv_baseline_adjust_cb = cb

        self.uv_deviation_ratio = Tk.DoubleVar()
        self.uv_deviation_ratio.set( get_setting( 'dev_allow_ratio' ) )
        self.build_penalty_widgets( frame, self.uv_deviation_ratio )

        self.uv_baseline_opt.trace( 'w', self.uv_baseline_opt_tracer )
        self.uv_baseline_type.trace( 'w', self.blink_start_tracer )
        self.uv_baseline_adjust.trace( 'w', self.baseline_adjust_tracer )
        self.uv_deviation_ratio.trace( 'w', lambda *args: self.deviation_ratio_tracer( 0 ) )

    def build_uv_absorbance_option_widgets_dummy(self):
        self.uv_deviation_ratio = Tk.DoubleVar()
        self.allowance_vars.append(self.uv_deviation_ratio)

    def lpm_right_button(self, event):
        print('lpm_right_button', event)

        if self.uv_lpm_menu is None:
            self.uv_lpm_option = Tk.IntVar()
            self.uv_lpm_option.set(get_setting('uv_lpm_option'))
            self.uv_lpm_menu = Tk.Menu( self, tearoff=0 )
            self.uv_lpm_menu.add_command( label=UV_LPM_OPTIONS[0], command=lambda: self.set_uv_lpm_option(0) )
            self.uv_lpm_menu.add_command( label=UV_LPM_OPTIONS[1], command=lambda: self.set_uv_lpm_option(1) )

        defaultbg = self.uv_lpm_menu.cget('bg')
        uv_lpm_option = self.uv_lpm_option.get()
        for k in range(2):
            color = 'cyan' if uv_lpm_option == k else defaultbg
            self.uv_lpm_menu.entryconfigure(k, background=color)
        self.uv_lpm_menu.post(event.x_root, event.y_root)

    def set_uv_lpm_option(self, opt):
        self.uv_lpm_option.set(opt)
        set_setting('uv_lpm_option', opt)

    def get_uv_base_const_state( self ):
        return Tk.NORMAL if self.uv_baseline_opt.get() == 1 else Tk.DISABLED

    def uv_baseline_opt_tracer( self, *args ):
        import molass_legacy.KekLib.CustomMessageBox as MessageBox

        uv_adjust_suppressed = get_setting('uv_adjust_suppressed')
        if not uv_adjust_suppressed:
            if self.uv_baseline_opt.get() == 0:
                # when no correction, choosing adjust is unusual
                self.uv_baseline_adjust.set(0)
                MessageBox.showwarning('Linear adjustment change',
                    'Be aware that choosing "No correction" forces\n'
                    '"Linear adjustment" to be "OFF" by default.\n'
                    'You should change it manually if not appropriate.',
                    parent=self
                    )
            else:
                self.uv_baseline_adjust.set(1)
                MessageBox.showwarning('Linear adjustment change',
                    'Be aware that choosing "Correction" forces\n'
                    '"Linear adjustment" to be "ON" by default.\n'
                    'You should change it manually if not appropriate.',
                    parent=self
                    )

        state =self.get_uv_base_const_state()
        for w in self.uv_baseline_opt_dependents:
            w.config( state=state )
        self.start_blink()

    def build_xray_scattering_option_widgets( self, m_frame1 ):
        xray_baseline_frame = Tk.Frame( m_frame1 )
        xray_baseline_frame.grid( row=0, column=0, sticky=Tk.W )

        self.xray_baseline_opt = Tk.IntVar()
        self.xray_baseline_opt.set( get_setting( 'xray_baseline_opt' ) )
        self.xray_baseline_type = Tk.IntVar()
        self.xray_baseline_type.set( get_setting( 'xray_baseline_type' ) )
        self.xray_radio_buttons = []
        self.xray_base_const_dependents = []

        grid_row = -1
        indent = 40

        for i, t in enumerate( [ "No correction to input data", "Correction to input data with method option:" ] ):
            grid_row += 1
            state = Tk.DISABLED if self.use_mtd_conc else Tk.NORMAL
            b = Tk.Radiobutton( xray_baseline_frame, text=t, variable=self.xray_baseline_opt, value=i, state=state )
            b.grid( row=grid_row, column=0, sticky=Tk.W )
            self.xray_radio_buttons.append( b )
            if i == 1:
                grid_row += 1
                frame = Tk.Frame( xray_baseline_frame )
                frame.grid( row=grid_row, column=0 , sticky=Tk.W)
                space = Tk.Frame( frame, width=indent )
                space.grid( row=0, column=0 )
                state_ = self.get_xray_base_const_state()
                label = Tk.Label( frame, text="using LPM {", state=state_ )
                label.grid( row=0, column=1 )
                self.xray_base_const_dependents.append( label )
                value_list = [1, 5]
                text_list = ["linear", "integral"]

                for j, t in zip(value_list, text_list):
                    # state__ = state_ if j < 3 else Tk.DISABLED
                    state__ = state_ 
                    b = Tk.Radiobutton( frame, text=t, variable=self.xray_baseline_type, value=j, state=state__ )
                    b.grid( row=0, column=j+2 )
                    self.xray_base_const_dependents.append( b )
                label = Tk.Label( frame, text="} baseline", state=state_ )
                label.grid( row=0, column=j+3 )
                self.xray_base_const_dependents.append( label )

                self.xray_baseline_with_bpa = Tk.IntVar()
                self.xray_baseline_with_bpa.set(get_setting('xray_baseline_with_bpa'))
                cb = Tk.Checkbutton( frame, text="with BPA", variable=self.xray_baseline_with_bpa, state=state )
                cb.grid( row=0, column=j+4, sticky=Tk.W )
                cb.bind("<Button-3>", self.bpa_right_button)
                self.xray_baseline_with_bpa_cb = cb
                self.xray_baseline_with_bpa.trace("w", self.blink_start_tracer)
                self.xray_base_const_dependents.append( cb )

        grid_row += 1
        frame = Tk.Frame( xray_baseline_frame )
        frame.grid( row=grid_row, column=0, sticky=Tk.W )

        self.xray_baseline_adjust = Tk.IntVar()
        xray_baseline_adjust = get_setting('xray_baseline_adjust')
        disable_xray_adjust = get_setting('disable_xray_adjust')
        if disable_xray_adjust:
            xray_baseline_adjust = 0
        self.xray_baseline_adjust.set(xray_baseline_adjust)

        if disable_xray_adjust:
            dev_allow_ratio = 1
        else:
            dev_allow_ratio = get_setting('dev_allow_ratio')
            state = Tk.DISABLED if self.use_xray_conc or self.use_mtd_conc else Tk.NORMAL
            cb = Tk.Checkbutton( frame, text="Linear adjutment by iterated optimization",
                                    variable=self.xray_baseline_adjust, state=state )
            cb.grid( row=0, column=0, sticky=Tk.W )

        self.xray_deviation_ratio = Tk.DoubleVar()
        self.xray_deviation_ratio.set(1 - dev_allow_ratio)
        self.build_penalty_widgets( frame, self.xray_deviation_ratio )

        self.xray_baseline_opt.trace( 'w', self.xray_baseline_opt_tracer )
        self.xray_baseline_type.trace( 'w', self.blink_start_tracer )
        self.xray_baseline_adjust.trace( 'w', self.baseline_adjust_tracer )
        self.xray_deviation_ratio.trace( 'w', lambda *args: self.deviation_ratio_tracer( 1 ) )
        self.baseline_adjust_tracer()

    def build_penalty_widgets( self, frame, var ):
        self.allowance_vars.append( var )

        label = Tk.Label( frame, text="with deviation ratio" )
        label.grid( row=0, column=1 )
        # label.grid_forget()
        self.penalty_widgets.append( label )
        pw_entry = Tk.Spinbox( frame, textvariable=var,
                                        from_=0, to=1, increment=0.1, 
                                        justify=Tk.CENTER, width=6 )
        pw_entry.grid( row=0, column=2, padx=2 )
        # pw_entry.grid_forget()
        self.penalty_widgets.append( pw_entry )

    def deviation_ratio_tracer( self, i ):
        if self.trace_suppress:
            return

        val = self.allowance_vars[i].get()
        self.allowance_vars[1-i].set( float('%g' % ( 1 - val ) ) )
        if val == 0:
            if i == 0:
                self.uv_baseline_adjust.set( 0 )
            else:
                self.xray_baseline_adjust.set( 0 )

        self.start_blink()

    def bpa_right_button(self, event):
        print('bpa_right_button', event)

        if self.xr_bpa_menu is None:
            self.xr_bpa_option = Tk.IntVar()
            self.xr_bpa_option.set(get_setting('xr_bpa_option'))
            self.xr_bpa_menu = Tk.Menu( self, tearoff=0 )
            self.xr_bpa_menu.add_command( label=XR_BPA_OPTIONS[0], command=lambda: self.set_xr_bpa_option(1) )
            self.xr_bpa_menu.add_command( label=XR_BPA_OPTIONS[1], command=lambda: self.set_xr_bpa_option(2) )

        defaultbg = self.xr_bpa_menu.cget('bg')
        xr_bpa_option = self.xr_bpa_option.get()
        for k in range(2):
            color = 'cyan' if xr_bpa_option == k+1 else defaultbg
            self.xr_bpa_menu.entryconfigure(k, background=color)
        self.xr_bpa_menu.post(event.x_root, event.y_root)

    def set_xr_bpa_option(self, cd):
        print('bpa cd=', cd)
        self.xr_bpa_option.set(cd)
        set_setting('xr_bpa_option', cd)

    def get_xray_base_const_state( self ):
        return Tk.NORMAL if self.xray_baseline_opt.get() == 1 else Tk.DISABLED

    def xray_baseline_opt_tracer( self, *args ):
        if self.xray_baseline_opt.get() == 0:
            # when no correction, choosing adjust is unusual
            self.xray_baseline_adjust.set(0)
        else:
            if self.xray_baseline_type.get() == 0:
                self.xray_baseline_type.set(1)

        state =self.get_xray_base_const_state()
        for w in self.xray_base_const_dependents:
            w.config( state=state )
        self.start_blink()

    def get_xray_base_degree( self ):
        return self.xray_baseline_type.get() + 1

    def baseline_adjust_tracer( self, *args ):
        if self.use_xray_conc or self.use_mtd_conc:
            uv_baseline_adjust  = self.xray_baseline_adjust.get()
        else:
            uv_baseline_adjust  = self.uv_baseline_adjust.get()
        xray_baseline_adjust    = self.xray_baseline_adjust.get()
        self.start_blink()
        if uv_baseline_adjust == 1 and xray_baseline_adjust == 1:
            activate = True
        else:
            activate = False
        for i, w in enumerate( self.penalty_widgets ):
            j = i % 2
            if activate:
                w.grid( row=0, column=2+j, padx=j*2 )
            else:
                w.grid_forget()

        allow_ratio = self.allowance_vars[0].get()
        if 0 < allow_ratio and allow_ratio < 1:
            pass
        else:
            if allow_ratio == 0:
                allow_ratio = 0.1
            else:
                allow_ratio = 0.9
            self.trace_suppress = True
            self.allowance_vars[0].set( allow_ratio )
            self.allowance_vars[1].set( float( '%g' % (1 - allow_ratio) ) )
            self.trace_suppress = False

    def build_mapping_precision_measure_widgets( self, m_frame2_adjust ):
        self.chi_labels = []
        label1 = CopyableLabel( m_frame2_adjust, text="", font=self.fixed_font )
        label1.grid( row=0, column=0, padx=5 )
        self.chi_labels.append( label1 )
        label2 = CopyableLabel( m_frame2_adjust, text="", font=self.fixed_font )
        label2.grid( row=1, column=0, padx=5 )
        self.chi_labels.append( label2 )

        self.cframe = CanvasFrame( m_frame2_adjust, figsize=( 3.0, 0.7 ) )
        self.cframe.grid( row=0, column=1, rowspan=2 )
        self.ax_for_chi = None

        self.sync_options = Tk.IntVar()
        feature_mapped = self.mapper.feature_mapped
        mapper_sync_options = get_setting('mapper_sync_options')
        if mapper_sync_options is None:
            mapper_sync_options = 1 if feature_mapped else 0
        self.sync_options.set(mapper_sync_options)
        sync_opt_frame = Tk.Frame(m_frame2_adjust)
        sync_opt_frame.grid( row=2, column=0, columnspan=2, sticky=Tk.W + Tk.E )
        sync_opt_frame.columnconfigure( 0, weight=1 )
        sync_opt_frame.columnconfigure( 1, weight=1 )
        sync_opt_label = Tk.Label(sync_opt_frame, text="Syncronization Options")
        sync_opt_label.grid(row=0, column=0, sticky=Tk.W, padx=5)
        sybc_opt_btn_frame = Tk.Frame(sync_opt_frame)
        sybc_opt_btn_frame.grid(row=1, column=0, padx=35)
        manual_time_scale = get_setting('manual_time_scale')
        self.sync_opt_btns = []
        # for j, t in enumerate( [ "rmsd finally", "features only", "manual adjust" ] ):
        for j, t in [(0, "automatic"), (2, "manual adjust")]:
            if j == 2 and manual_time_scale is None:
                state = Tk.DISABLED
            else:
                state = Tk.NORMAL
            b = Tk.Radiobutton( sybc_opt_btn_frame, text=t, variable=self.sync_options, value=j, state=state )
            b.grid( row=0, column=j, sticky=Tk.W, padx=5 )
            self.sync_opt_btns.append(b)

        self.sync_options.trace('w', self.sync_options_tracer)

    def sync_options_tracer(self, *args):
        self.start_blink()

    def update_mapping_precision_measure( self, std_diff=None ):
        self.current_std_diff = std_diff
        chi_list = [ self.initial_std_diff ]
        std_diff_ = self.initial_std_diff if std_diff is None else std_diff
        chi_list.append( std_diff_ )

        for i, chi in enumerate( chi_list ):
            varname = 'Initial' if i==0 else 'Current'
            self.chi_labels[i].config( text='%s nRMSD=%.5g' % ( varname, chi ) )

        def draw_std_diff_func( fig ):
            if self.ax_for_chi is None:
                self.ax_for_chi = fig.add_subplot( 111 )
            else:
                self.ax_for_chi.cla()
            self.ax_for_chi.set_axis_off()
            self.ax_for_chi.set_xlim( 0, 1 )
            self.ax_for_chi.set_ylim( 0, 1 )
            prev_chi = None
            for i, chi in enumerate( chi_list ):
                y = 0.75 - 0.5*i
                if chi <= 0.3:
                    color = 'green'
                elif chi <= 0.5:
                    color = 'orange'
                else:
                    color = 'red'
                if prev_chi is not None and chi > prev_chi:
                    color = 'red'
                self.ax_for_chi.plot( [ 0, chi ], [ y,  y ], color=color, linewidth=10, solid_capstyle='butt', alpha=0.5 )
                prev_chi = chi

        self.cframe.draw( draw_std_diff_func )
        self.canvas.update_guide_message()

    def build_optimize_button( self, iframe1 ):
        label11 = Tk.Label( iframe1, text="Press " )
        label11.pack( side=Tk.LEFT )

        self.optimize_btn_blink = BlinkingFrame( iframe1 )
        self.optimize_btn_blink.pack( side=Tk.LEFT )

        opt_btn_state = Tk.DISABLED if self.canvas is None or self.use_mtd_conc else Tk.NORMAL
        self.optimize_btn = Tk.Button( self.optimize_btn_blink, text="Optimize", command=self.optimize, state=opt_btn_state )
        self.optimize_btn.pack()

        self.optimize_btn_blink.objects = [self.optimize_btn]

        label12 = Tk.Label( iframe1, text=' to get adjusted with the specified parameters above; Press "OK" button below when it is appropriate.' )
        label12.pack( side=Tk.LEFT )

    def build_other_buttons( self, iframe2 ):
        self.manual_sync_btn = Tk.Button( iframe2, text="Manual Adjust", command=self.show_manual_adjuster )
        self.manual_sync_btn.pack( side=Tk.RIGHT, anchor=Tk.E, padx=5 )
        self.enable_mapping_anim    = get_setting( 'enable_mapping_anim' ) == 1
        if self.enable_mapping_anim:
            self.anim_btn = Tk.Button( iframe2, text="Animation", command=self.show_animation )
            self.anim_btn.pack( side=Tk.RIGHT, anchor=Tk.E, padx=5 )

    def optimize( self, helper_info=None ):
        self.change_cursor( 'wait' )

        if helper_info is None:
            helper_info = self.canvas.get_helper_info()

        if not self.use_xray_conc:
            pre_recog = self.canvas.serial_data.pre_recog
            btype = self.uv_baseline_type.get()
            self.mapper.absorbance.compute_base_curve(pre_recog, btype)

        while True:
            try:
                self.optimize_impl( helper_info )
                break
            except RuntimeError as exc:
                from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker
                exception = exc
                etb = ExceptionTracebacker()
                self.logger.warning('optimize_impl failed etb=%s', str(etb))
                assert False

        self.change_cursor( '' )

    def change_cursor( self, cursor ):
        for w in self.canvas.get_cursor_widgets():
            w.config( cursor=cursor )   # does not work
            w.update()

    def update_proportional_uv_curve(self):
        self.stop_blink()
        self.mapper.x_base = self.mapper.compute_xray_baseline( self.get_mapping_params() )
        self.mapper.a_base = self.mapper.x_base / self.mapper.x_curve.max_y
        self.mapper.A_init = 1
        self.mapper.B_init = 0
        self.canvas.draw( clear=True )

    def get_mapping_params( self ):
        if self.use_xray_conc or self.use_mtd_conc:
            uv_baseline_opt = self.xray_baseline_opt.get()
            if self.use_xray_conc:
                uv_baseline_type = 0
            else:
                uv_baseline_type = self.xray_baseline_type.get()
            uv_baseline_adjust = self.xray_baseline_adjust.get()
            uv_baseline_with_bpa = self.xray_baseline_with_bpa.get()
        else:
            uv_baseline_opt = self.uv_baseline_opt.get()
            uv_baseline_type = self.uv_baseline_type.get()
            uv_baseline_adjust = self.uv_baseline_adjust.get()
            uv_baseline_with_bpa = self.uv_baseline_with_bpa.get()
        return MappingParams([
            ( 'uv_baseline_opt',            uv_baseline_opt ),
            ( 'uv_baseline_type',           uv_baseline_type ),
            ( 'uv_baseline_adjust',         uv_baseline_adjust ),
            ( 'uv_baseline_with_bpa',       uv_baseline_with_bpa ),
            ( 'xray_baseline_opt',          self.xray_baseline_opt.get() ),
            ( 'xray_baseline_type',         self.xray_baseline_type.get() ),
            ( 'xray_baseline_adjust',       self.xray_baseline_adjust.get() ),
            ( 'xray_baseline_with_bpa',     self.xray_baseline_with_bpa.get() ),
            ( 'dev_allow_ratio',            self.allowance_vars[0].get() ),
            ])

    def optimize_impl( self, helper_info ):
        self.change_cursor( 'wait' )

        self.stop_blink()

        params = self.get_mapping_params()
        sync_options = self.sync_options.get()
        self.mapper.optimize( opt_params=params, sync_options=sync_options, helper_info=helper_info )

        self.canvas.update_helper_info( helper_info )
        self.canvas.draw( clear=True )
        self.canvas.update_button_colors()

        # Pending:
        # self.canvas.update_range_info( update_list=True )   # update range list since range info may have been changed.
        self.current_std_diff   = self.mapper.std_diff
        self.update_mapping_precision_measure( self.current_std_diff )

        self.change_cursor( '' )

    def show_manual_adjuster(self, debug=True):
        if debug:
            from importlib import reload
            import Mapping.ManualAdjuster
            reload(Mapping.ManualAdjuster)
        from .ManualAdjuster import ManualAdjuster
        ms = ManualAdjuster(self.canvas, self.canvas.serial_data, self.mapper)
        ms.show()
        if ms.applied:
            self.sync_opt_btns[1].config(state=Tk.NORMAL)
            self.sync_options.set(2)

    def show_animation( self ):
        self.change_cursor( 'wait' )
        from molass_legacy.Mapping.ElutionMapperAnimator  import ElutionMapperAnimation
        anim = ElutionMapperAnimation( self.parent, self.canvas.serial_data )
        self.change_cursor( '' )
        anim.show()
