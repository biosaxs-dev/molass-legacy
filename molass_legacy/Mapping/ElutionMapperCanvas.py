"""

    ElutionMapperCanvas.py

        recognition of peaks

    Copyright (c) 2018-2023, SAXS Team, KEK-PF

"""
import os
import numpy as np
import copy
import time
import threading
import logging
import queue
from scipy.interpolate import UnivariateSpline
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
import molass_legacy.KekLib.CustomMessageBox as MessageBox
from molass_legacy.KekLib.TkUtils import is_low_resolution
from molass_legacy.KekLib.BasicUtils import get_caller_module
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable, BlinkingFrame
from molass_legacy.QuickAnalysis.AnalysisGuide import get_analysis_guide_info
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting, get_xray_picking
from molass_legacy.UV.PlainCurveUtils import get_flat_wavelength
from molass_legacy.SerialAnalyzer.AnalyzerUtil import get_init_ranges
from molass_legacy.DataStructure.AnalysisRangeInfo import set_default_analysis_range_info
from .ElutionMapper import ElutionMapper
# from .ElutionMapperPlotter import ElutionMapperPlotter, SUGGEST_DISCOMFORT_BY_TEXT
from .ElutionMapperAdjuster import ElutionMapperAdjuster
from .MappingParams import get_mapper_opt_params
from molass_legacy._MOLASS.Version import is_developing_version

show_button_text_dict = {   'locally'   : 'Show Locally Scaled Curves', 
                            'uniformly' : 'Show Uniformly Scaled Curves',
                            }

DRAW_MAPPING_RANGES     = True
USE_NEW_3D_PLOT_CLASS   = True
ENABLE_SIMULATION       = False

class ElutionMapperCanvas(Dialog):
    def __init__(self, dialog, serial_data, sd_orig, pre_recog_orig, mapper, judge_holder, initial_navigation=True):
        assert serial_data.pre_recog is not None

        self.logger         = logging.getLogger( __name__ )
        self.grab = 'local'     # used in grab_set
        self.dialog = dialog
        self.parent = dialog.parent
        self.applied        = False
        self.ok_debugging   = False
        self.caller_module  = get_caller_module( level=2 )
        self.developer_mode = False
        self.absorbance_picking = get_setting('absorbance_picking')
        self.intensity_picking  = get_xray_picking()
        self.logger.info("dialog will be shown in the mapping plane with λ=%g and q=%g", self.absorbance_picking, self.intensity_picking)
        self.logger.info("UV extinct wavelength λ=%g", get_flat_wavelength(serial_data.lvector))

        self.serial_data    = serial_data

        self.sd_orig = sd_orig                  # currently used only for do_devel_test in QuickAnalysis.PreDecomposer.py
        self.pre_recog_orig = pre_recog_orig    # currently used only for do_devel_test in QuickAnalysis.PreDecomposer.py

        self.a_vector_size  = serial_data.conc_array.shape[1]       # better than len(self.mapper.a_vector) ?
        self.qvector        = serial_data.qvector
        self.intensity_array    = serial_data.intensity_array
        self.xray_slice     = serial_data.xray_slice
        self.mapper = mapper
        self.judge_holder = judge_holder
        self.judge_holder.set_mapper(mapper)

        self.update_range_info()

        self.mapping_show_mode  = 'locally'
        self.initial_std_diff   = self.mapper.std_diff
        self.current_std_diff   = self.mapper.std_diff
        self.final_std_diff     = None
        self.initial_navigation = initial_navigation
        self.state_depedents    = []
        self.get_guide_info()
        Dialog.__init__( self, self.parent, "Elution Mapping Confirmation", visible=False)
        self.update_button_colors()
        self.decomp_editor = None

    def cancel(self):
        # overiding cancel to cleanup self.fig
        # because the call to the destructor __del__ seems to be delayed
        self.plotter.close_fig()
        self.adjuster.close_fig()
        print("ElutionMapperCanvas: closed figs")
        Dialog.cancel(self)

    def show( self ):
        self.toolbar    = True
        fully_automatic = get_setting( 'fully_automatic' )
        if self.mapper.use_mtd_conc:
            if fully_automatic == 1:
                self.parent.after(2000, lambda: self.ok_button.invoke() )
            else:
                pass
        else:
            if fully_automatic == 1:
                self.start_full_automatic_control()
            else:
                auto_navigated_dailog = get_setting( 'auto_navigated_dailog' )
                if self.initial_navigation and auto_navigated_dailog == 2:
                    self.parent.after(100, self.auto_navigate_to_decomp_editor )
        self._show()

    def body(self, body_frame):   # overrides parent class method
        from .ElutionMapperPlotter import ElutionMapperPlotter

        tk_set_icon_portable( self, module=self.caller_module )

        self.upper_frame = Tk.Frame(body_frame)
        self.upper_frame.pack()

        self.plotter    = ElutionMapperPlotter( self, self.upper_frame, self.serial_data, mapper=self.mapper )
        self.plotter.in_range_adjustment = False    # should be changed depending on self.initial_std_diff
        self.plotter.draw()
        self.state_depedents.append( self.plotter.show_diff_btn )

        self.guide_frame = Tk.Frame(body_frame)
        self.guide_frame.pack(fill=Tk.X)
        guide_label = Tk.Label(self.guide_frame, bg='white')
        guide_label.pack(fill=Tk.X, pady=10)
        self.guide_message = guide_label

        self.set_basic_buttons(self.upper_frame)

        self.parent.config(cursor='')

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack(fill=Tk.X, expand=1, padx=20)
        bottom_space = Tk.Frame(self, height=10)
        bottom_space.pack()

        num_buttons = 4
        if is_developing_version():
            num_buttons += 1

        for j in range(num_buttons):
            box.columnconfigure(j, weight=1)

        w = Tk.Button(box, text="◀ Main", command=self.cancel)
        w.grid(row=0, column=0, sticky=Tk.W, padx=10, pady=5)
        self.cancel_button = w

        w = Tk.Button(box, text="▽ Range Editor", command=self.show_range_editor_dialog)
        w.grid(row=0, column=1)
        self.reditor_btn = w
        self.state_depedents.append(w)
        self.default_btn_colors = [ self.reditor_btn.cget(atr) for atr in ['fg', 'bg']]

        state = Tk.DISABLED if self.mapper.get_conc_type() == 2 else Tk.NORMAL

        self.decomp_btn_blink = BlinkingFrame(box)
        self.decomp_btn_blink.grid(row=0, column=2)

        self.decomp_btn = Tk.Button(self.decomp_btn_blink, text="▽ Decomposition Editor", command=self.show_decomp_editor, state=state)
        self.decomp_btn.pack()

        self.decomp_btn_blink.objects = [self.decomp_btn]

        self.state_depedents.append(self.decomp_btn)

        w = Tk.Button(box, text="▶ Execute", command=self.ok, default=Tk.ACTIVE)
        w.grid(row=0, column=3, sticky=Tk.E, padx=10, pady=5)
        self.ok_button = w
        self.state_depedents.append(w)

        if is_developing_version():
            w = Tk.Button(box, text="DevelTest", command=self.do_devel_test)
            w.grid(row=0, column=4, sticky=Tk.E, padx=10, pady=5)

        mapping_canvas_debug = get_setting('mapping_canvas_debug')
        if mapping_canvas_debug:
            w = Tk.Button(box, text="OK Debug", command=self.ok_debug, default=Tk.ACTIVE)
            w.grid(row=0, column=4, sticky=Tk.E, padx=10, pady=5)
            self.ok_debug = w
            self.state_depedents.append(w)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def update_button_colors(self):
        use_elution_models = get_setting('use_elution_models')
        ready_to_execute = False
        if use_elution_models:
            self.reditor_btn.config(fg=self.default_btn_colors[0], bg=self.default_btn_colors[1])
            self.decomp_btn.config(fg='white', bg='green')
            ready_to_execute = True
        else:
            if self.guide_info.decomp_proc_needed():
                self.reditor_btn.config(fg='black', bg='orange')
                self.decomp_btn.config(fg=self.default_btn_colors[0], bg=self.default_btn_colors[1])
                self.decomp_btn_blink.start()
            else:
                self.reditor_btn.config(fg='white', bg='green')
                self.decomp_btn.config(fg=self.default_btn_colors[0], bg=self.default_btn_colors[1])
                ready_to_execute = True

        if ready_to_execute:
            self.ok_button.config(fg='white', bg='green')
        else:
            self.ok_button.config(fg=self.default_btn_colors[0], bg=self.default_btn_colors[1])

    def validate( self ):   # called on "OK" button press
        if self.adjuster.optimize_btn_blink.switch:
            ok = MessageBox.showinfo( "Confirmation",
                    "You have to 'Cancel' to finish because you have not 'Optimize'd the changed parameters.",
                    parent=self )
            return False

        return True

    def ok_debug(self):
        self.ok_debugging = True
        self.ok()

    def apply( self ):  # overrides parent class method
        self.applied = True
        self.final_std_diff = self.current_std_diff
        self.update_params()
        mapping_params = get_mapper_opt_params()
        self.logger.info("memorizing %s as used_mapping_params.", str(mapping_params))
        set_setting('used_mapping_params', mapping_params)
        self.try_image_save()

    def get_cd_colors(self):
        self.judge_holder.get_cd_colors()
        self.update_guide_info()
        return self.judge_holder.get_cd_colors(self.guide_info)

    def update_params( self ):
        analysis_range_info = get_setting('analysis_range_info')
        if analysis_range_info is None:
            set_default_analysis_range_info(self.mapper.x_curve)

        # set_setting also for adjuster items
        adjuster    = self.adjuster
        if adjuster.use_xray_conc or adjuster.use_mtd_conc:
            uv_baseline_opt = 0
            uv_baseline_type = 0
            uv_baseline_adjust = 0
            uv_baseline_with_bpa = None
        else:
            uv_baseline_opt = adjuster.uv_baseline_opt.get()
            uv_baseline_type = adjuster.uv_baseline_type.get()
            uv_baseline_adjust = adjuster.uv_baseline_adjust.get()
            uv_baseline_with_bpa = adjuster.uv_baseline_with_bpa.get()

        set_setting( 'uv_baseline_opt',         uv_baseline_opt )
        set_setting( 'uv_baseline_type',        uv_baseline_type )
        set_setting( 'uv_baseline_adjust',      uv_baseline_adjust )
        set_setting( 'uv_baseline_with_bpa',    uv_baseline_with_bpa )
        set_setting( 'xray_baseline_opt',       adjuster.xray_baseline_opt.get() )
        set_setting( 'xray_baseline_type', adjuster.xray_baseline_type.get() )
        set_setting( 'xray_baseline_adjust',    adjuster.xray_baseline_adjust.get() )
        set_setting( 'xray_baseline_with_bpa',     adjuster.xray_baseline_with_bpa.get() )

        # for backward compatibiliy
        baseline_degree = self.adjuster.get_xray_base_degree()
        set_setting( 'baseline_degree', baseline_degree )
        # scattering_correction will be set in AnalyzerDialog

    def update_params_for_data_correction(self):
        self.update_params()

    def try_image_save( self ):
        test_pattern = get_setting('test_pattern')
        if test_pattern is None or test_pattern < 7:
            return

        mapping_image_folder = get_setting('mapping_image_folder')
        if mapping_image_folder is not None and os.path.exists(mapping_image_folder):
            self.plotter.save_the_figure( mapping_image_folder, get_setting('analysis_name') )

    def get_xray_correction_necessity( self ):
        adjuster    = self.adjuster
        return adjuster.xray_baseline_opt.get() > 0 or adjuster.xray_baseline_adjust.get() == 1

    def update_guide_info(self):
        guide_info = get_analysis_guide_info(self.mapper, self.judge_holder, self.discomfort)
        if guide_info.extra_infos is None:
            # i.e, when "already decomposed"
            # we could replace guide_info now since "already decomposed" info has been separated
            pass
        else:
            self.guide_info = guide_info
        return self.guide_info      # this return value will be used in QuickAnalysis.JudgeHolder

    def get_guide_info( self ):
        self.discomfort = self.serial_data.compute_scattering_baseline_discomfort()
        assert self.initial_std_diff is not None
        self.update_guide_info()
        self.three_d_guide  = self.guide_info.three_d_guide

    def update_guide_message(self):
        self.update_guide_info()
        self.guide_message.config( text=self.guide_info.message, fg=self.guide_info.fg )
        self.logger.info('guide: ' + self.guide_info.message.replace('▶ ', '>>'))  # replace('▶ ', '>>') : to avoid command prompt encode error

    def set_basic_buttons( self, body_frame ):
        self.edit_mode = self.guide_info.edit_mode

        button_bar = Tk.Frame( body_frame )
        button_bar.pack( fill=Tk.X, padx=10, pady=5 )

        self.toggle_button = Tk.Button( button_bar, text=self.get_toggle_text(), command=self.toggle_mode )
        self.toggle_button.pack( side=Tk.LEFT )
        self.state_depedents.append( self.toggle_button )

        self.adjuster   = ElutionMapperAdjuster(body_frame, self)
        self.adjuster.pack( side=Tk.LEFT, fill=Tk.BOTH, expand=1, padx=20 )
        self.state_depedents.append(self.adjuster.manual_sync_btn)

        other_btn_frame = self.plotter.other_btn_frame
        btn_pady = 5
        text = show_button_text_dict[ self.get_toggled_mode() ]
        self.show_button = Tk.Button(other_btn_frame, text=text, width=25,
                                command=self.show_curve, state=Tk.NORMAL )
        self.show_button.pack(side=Tk.RIGHT, padx=10, pady=btn_pady )
        self.state_depedents.append( self.show_button )

        space = Tk.Frame(other_btn_frame, width=self.get_canvas_width()*0.14)
        space.pack(side=Tk.RIGHT)

        investigate_btn = Tk.Button(other_btn_frame, text="Investigate Correction", command=self.investigate_correction)
        investigate_btn.pack(side=Tk.RIGHT, padx=10)
        self.state_depedents.append(investigate_btn)

        pre_sync_btn = Tk.Button(other_btn_frame, text="3D View", command=self.show_threedim_dialog)
        pre_sync_btn.pack(side=Tk.RIGHT, padx=10)
        self.state_depedents.append(pre_sync_btn)

        # work-around to get the layout right; still required?
        self.edit_mode ^= True
        self.after( 100, lambda: self.toggle_mode() )

    def get_edit_mode( self ):
        return self.edit_mode

    def toggle_mode( self ):
        if self.edit_mode:
            self.edit_mode  = False
            self.adjuster.pack_forget()
            self.toggle_button.config( text=self.get_toggle_text() )
        else:
            self.edit_mode  = True
            self.adjuster.pack( fill=Tk.BOTH, expand=Tk.Y, pady=0 )
            self.toggle_button.config( text=self.get_toggle_text() )

    def get_toggle_text( self ):
        text = 'Hide Options' if self.edit_mode else 'Show Options'
        return text

    def change_depedent_states( self, state ):
        for w in self.state_depedents:
            w.config(state=state, fg=self.default_btn_colors[0], bg=self.default_btn_colors[1])

    def draw( self, clear=False, restrict_info=None ):
        self.plotter.draw( clear=clear, restrict_info=restrict_info )

    def draw_mapped( self, clear=False ):
        self.plotter.draw( clear=clear )

    def get_helper_info( self ):
        # temporary fix. to be removed
        # return self.plotter.helper_info
        return None

    def update_helper_info( self, helper_info ):
        # temporary fix. to be removed
        # self.plotter.helper_info        = helper_info
        # self.serial_data.helper_info    = helper_info
        pass

    def get_cursor_widgets( self ):
        return [ self, self.plotter.mpl_canvas_widget ]

    def update_range_info( self, update_list=False ):
        if get_setting( 'range_type' ) == 2:
            manual_range_info = get_setting( 'manual_range_info' )
            if manual_range_info is not None:
                self.parent.range_info = [ manual_range_info ]
            ranges_ = self.parent.range_info[0]
        else:
            ranges_ = self.mapper.get_int_ranges()

        self.range_info     = [ ranges_ ]
        self.annotation_points  = ranges_

        if update_list:
            self.range_changed  = True
            init_ranges = get_init_ranges( just_ranges=ranges_ )
            self.range_list_entry.refresh_spin_boxes( init_ranges )
        else:
            self.range_changed  = False

    def get_range_info( self ):
        analysis_range_info = get_setting('analysis_range_info')
        if analysis_range_info is None:
            return self.range_info
        else:
            return [ analysis_range_info.get_ranges() ]

    def set_adjust_frame( self, bframe ):
        # to avoid for the right column to shrink too small (i.e., smaller than the right pane)
        # self.update()
        self.update_idletasks()
        # self.window_width   = self.winfo_width()
        # get the figure size in pixels in stead because the above does not work properly
        figsize = self.plotter.get_figsize()
        self.window_width   = int( figsize[0] )
        self.right_col_adjust_frame = Tk.Frame( bframe, width=int( self.window_width/3 ), height=0 )
        self.right_col_adjust_frame.grid( row=0, column=2 )

    def get_canvas_width( self ):
        return self.plotter.get_canvas_width()

    def show_absorbance_figure( self ):
        pass

    def show_scattering_figure( self ):
        from .ElutionMapperPlotter import SUGGEST_DISCOMFORT_BY_TEXT
        if self.adjuster.optimize_btn_blink.switch:
            ok = MessageBox.showinfo( "Confirmation",
                    "You have to complete 'Optimize' before investigation\nbecause you have changed optimizer options.",
                    parent=self )
            return False

        # TODO: unify with show_absorbance_figure_util in AbsorbancePlot.py
        print( 'show_scattering_figure' )
        if not SUGGEST_DISCOMFORT_BY_TEXT:
            self.blink_button_frame.stop()

        # TODO: this setting shoud be temporary
        # set_setting( 'correction_iteration', self.adjuster.correction_iteration.get() )

        self.config( cursor='wait' )
        self.update()

        affine_info = self.mapper.get_affine_info()

        if USE_NEW_3D_PLOT_CLASS:
            from molass_legacy.SerialAnalyzer.Scattering3dPlot   import Scattering3dPlotDialog
            data_list = [ self.serial_data.intensity_array ]
            if ENABLE_SIMULATION or get_setting('enable_drift_simulation') == 1:
                if self.three_d_guide:
                    from molass_legacy.SerialAnalyzer.DriftSimulation import DriftSimulation
                    from molass_legacy.SerialAnalyzer.DriftAnalyzer import create_mapper_for_drift_analysis

                    sd  = self.serial_data
                    mapper = create_mapper_for_drift_analysis( self, sd )
                    sim = DriftSimulation( sd, mapper )
                    self.sim_info = [ mapper, sim ]
                    sim_data, sim_base = sim.create_sim_data()
                    data_list.append( sim_data )
                else:
                    sim_base = None
            else:
                sim_base = None
            self.scatterning_plot = dialog = Scattering3dPlotDialog( self, self.serial_data, self.mapper, data_list=data_list, sim_base=sim_base )
            dialog.show()
            if dialog.applied:
                if dialog.apply_fitted_base.get() == 1:
                    set_setting( 'base_drift_params', sim.drift_params )
        else:
            from molass_legacy.SerialAnalyzer.ScatteringPlot import ScatteringPlot
            # for the construction of ScatteringPlot, curve_y should be self.mapper.x_vector
            # i.e., not corrected xray-curve
            self.scatterning_plot = ScatteringPlot( self.qvector, self.mapper.x_vector, self.intensity_array, self.xray_slice, affine_info )
            if self.adjuster is None:
                baseline_degree = None
                do_correction   = False
            else:
                baseline_degree = self.adjuster.get_xray_base_degree()
                # do_correction   = self.adjuster.xray_baseline_opt.get() > 0
                do_correction   = True
            self.scatterning_plot.draw_3d_for_gui( self, 'Xray Scattering from ' + get_setting( 'in_folder' ), do_correction, baseline_degree )
        self.config( cursor='' )

    def get_toggled_mode( self ):
        if self.plotter.mapping_show_mode == 'locally':
            mode = 'uniformly'
        else:
            mode = 'locally'
        return mode

    def show_absorbance_viewer( self ):
        from molass_legacy.UV.AbsorbanceViewer import AbsorbanceViewer
        self.viewer = AbsorbanceViewer( self.serial_data.absorbance, helper_info=self.serial_data.helper_info )
        self.viewer.show( self )

    def show_curve( self ):
        previous_mode = self.plotter.mapping_show_mode
        mode = self.get_toggled_mode()
        self.plotter.mapping_show_mode = mode
        self.plotter.update_show_diff_btn_state()
        text = show_button_text_dict[ previous_mode ]
        self.show_button.config( text=text )
        self.draw_mapped( clear=True )

    def investigate_correction( self ):
        if self.adjuster.optimize_btn_blink.switch:
            ok = MessageBox.showinfo( "Confirmation",
                    "You have to complete 'Optimize' before investigation\nbecause you have changed optimizer options.",
                    parent=self )
            return False

        data = self.intensity_array
        jvector = np.arange( data.shape[0] )
        qvector = data[0,:,0]
        corrected_data = copy.deepcopy( data )
        affine_info = self.mapper.get_affine_info()
        need_adjustment = self.adjuster.xray_baseline_adjust.get()
        baseline_opt = self.adjuster.xray_baseline_opt.get()
        baseline_type = self.adjuster.xray_baseline_type.get()

        from molass_legacy.SerialAnalyzer.ScatteringBaseCorrector import ScatteringBaseCorrector

        corrector = ScatteringBaseCorrector( jvector, qvector, corrected_data,
                                        curve=self.mapper.sd_xray_curve,
                                        affine_info=affine_info,
                                        inty_curve_y=self.mapper.x_vector,
                                        baseline_opt=baseline_opt,
                                        baseline_type=baseline_type,
                                        need_adjustment=need_adjustment,
                                        parent=self, with_demo=True )

        from molass_legacy.SerialAnalyzer.ScatteringBaseInvestigator import InvestigatorDialog
        investigator = InvestigatorDialog( self.parent, corrector )
        investigator.show()

    def show_range_editor_dialog(self, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.RangeEditors.RangeEditorDialog
            reload(molass_legacy.RangeEditors.RangeEditorDialog)
        from molass_legacy.RangeEditors.RangeEditorDialog import RangeEditorDialog

        self.grab_set()     # temporary fix to the grab_release problem
        self.change_depedent_states(Tk.DISABLED)
        self.update_params_for_data_correction()
        self.range_editor = RangeEditorDialog(self.parent, "Range Editor", self.serial_data, self.mapper, self.judge_holder)
        self.range_editor.show()
        self.change_depedent_states(Tk.NORMAL)
        self.update_button_colors()
        self.grab_set()     # temporary fix to the grab_release problem

    def debug_sd(self):
        from molass_legacy.DataStructure.MdViewer import MdViewer
        sd = self.serial_data
        M = sd.intensity_array[:,:,1].T
        mdv = MdViewer(self, M, xvector=sd.qvector)
        mdv.show()

    def show_decomp_editor( self, counter=None, debug=True):
        # import Decomposer
        if debug:
            from importlib import reload
            import molass_legacy.RangeEditors.DecompEditorDialog
            reload(molass_legacy.RangeEditors.DecompEditorDialog)
        from molass_legacy.RangeEditors.DecompEditorDialog import DecompEditorDialog

        self.decomp_btn_blink.stop()
        self.grab_set()     # temporary fix to the grab_release problem
        self.change_depedent_states(Tk.DISABLED)
        self.update_params_for_data_correction()
        # self.serial_data.log_id_values("before DecompEditorDialog")
        self.decomp_editor = DecompEditorDialog(self.parent, "Decomposition Editor", self.serial_data, self.mapper, self.judge_holder)
        if counter is not None:
            counter[0] += 1
            self.update()
        self.decomp_editor.show(counter=counter)
        # self.serial_data.log_id_values("after DecompEditorDialog")
        self.update_guide_message()
        self.change_depedent_states(Tk.NORMAL)
        self.update_button_colors()
        self.grab_set()     # temporary fix to the grab_release problem

    def has_decomp_editor(self):
        return self.decomp_editor is not None

    def auto_navigate_to_decomp_editor( self ):
        counter = [0]
        def progress_cb():
            self.update()
            return counter[0]

        def show_progress():
            def navigate():
                counter[0] += 1
                self.update()
                """
                adjuster = self.adjuster
                if not adjuster.get_edit_mode():
                    adjuster.toggle_button.invoke()
                    self.update()
                if adjuster.get_conc_type() == 0:
                    adjuster.uv_baseline_adjust.set( 1 )
                adjuster.xray_radio_buttons[1].invoke()
                adjuster.xray_baseline_adjust.set( 1 )
                counter[0] += 1
                self.update()
                adjuster.optimize_btn.invoke()
                counter[0] += 1
                self.update()
                """
                self.show_decomp_editor(counter=counter)

            self.after(100, navigate)
            from molass_legacy.KekLib.ProgressMinDialog import ProgressMinDialog
            counter[0] += 1
            for k in range(3):
                try:
                    self.progress = ProgressMinDialog(self,
                                title="Navigation Progress",
                                message="Please be patient until the Decomposition Editor appears.",
                                length=240,
                                num_steps=6, progress_cb=progress_cb, interval=100)
                    break
                except:
                    # this case may occur when the machine is buzy for running anti-virus software.
                    # better solution is desired.
                    self.logger.warning("failed to show Navigation Progress. retry after 1 sec.")
                    self.update()
                    time.sleep(1)

        self.after(100, show_progress)

    def start_full_automatic_control( self ):
        if self.guide_info.decomp_proc_needed():     # to be consistent with the Tester
            self.parent.after(100, self.auto_navigate_to_decomp_editor)
            self.parent.after(1000, self.start_full_automatic_control_impl)
        else:
            self.parent.after(100, self.auto_navigate_no_decomp)

    def start_full_automatic_control_impl( self ):
        self.control_thread = threading.Thread(
                                target=self.control_gui_automatically,
                                name='GuiControlThread',
                                args=[]
                                )

        self.control_queue = queue.Queue()
        self.control_thread.start()
        self.after(1000, self.monitor_control_thread)

    def monitor_control_thread( self ):
        try:
            ret = self.control_queue.get(block=False)
            self.control_thread.join()
            time.sleep(1)
            self.ok_button.invoke()
        except Exception as exc:
            print("monitor_control_thread", exc)
            self.parent.after(1000, self.monitor_control_thread)

    def decomp_editor_state( self ):
        try:
            return self.decomp_editor.is_ready()
        except:
            print( 'no self.decomp_editor' )
        return False

    def control_gui_automatically( self ):
        print("control_gui_automatically")
        while not self.decomp_editor_state():
            print( "wainting for decomp_editor being ready" )
            time.sleep(1)

        time.sleep(1)
        self.decomp_editor.ok_button.invoke()
        self.control_queue.put(1)

    def auto_navigate_no_decomp(self):
        self.ok_button.invoke()

    def show_threedim_dialog(self):
        self.dialog.show_threedim_dialog()

    def get_sd(self):
        return self.serial_data

    def do_devel_test(self):
        from importlib import reload
        import molass_legacy.Tools.EmbedCushion
        reload(molass_legacy.Tools.EmbedCushion)
        from molass_legacy.Tools.EmbedCushion import embed_cushion

        embed_cushion(self)
