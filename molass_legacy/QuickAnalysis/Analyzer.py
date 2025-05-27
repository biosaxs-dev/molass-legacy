"""

    QuickAnalysis.Analyzer.py

    Copyright (c) 2016-2024, SAXS Team, KEK-PF

"""
import os
from molass_legacy.KekLib.BasicUtils import exe_name, clear_dirs_with_retry
import molass_legacy.KekLib.CustomMessageBox as MessageBox
from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker, log_exception
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.SerialAnalyzer.DevSettings import get_dev_setting
from molass_legacy.QuickAnalysis.JudgeHolder import JudgeHolder

SD_DEBUG = False
if SD_DEBUG:
    from molass_legacy.SerialAnalyzer.SdDebugger import SdDebugger

class Analyzer:
    def __init__( self, main_dialog, loader ):
        self.dialog = main_dialog   # 
        self.parent = main_dialog   # temp fix
        self.loader = loader
        self.mapper_canvas  = None
        self.mapper_hook    = None
        self.app_logger     = None
        self.logger_visible = False
        self.cancel_reports = False     # True for testing only
        self.preview_survey = False
        self.waiting_for_reply = False  # used in test control
        self.mfc_dialog = None

    def set_cancel_mode(self):
        self.cancel_reports = True

    def set_preview_survey(self):
        self.preview_survey = True

    def has_mapper_canvas(self):
        return self.mapper_canvas is not None

    def set_mapper_hook( self, hook ):
        self.mapper_hook    = hook

    def change_log_to(self, analysis_folder):
        from molass_legacy.KekLib.ChangeableLogger import Logger

        self.app_logfile    = '%s/%s.log' % ( analysis_folder, exe_name() )
        if self.parent.tmp_logger is None:
            # TODO: investigate this case
            pass
        else:
            self.parent.tmp_logger.moveto( self.app_logfile )
            del self.parent.tmp_logger
            self.parent.tmp_logger = None
        self.app_logger     = Logger( self.app_logfile )

    def do_analysis(self, sd, pre_recog, md, analysis_folder, analysis_name, devel=False):
        # delayed imports to reduce initial load time
        from molass_legacy.KekLib.OurTkinter import Tk
        from molass_legacy.KekLib.ReadOnlyText import ReadOnlyText
        from molass_legacy.SerialAnalyzer.ProgressInfoUtil import estimate_init_max_dist
        from molass_legacy.KekLib.ProgressInfo import ProgressInfo
        from molass_legacy.KekLib.ProgressInfoDialog import ProgressInfoDialog
        # import Extrapolation
        from .AnalyzerDialogProxy import AnalyzerDialogProxy
        from molass_legacy.SerialAnalyzer.InputOutputDisplay import InputOutputDisplay
        from molass_legacy.SerialAnalyzer.AnalyzerUtil import compute_conc_factor_util
        from molass_legacy.Mapping.MapperConstructor import create_mapper        
        from molass_legacy.Mapping.ElutionMapperCanvas import ElutionMapperCanvas

        """
        in earlier versions than 20210604, pre_recog.set_info() has been repeated here.
        revised so as not to repeat that recognition
        """
        analysis_copy = sd._get_analysis_copy_impl(pre_recog)

        self.mapper_canvas  = None

        self.progress_dialog = None

        parent  = self.parent
        parent.config(cursor='wait')
        parent.update()

        self.data_folder    = get_setting( 'in_folder' ).replace( '\\', '/' )
        tester_info = self.dialog.tester_info
        if tester_info is None:
            tester_info_log = ''
        else:
            tester_info_log = ' with test pattern ' + str( tester_info.test_pattern )
        self.app_logger.info( "start analysis for " + self.data_folder + tester_info_log )

        self.set_controller_env( analysis_folder )

        self.analysis_copy = analysis_copy
        self.judge_holder = JudgeHolder(sd, pre_recog, analysis_copy)

        self.mapper = create_mapper(self.parent, self.analysis_copy, sd, pre_recog,
                                    callbacks=self.judge_holder.get_callbacks(),
                                    analyzer_dialog=self.parent, logger=self.app_logger)

        self.initial_navigation = True
        conc_factor = compute_conc_factor_util()

        sd_copy = self.analysis_copy.get_exec_copy(self.mapper, conc_factor)
        if SD_DEBUG:
            debugger = SdDebugger()
            debugger.save_info(self.analysis_copy)
            debugger.save_info(sd_copy)

        parent.config(cursor='')
        parent.update()

        self.mfc_dialog = None
        while True:
            if self.parent.use_mtd_conc:
                applied = self.do_microfluidic_analysis()
                if applied:
                    correction_necessity = False    # temporary
                else:
                    return
            else:
                self.mapper_canvas = ElutionMapperCanvas(self.dialog, sd_copy, sd, pre_recog, self.mapper, self.judge_holder, initial_navigation=self.initial_navigation )
                self.mapper_canvas.show()
                self.initial_navigation = False
                if self.mapper_canvas.applied:
                    # TODO
                    self.mapper =  self.mapper_canvas.mapper
                    correction_necessity = self.mapper_canvas.get_xray_correction_necessity()
                else:
                    return

            if SD_DEBUG:
                debugger.save_info(sd_copy)

            try:
                dialog = AnalyzerDialogProxy(parent, sd_copy, self.mapper, self.judge_holder, correction_necessity)
                dialog.apply()
            except:
                log_exception(self.app_logger, "AnalyzerDialogProxy Error: ")
                assert False

            if self.cancel_reports:
                self.cleanup_on_cancel()
                return

            break

        if self.mapper_canvas.ok_debugging:
            self.do_minimal_to_debug()
            return

        range_type      = 4
        min_value       = None
        preview_params = get_setting('preview_params')
        pdata, popts = preview_params
        if pdata.is_for_sec:
            # for compatibility with DecompEditorDialog concerning Preview
            analysis_range_info = get_setting('analysis_range_info')
            analysis_ranges = analysis_range_info.get_ranges()
        else:
            analysis_ranges = pdata.get_analysis_ranges()

        self.serial_data = sd_copy
        exec_sd = sd_copy
        mapped_info = self.mapper.get_mapped_info()

        parent.analysis_button.config( state=Tk.DISABLED )

        zx_flag = True

        scattering_correction = get_setting( 'scattering_correction' )
        num_files = len( exec_sd.datafiles )
        init_max_dict = estimate_init_max_dist( num_files, scattering_correction, analysis_ranges, zx_flag )

        pinfo = ProgressInfo( init_max_dict )
        stream_labels = [   "Baseline Correction  ",
                            "Guinier Analysis ",
                            "Extrapolation    " ]

        def description_cb( not_used_parent, frame ):
            dframe = Tk.Frame( frame )
            dframe.pack( fill=Tk.X, pady=10 )

            disp_label = Tk.Label( dframe, text="Input / Output", width=12, anchor=Tk.W )
            disp_label.grid( row=0, column=0, sticky=Tk.N + Tk.W )

            disp_frame = InputOutputDisplay( dframe, value_width=560 )
            disp_frame.grid( row=0, column=1, padx=5 )

        if self.parent.env_info.excel_is_available:
            text = None
            def plot_init_cb(not_used_parent, frame):
                nonlocal text
                text = ReadOnlyText(frame, relief=Tk.FLAT, height=12, bg='white', fg='orange')
                text.tag_config('CENTER_LARGE', justify=Tk.CENTER, font=('', 20))
                text.insert(Tk.END, "Caution:\n"
                                    "Do not open other Excel books by\n"
                                    "double-clicking during this execution\n",
                                    'CENTER_LARGE')
                text.tag_config('CENTER', justify=Tk.CENTER)
                text.insert(Tk.END, "\n"
                                    "since it may result in a confusion stated below.\n"
                                    "On double-click, Windows tries to use the Excel instance we created\n"
                                    "and it will be simultaneuosly controlled by the program and your manipulation\n"
                                    "and that can be confusing. It is hard to persuade Windows not to do that.",
                                    'CENTER')
                text.pack(fill=Tk.X)
            def plot_final_cb():
                text.pack_forget()
        else:
            plot_init_cb = None
            plot_final_cb = None

        self.boundary_test_deferred = False
        refresh_interval = 20 if get_dev_setting( 'use_simpleguinier' ) == 1 else 50
        self.progress_dialog = ProgressInfoDialog( self.dialog.parent,
                                    "Execution Progress", pinfo,
                                    stream_labels=stream_labels,
                                    description_cb=description_cb,
                                    plot_init_cb=plot_init_cb,
                                    refresh_log_cb=self.refresh_log_cb,
                                    plot_final_cb=plot_final_cb,
                                    is_alive_cb=self.is_alive_cb,
                                    refresh_interval=refresh_interval,
                                    logger=self.app_logger,
                                    )
        self.logger_visible = True

        def start_controller_closure():
            self.start_controller( exec_sd, range_type, analysis_ranges, preview_params, min_value, zx_flag, mapped_info )

        # start controller after progress_dialog.show to get all logs from controller
        parent.after( 100, start_controller_closure )

        btn_text = self.progress_dialog.show( logger=self.app_logger )
        self.logger_visible = False
        if btn_text == 'Cancel':
            self.controller.kill()
            self.controller.terminate()
            self.app_logger.info( 'Canceled.' )

        cleaner = self.parent.cleaner
        if cleaner is not None:
            cleaner.set_cleanup(None)
        self.progress_dialog = None
        self.app_logger = None      # release the log file at this timing
        parent.analysis_button.config( state=Tk.NORMAL )
        parent.update_plot_button_state()

    def get_final_std_diff( self ):
        # self.mapper_canvas.final_std_diff should have been set by show_mapping_figure_func call
        print( 'Analyzer: get_final_std_diff' )
        return self.mapper_canvas.final_std_diff

    def set_controller_env( self, analysis_folder ):
        conc_folder = get_setting( 'uv_folder' ).replace( '\\', '/' )
        self.conc_file   = get_setting( 'uv_file'   )
        if self.conc_file.find( '*' ) >= 0:
            self.conc_file = None
        outp_folder = analysis_folder
        self.conc_folder    = conc_folder
        self.work_folder    = outp_folder
        self.temp_folder    = self.work_folder + '/.temp'
        self.make_temp_folder()
        set_setting( 'temp_folder', self.temp_folder )
        self.guinier_folder = self.work_folder + '/.guinier_result'
        self.stamp_file     = os.path.join( self.guinier_folder, '--stamp.csv' )

    def make_temp_folder( self ):
        """
        TODO:
            unify with SerialController
        """
        try:
            clear_dirs_with_retry( [self.temp_folder] )
        except Exception as exc:
            etb = ExceptionTracebacker()
            self.app_logger.error( etb )
            raise exc

    def start_controller( self, serial_data, range_type, analysis_ranges, preview_params, min_value, zx_flag, mapped_info ):
        # delayed imports to reduce initial load time
        from molass_legacy.SerialAnalyzer.SerialControlInfo import SerialControlInfo
        from molass_legacy.SerialAnalyzer.SerialController import SerialController

        """
        temporarily commented out due to the following error
                pickle.dump( self.dictionary, pf )
            TypeError: cannot pickle '_tkinter.tkapp' object
        """
        # save_settings()     # make sure that settings be consistent in the execution thread

        result_book = get_setting( 'result_book' )
        book_file   = '/'.join( [ self.work_folder, result_book ] )
        serial_file = self.work_folder + '/.temp/--serial_result.csv'
            # TODO: unify .temp

        self.app_logger.info("ATSAS avalability is %s", str(self.parent.env_info.atsas_is_available))

        known_info_list = get_setting('known_info_list')
        control_info = SerialControlInfo(  self.data_folder,
                                        self.conc_folder,
                                        self.work_folder,
                                        self.temp_folder,
                                        conc_file=self.conc_file,
                                        serial_data=serial_data,
                                        guinier_folder=self.guinier_folder,
                                        stamp_file=self.stamp_file,
                                        serial_file=serial_file, book_file=book_file,
                                        zx=zx_flag,
                                        mapped_info=mapped_info,
                                        min_analysis_value=min_value,
                                        range_type=range_type,
                                        analysis_ranges=analysis_ranges,
                                        preview_params=preview_params,
                                        maintenance_log=self.parent.maintenance_log,
                                        env_info=self.parent.env_info,
                                        known_info_list=known_info_list,
                                        cleaner=self.parent.cleaner,
                                        )

        self.controller = SerialController(control_info)
        self.controller.start()

    def is_alive_cb(self):
        return self.controller.exe_thread.is_alive()

    def cleanup_on_cancel( self ):
        self.parent.update()
        self.app_logger = None      # release the log file when canceled
        log_size = os.path.getsize( self.app_logfile )
        yn = MessageBox.askyesno( "Log file removal confirmation",
                "Would you like to remove the log in %s (%d bytes)?" % ( self.app_logfile, log_size ),
                parent=self.parent )
        if yn:
            os.remove( self.app_logfile )

    def refresh_log_cb( self, text, level ):
        if text.find( 'deferred' ) >= 0:
            self.boundary_test_deferred = True

    def get_boundary_test_deferred( self ):
        return self.boundary_test_deferred

    def get_logfile( self ):
        return self.app_logfile

    def do_microfluidic_analysis(self):
        from Microfluidics.MctDecompDialog import MctDecompDialog
        xdata = self.parent.measured_data.get_xdata_for_mct()
        in_folder = self.data_folder
        self.mfc_dialog = MctDecompDialog(self.parent, xdata, in_folder)
        self.mfc_dialog.decompose()
        self.mfc_dialog.show()
        applied = self.mfc_dialog.applied
        self.mfc_dialog.destroy()
        return applied

    def do_minimal_to_debug(self):
        print("do_minimal_to_debug")
