"""

    SerialController.py

    Copyright (c) 2016-2024, SAXS Team, KEK-PF

"""
import os
import glob
import threading
import logging
import psutil
import time
from openpyxl import Workbook

from molass_legacy.KekLib.BasicUtils import get_filename_extension
from .SerialDataUtils import save_xray_base_profiles
from .StageGuinier import run_gunier_analysis, make_guinier_analysis_report
from .StageExtrapolation import control_extrapolation
from .StageSummary import do_summary_stage
from molass_legacy.KekLib.ExcelCOM import cleanup_created_excels
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting, save_settings
from molass_legacy.KekLib.NumpyUtils import np_savetxt
from molass_legacy.KekLib.BasicUtils import ( clear_dirs_with_retry, mkdirs_with_retry, ordinal_str )
from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
from .AnalysisRangeManager   import get_analysis_ranges_for_exec
from molass_legacy.DataStructure.AnalysisRangeInfo import convert_to_paired_ranges, report_ranges_from_analysis_ranges
from molass_legacy.KekLib.ProgressInfo import put_info, send_stop, on_stop_raise, put_error, STATE_FATAL, SAFE_FINISH
from .ProgressInfoUtil import ( NUM_SHEET_TYPES,
                                STREAM_BASECOR, STREAM_GUINIER, STREAM_ZERO_EX, STREAM_BASECOR_MAX,
                                ProgressCallback )
from DevSettings import get_dev_setting
from DriftSimulation import apply_simulated_baseline_correction
from molass_legacy.Test.Tester import ( write_to_tester_log,
                            create_log_queue, open_dev_log, write_to_dev_log, close_dev_log,
                            write_from_log_queue
                            )
from molass_legacy.Reports.DefaultFont import set_default_font
set_default_font()

DEBUG = False

NUM_ZX_STEPS        = ( 2 + 1 + 2 ) * 2 + 1
NUM_ZX_BOOK_STEPS   = 2

class ExecInfo:
    def __init__( self, **entries ): 
        self.__dict__.update(entries)

    def __str__( self ):
        return str( self.state )

class SerialController:
    def __init__( self, control_info):
        self.control_info = control_info
        self.cleaner = control_info.cleaner
        self.exe_thread     = None

    def start( self, detail=False ):
        serial_data = self.control_info.serial_data
        assert serial_data is not None

        serial_data.wait_until_ready()

        """
        temporarily commented out due to the following error
                pickle.dump( self.dictionary, pf )
            TypeError: cannot pickle '_tkinter.tkapp' object
        """
        # temporary fix for inconsystency bug or risk of persistent memory between threads
        save_settings()

        self.executer = SerialExecuter()
        if self.cleaner is not None:
            self.cleaner.set_cleanup(self.executer.cleanup)

        self.exe_thread = threading.Thread(
                                target=self.executer.run,
                                name='ExecutionThread',
                                args=[self.control_info],
                                )
        self.exe_thread.start()

    def is_alive(self):
        return self.exe_thread.is_alive()

    def kill( self ):
        send_stop()

    def terminate( self ):
        from molass_legacy.KekLib.ThreadUtils import terminate_thread
        self.logger = logging.getLogger( __name__ )
        self.logger.info( 'terminate' )
        if self.exe_thread is None:
            return

        self.exe_thread.join( timeout=5 )
        if self.exe_thread.is_alive():
            self.logger.warning( 'terminate_thread' )
            terminate_thread( self.exe_thread )
        cleanup_created_excels()

class SerialExecuter:
    def __init__(self):
        self.current_datafile   = None
        self.log_memory_usage   = get_dev_setting( 'log_memory_usage' )
        self.process    = psutil.Process()
        self.excel_list = []
        self.logger     = logging.getLogger( __name__ )

    def run(self, control_info):
        from molass_legacy.KekLib.ExcelCOM import CoInitialize, ExcelComClient

        self.temp_folder    = control_info.temp_folder
        self.more_multicore = control_info.more_multicore
        self.make_temp_folder()

        if self.more_multicore:
            from molass_legacy.ExcelProcess.ExcelTeller import ExcelTeller
            self.teller = ExcelTeller(log_folder=self.temp_folder)
            self.logger.info('teller created with log_folder=%s', control_info.temp_folder)

        self.parent = control_info.parent
        serial_data = control_info.serial_data
        self.stream  = STREAM_BASECOR
        scattering_correction = get_setting( 'scattering_correction' )
        start_time0 = time.time()
        if scattering_correction == 1:
            base_drift_params = get_setting( 'base_drift_params' )

            try:
                if base_drift_params is None:
                    max_progress    = len( serial_data.qvector ) - 1
                    progress_cb     = ProgressCallback( max_progress )

                    enable_xb_save  = get_dev_setting( 'enable_xb_save' )
                    return_base     = enable_xb_save == 1
                    corrected_base  = serial_data.apply_baseline_correction( control_info.mapped_info, progress_cb=progress_cb, return_base=return_base )
                    if return_base:
                        save_xray_base_profiles( serial_data, corrected_base )
                else:
                    max_progress    = len( serial_data.datafiles ) - 1
                    progress_cb     = ProgressCallback( max_progress )

                    apply_simulated_baseline_correction( base_drift_params, serial_data.intensity_array, progress_cb=progress_cb )

                self.logger.info( 'Baseline correction (degree=%d) applied to scattering data.' % ( get_setting('baseline_degree') ) )
            except ( RuntimeError ) as exc:
                self.logger.warning( exc )
            except:
                etb = ExceptionTracebacker()
                self.logger.error( str(etb) )
                put_error( self.stream, error_state=STATE_FATAL )
                return

            progress_cb( max_progress )

        start_time1 = time.time()
        self.seconds_correction = int(start_time1 - start_time0)
        # self.input_smoothing = get_setting( 'input_smoothing' )
        self.input_smoothing = 1

        self.temp_books     = []
        self.temp_books_atsas   = []
        self.data_folder    = control_info.data_folder
        self.conc_folder    = control_info.conc_folder
        self.work_folder    = control_info.work_folder
        self.maintenance_log    = control_info.maintenance_log
        self.maintenance_mode   = control_info.maintenance_log is not None
        self.env_info       = control_info.env_info
        self.excel_is_available = self.env_info.excel_is_available
        self.atsas_is_available = self.env_info.atsas_is_available
        self.guinier_folder = control_info.guinier_folder
        d_, f_ = os.path.split( control_info.serial_file )
        self.guinier_file   = os.path.join( self.guinier_folder, f_ )
        self.stamp_file     = control_info.stamp_file
        self.quality_file   = os.path.join( self.guinier_folder, '--quality_array.csv' )
        self.serial_file    = control_info.serial_file
        self.book_file      = control_info.book_file
        self.min_analysis_value    = control_info.min_analysis_value
        self.range_type     = control_info.range_type
        self.zx             = control_info.zx
        self.mapped_info    = control_info.mapped_info
        self.known_info_list = control_info.known_info_list
        self.zx_out_folder  = None
        self.use_simpleguinier  = get_dev_setting( 'use_simpleguinier' )
        if self.use_simpleguinier == 1:
            current_quality_weighting = get_setting( 'quality_weighting' )
            new_weighting = [
                                0.2,    # basic_quality
                                0.2,    # positive_score
                                0.2,    # fwd_consistency
                                0.2,    # rg_stdev_score
                                0.2,    # q_rg_score
                              ]
            set_setting( 'quality_weighting', new_weighting )

        self.stream  = STREAM_GUINIER
        self.zx_summary_list = []
        self.zx_summary_list2 = []

        try:
            if self.maintenance_mode:
                self.logger.warning( 'executing in the maintenance mode' )

            if self.log_memory_usage == 1:
                m_info = self.process.memory_info()
                memory_usage    = " " + str( ( m_info.vms, ) )
            else:
                memory_usage    = ""

            self.preview_params = control_info.preview_params
            pdata = self.preview_params[0]
            self.doing_sec = pdata.is_for_sec

            if self.doing_sec:
                self.xr_j0 = serial_data.xr_j0
            else:
                self.xr_j0 = pdata.sd.xr_j0

            analysis_ranges = control_info.analysis_ranges
            range_type = control_info.range_type
            data_folder = control_info.data_folder
            conc_folder = control_info.conc_folder
            book_file = control_info.book_file

            if analysis_ranges is None:
                # set_analysis_range must be delayed for the range_type '3'
                self.report_ranges = None
            else:
                self.set_analysis_range( range_type, analysis_ranges )
                self.report_ranges = report_ranges_from_analysis_ranges(self.xr_j0, self.applied_ranges)

            self.logger.info( 'Xray scattering folder=' + data_folder )
            self.logger.info( 'UV absorbance folder=' + conc_folder )
            self.logger.info( 'concetration factor has been set to %g' % serial_data.conc_factor )
            self.logger.info( 'range_type=%d and analysis_ranges=%s%s' % ( range_type, self.report_ranges, memory_usage ) )

            write_to_tester_log( 'book_file=' + book_file + '\n' )
            if self.maintenance_mode:
                create_log_queue()
                open_dev_log( self.maintenance_log )
                write_to_dev_log( 'Doing ' + data_folder + '\n' )

            self.save_settings()

            self.set_control_data( serial_data )

            write_to_tester_log( 'guinier start\n' )

            run_gunier_analysis( self )

            if analysis_ranges is None:
                self.set_analysis_range( range_type, analysis_ranges )
            else:
                pass
                # already set at the start (above)

            if self.excel_is_available:
                if self.more_multicore:
                    self.excel_client = None
                else:
                    CoInitialize()
                    self.excel_client = ExcelComClient()
                self.result_wb = None
            else:
                self.excel_client = None
                self.result_wb = Workbook()
                # self.result_wb.remove_sheet(self.result_wb.active)

            make_guinier_analysis_report( self )

            write_to_tester_log( 'guinier finish\n' )
            start_time2 = time.time()
            self.seconds_guinier = int(start_time2 - start_time1)

            self.stream  = STREAM_ZERO_EX
            if self.zx:
                control_extrapolation( self )

            start_time3 = time.time()
            self.seconds_extrapolation = int(start_time3 - start_time2)

            do_summary_stage( self )

            start_time4 = time.time()
            self.seconds_summary = int(start_time4 - start_time3)

            self.cleanup()

            if self.log_memory_usage == 1:
                m_info = self.process.memory_info()
                memory_usage    = " " + str( ( m_info.vms, ) )
            else:
                memory_usage    = ""

            self.logger.info( 'Finished.' + memory_usage )
            self.logger.info( "The report has been saved to '%s'" % ( book_file ) )

            put_info( (STREAM_ZERO_EX,2000), SAFE_FINISH )

            put_info( (STREAM_ZERO_EX,2000), 1 )

            if self.maintenance_mode:
                write_from_log_queue()
                write_to_dev_log( 'Finished ' + data_folder + '\n' )
                close_dev_log()

        except Exception as exc:
            etb = ExceptionTracebacker()
            self.logger.error( etb )
            if self.current_datafile is not None:
                self.logger.error( 'current_datafile:' + str(self.current_datafile) )   # str to allow None
            self.error_cleanup()
            self.error_wrapup( book_file )
            put_error( self.stream, error_state=STATE_FATAL )

        if self.use_simpleguinier == 1:
            set_setting( 'quality_weighting', current_quality_weighting )

    def stop_check( self ):
        def log_closure(cmd):
            # this closure is expected to be called only in cancel operations
            self.logger.info("cmd=%s", str(cmd))
        on_stop_raise(cleanup=self.error_cleanup, log_closure=log_closure)

    def make_temp_folder( self ):
        try:
            clear_dirs_with_retry( [self.temp_folder] )
        except Exception as exc:
            etb = ExceptionTracebacker()
            self.logger.error( etb )
            raise exc

    def compute_num_guinier_steps( self, num_datafiles ):
        return 1 + num_datafiles + 5

    def set_control_data( self, serial_data ):
        self.serial_data = serial_data
        self.datafiles = serial_data.datafiles 
        self.num_datafiles = len( serial_data.datafiles )
        self.num_peaks_to_exec = len( self.mapped_info.x_ranges )

    def get_intensity_array( self ):
        """
            input_smoothing
                1 : 
                2 : 
        """

        self.serial_data.apply_data_reduction()
        # if get_dev_setting( 'no_usable_q_limit' ) == 1:
        if True:
            self.usable_slice = slice( 0, self.serial_data.intensity_array.shape[1] )
        else:
            self.usable_slice = self.serial_data.get_usable_slice()
            if self.usable_slice.stop is None:
                predicate = ' is not set'
            else:
                predicate = ' is set at ' + str(self.usable_slice.stop)
            self.logger.info( 'Usable Xray scattering data limit' + predicate )

        """
            concentration vector variation

            SerialData          oc_vector
                                mc_evector

            SerialControler     c_vector  ( mc_vector or smoothed mc_vector )

            ZeroExtrapolator    c_vector
        """

        self.using_avareged_files = False
        if self.input_smoothing == 1:
            num_curves_averaged = get_setting( 'num_curves_averaged' )
            intensity_array_, average_slice_array, c_vector = self.serial_data.get_averaged_data( num_curves_averaged )
            assert c_vector is not None

            # save
            save_averaged_data = get_setting( 'save_averaged_data' )
            if save_averaged_data == 1:
                self.using_avareged_files = True        # TODO: check consistency
                self.save_smoothed_data( intensity_array_, average_slice_array )
        else:
            intensity_array_ = self.serial_data.intensity_array
            c_vector = self.serial_data.mc_vector

        if self.doing_sec:
            self.c_vector = c_vector
        else:
            # temp fix; TODO: do this correctly
            pdata = self.preview_params[0]
            self.c_vector = pdata.mc_vector

        return intensity_array_

    def save_smoothed_data( self, intensity_array, average_slice_array ):
        self.averaged_data_folder = get_setting( 'averaged_data_folder' )
        if not os.path.exists( self.averaged_data_folder ):
            mkdirs_with_retry( self.averaged_data_folder )

        avg_file_postfix = get_setting( 'avg_file_postfix' )
        extension = '.' + get_filename_extension( self.datafiles[0] )
        self.averaged_datafiles = []
        for i, intensity in enumerate( intensity_array ):
            _, filename = os.path.split( self.datafiles[i] )
            # print( 'saving', filename )
            filepath = '/'.join( [ self.averaged_data_folder, filename.replace( extension, avg_file_postfix + extension ) ] )
            with open( filepath, "wb" ) as fh:
                slice_ = average_slice_array[i]
                fh.write( str.encode( '# Created by averaging the following %d files with []-numbering starting from 1.\n' % ( slice_.stop - slice_.start ) ) )
                for j in range( slice_.start, slice_.stop ):
                    fh.write( str.encode( '# [%d] %s\n' % ( j+1, self.datafiles[j] ) ) )
                fh.write( str.encode( "#\n# Q\tIntensity\tError\n" ) )
                fh.close()
            np_savetxt( filepath, intensity, mode="a" )
            self.averaged_datafiles.append( filepath )

    def set_analysis_range( self, range_type, analysis_ranges ):
        ranges = get_analysis_ranges_for_exec( self, range_type, analysis_ranges, self.logger )
        # self.logger.info( arg_join( 'set_analysis_range: ranges=', ranges ) )

        # change ranges to internal ranges
        self.applied_ranges = ranges

    def write_state( self, state ):
        self.clear_state()
        path = self.temp_folder + '/--state-' + state
        fh = open( path, "w" )
        fh.close()

    def clear_state( self ):
        state_files = glob.glob( self.temp_folder + '/--state*' )
        for path in state_files:
            os.remove( path )

    def error_wrapup( self, book_file ):
        self.logger.warning( 'Error wrap-up start.' )
        try:
            self.merge_books()
            self.cleanup()
            self.logger.warning( 'Error wrap-up done.' )
            if len(self.temp_books) > 0:
                self.logger.warning( "The incomplete report has been saved to '%s'" % ( book_file ) )
        except Exception as exc:
            etb = ExceptionTracebacker()
            self.logger.error( etb )
            self.logger.info( "Temporary files are left as worked for investigation." )

    def save_settings( self ):
        save_folder = self.temp_folder.replace( '.temp', '.save' )
        clear_dirs_with_retry( [ save_folder ] )
        save_path = save_folder + '/serial_settings.dump'
        save_settings( file=save_path )

    def cleanup(self):
        self.logger.info("Cleanup started. This may take some time (not more than a few minutes). Please be patient.")

        if self.more_multicore:
            self.teller.stop()   # must be done before the removal below of the temp books

        if self.excel_is_available:
            if self.more_multicore:
                pass
            else:
                from molass_legacy.KekLib.ExcelCOM import CoUninitialize
                self.excel_client.quit()
                self.excel_client = None
                CoUninitialize()

            for path in self.temp_books + self.temp_books_atsas:
                os.remove( path )

        self.logger.info("Cleanup done.")

    def error_cleanup(self):
        self.cleanup()
        cleanup_created_excels()
