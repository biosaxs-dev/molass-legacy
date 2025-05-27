"""

    StageGuinier.py

    Copyright (c) 2016-2024, SAXS Team, KEK-PF

"""
import os
import time
import numpy                as np
from openpyxl               import Workbook
from molass_legacy.KekLib.NumpyUtils import np_savetxt, simply_safe_sprintf
from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
from molass_legacy.AutorgKek.KekToolsGP import AutorgKek, ErrorResult
from molass_legacy.AutorgKek.LightObjects import LightIntensity, LightResult
from molass_legacy.AutorgKek.AtsasTools import autorg as autorg_atsas
from molass_legacy.KekLib.ProgressInfo import put_info
from .ProgressInfoUtil import STREAM_GUINIER
from .AnalyzerUtil import make_guinier_result_stamp, exists_guinier_analysis_result
from molass_legacy.Reports.ReportUtils import get_header_record, make_record, make_guinier_analysis_book
from molass_legacy.GuinierAnalyzer.SimpleGuinier import QUALITY_BOUNDARY_QRG
from molass_legacy.GuinierAnalyzer.AutorgKekAdapter import AutorgKekAdapter
from molass_legacy.Test.Tester import compare_tester_rg
from molass_legacy.Reports.DefaultFont import set_default_font
set_default_font()

NUM_LATEST_RGS          = 5
LOG_DEFATED_MIN_STDEV   = 0.2
LOG_DEFATED_RATIO       = 2.0

def run_gunier_analysis(self, debug=False):
    print( 'run_gunier_analysis' )

    intensity_array_ = self.get_intensity_array()
    self.qvector = intensity_array_[0,:,0]

    self.guinier_result_array = []
    quality_array = []

    self.write_state( 'gunier_analysis-doing' )
    if self.using_avareged_files:
        datafiles_  = self.averaged_datafiles
    else:
        datafiles_  = self.datafiles

    fh = open( self.serial_file, 'w' )
    fh.write( get_header_record() + '\n' )
    num_successes = 0
    total_time = 0
    self.latest_rgs_atsas   = []
    self.latest_rgs_self    = []
    self.rg_ratio_range     = [ None, None ]
    self.head_gap_ratio_range   = [ None, None ]

    for i, datafile in enumerate( datafiles_ ):
        self.stop_check()

        self.current_datafile = datafile
        data = intensity_array_[ i, self.usable_slice, : ]
        # print( 'data.shape=', data.shape )
        try:
            t0 = time.time()
            if self.use_simpleguinier == 0:
                autorg_kek  = AutorgKek( data )
            else:
                autorg_kek  = AutorgKekAdapter( data )
            result_kek  = autorg_kek.run( robust=True, optimize=True )
            num_successes += 1
            total_time += time.time() - t0
        except Exception as exc:
            etb = ExceptionTracebacker()
            self.logger.error( etb )
            self.logger.error( 'current_datafile:' + str(self.current_datafile) )   # str to allow None
            result_kek = ErrorResult()

        result_atsas, result_atsas_eval = autorg_atsas( datafile )
        assert result_atsas is not None

        if result_atsas.Rg is not None and result_atsas_eval is None:
            self.logger.warning( 'unexpected None result_atsas_eval; error column value=0 may have caused this result' )

        compare_rg_stdev( self, [ result_kek.Rg, result_atsas.Rg ] )
        update_rg_ratio_range( self, result_kek )

        try:
            compare_tester_rg( i, result_kek.Rg, QUALITY_BOUNDARY_QRG )
        except:
            etb = ExceptionTracebacker()
            self.logger.error( etb )

        record = make_record( datafile, result_kek, result_atsas )
        fh.write( record + '\n' )
        fh.flush()

        # convert into light objects to facilitate garbage collection
        light_intensity = LightIntensity( autorg_kek.intensity )
        light_result    = LightResult( result_kek )

        self.guinier_result_array.append( [ light_intensity, light_result, result_atsas ] )
        quality = None if result_kek is None else result_kek.Quality

        if self.log_memory_usage == 1:
            m_info = self.process.memory_info()
            memory_usage    = " " + str( ( m_info.vms, ) )
        else:
            memory_usage    = ""

        _, file = os.path.split( datafile )

        if result_kek.Rg is None:
            time.sleep( 0.1 )   # to let the log window to handle this separately
            self.logger.warning( "Rg estimation failed in %s.%s" % ( file, memory_usage ) )
            time.sleep( 0.1 )   # to let the log window to handle this separately
        else:
            log_Rg      = simply_safe_sprintf( '%.3g', result_kek.Rg )
            log_quality = simply_safe_sprintf( '%.2g', quality )
            self.logger.info( "Rg from %s is estimated to be %s with quality %s.%s" % ( file, log_Rg, log_quality, memory_usage ) )

        if quality is None:
            quality = 0
        quality_array.append( quality )
        if debug:
            self.logger.info("put_info( (%d, 0), %d )", STREAM_GUINIER, i+1)
        put_info( (STREAM_GUINIER, 0), i+1 )
    fh.close()
    self.quality_array = np.array(quality_array)
    self.current_datafile = None

    self.logger.info( "average time for successful Rg evaluation was %.2g seconds" % ( total_time/num_successes ))
    self.logger.info( "rg_ratio_range=%s" % str( self.rg_ratio_range ) )
    self.logger.info( "head_gap_ratio_range=%s" % str( self.head_gap_ratio_range ) )
    self.write_state( 'gunier_analysis-done' )

    exists_guinier_analysis_result( self )  # to create folder
    save_guinier_result( self )

def compare_rg_stdev( self, rg_pair ):

    array_list = []
    for i, list_ in enumerate([self.latest_rgs_self, self.latest_rgs_atsas]):
        list_.append( rg_pair[i] )
        if len(list_) > NUM_LATEST_RGS:
            list_.pop(0)
        array_list.append( np.array( [ np.nan if v is None else v for v in list_ ] ) )

    if len( self.latest_rgs_self ) < 3:
        return

    stdev_k = np.std( array_list[0] )
    stdev_a = np.std( array_list[1] )

    if np.isnan(stdev_k) or np.isnan(stdev_a):
        return

    if False:
        if stdev_a > LOG_DEFATED_MIN_STDEV and stdev_k/stdev_a > LOG_DEFATED_RATIO:
            self.logger.warning( 'possibly defeated by atsas: sa stdev=%.3g, atsas stdev=%.3g' % ( stdev_k, stdev_a ) )

def update_rg_ratio_range( self, result_kek ):
    if result_kek.Rg is None:
        return

    if result_kek.rg_ratio_for_better_peak is None:
        return

    if self.rg_ratio_range[0] is None or result_kek.rg_ratio_for_better_peak < self.rg_ratio_range[0]:
        self.rg_ratio_range[0] = result_kek.rg_ratio_for_better_peak

    if self.rg_ratio_range[1] is None or result_kek.rg_ratio_for_better_peak > self.rg_ratio_range[1]:
        self.rg_ratio_range[1] = result_kek.rg_ratio_for_better_peak

    if self.head_gap_ratio_range[0] is None or result_kek.head_gap_ratio < self.head_gap_ratio_range[0]:
        self.head_gap_ratio_range[0] = result_kek.head_gap_ratio

    if self.head_gap_ratio_range[1] is None or result_kek.head_gap_ratio > self.head_gap_ratio_range[1]:
        self.head_gap_ratio_range[1] = result_kek.head_gap_ratio

def write_gunier_stamp( self ):
    fh = open( self.stamp_file, "w" )
    fh.write( make_guinier_result_stamp( self ) )
    fh.close()

def save_guinier_result( self ):
    if os.path.exists( self.guinier_file ):
        os.remove( self.guinier_file )
    os.rename( self.serial_file, self.guinier_file )
    write_gunier_stamp( self )
    np_savetxt( self.quality_file, self.quality_array )

def make_guinier_analysis_report( self ):
    # print( 'make_guinier_analysis_report' )

    if self.excel_is_available:
        wb = Workbook()
        ws = wb.active
    else:
        wb = self.result_wb
        ws = wb.create_sheet('Guinier Analysis')

    book_file = self.temp_folder + '/--serial_analysis-temp.xlsx'
    make_guinier_analysis_book( self.excel_is_available, wb, ws, book_file, self.guinier_file, self.c_vector,
                                    self.xr_j0,
                                    applied_ranges=self.applied_ranges,
                                    parent=self, logger=self.logger )
    self.temp_books.append( book_file )
