"""

    ReportUtils.py

    Copyright (c) 2016-2024, SAXS Team, KEK-PF

"""
import os
import numpy as np
from molass_legacy.AutorgKek.Quality import fit_consistency_func, compute_atsas_fit_consistency
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.KekLib.NumpyUtils import np_savetxt, np_loadtxt
from molass_legacy.KekLib.ProgressInfo import put_info
from molass_legacy.SerialAnalyzer.ProgressInfoUtil import STREAM_GUINIER
from .GuinierAnalysisResultBook import GuinierAnalysisResultBook
from .ZeroExtrapolationResultBook import ZeroExtrapolationResultBook
from .ZeroExtrapolationOverlayBook import ZeroExtrapolationOverlayBook
from .SummaryBook import SummaryBook
from molass.Reports.Migrating import COLNAMES, make_gunier_row_values

BASIC_QUALITY   = 2
Q_RG_SCORE      = 5
I0          = 10
RG          = 11
GPFIT_RG    = 14
GPFIT_I0    = 19
I0_STDEV    = 27
RG_STDEV    = 28
ATSAS_BEGIN = 29
ATSAS_I0    = 29
ATSAS_RG    = 30
ATSAS_QUALITY   = 32

SELECT_COLUMS = [ 2, 3, 5, 6, 7, ATSAS_QUALITY, 10, GPFIT_I0, ATSAS_I0, 11, GPFIT_RG, ATSAS_RG ]

def toFloat( value ):
    return 0 if value == '' else float( value )

class RestoredResult:
    def __init__( self, saved_line ):
        rec_row = saved_line.split( ',' )

        self.I0         = toFloat( rec_row[I0] )
        self.I0_        = self.I0
        self.I0_stdev   = toFloat( rec_row[I0_STDEV] )
        self.Rg         = toFloat( rec_row[RG] )
        self.Rg_stdev   = toFloat( rec_row[RG_STDEV] )
        self.Rg_        = self.Rg
        self.gpfit_Rg   = toFloat( rec_row[GPFIT_RG] )
        self.gpfit_I0   = toFloat( rec_row[GPFIT_I0] )

        # TODO: add From, To
        self.From       = 0     # TBD
        self.To         = 0     # TBD

def get_header_record():
    return ','.join( COLNAMES )

def make_record( path, result, result_atsas, result_atsas_eval=None ):
    results = make_gunier_row_values(result, result_atsas)
    # print( 'Rg=', result.Rg )

    def convert_to_float( x ):
        if x is None:
            x = 0

        try:
            xstr = '%g' % x
        except:
            xstr = 'NA'

        return xstr

    folder, file = os.path.split( path )

    rec = ','.join( [ folder, file ]
                    + [ '%g' % ( 0 if x is None else x) for x in results ] )
                    # + list( map( convert_to_float, results ) ) )

    return rec

def load_guinier_result( result_file, mc_vector ):
    fh = open( result_file )
    array = []
    for j, line in enumerate(fh, start=-1):
        if j == -1:
            # skip header record
            continue
        in_row = line.split( ',' )
        if len(in_row) <= ATSAS_BEGIN:
            in_row += [ '' ] * (len(COLNAMES) - len(in_row))
        c = mc_vector[j]
        out_row = in_row[0:2] + [ c ]
        out_row += [ toFloat(in_row[i]) for i in SELECT_COLUMS]

        array.append( out_row )

    fh.close()
    return array

def just_load_guinier_result( result_file ):
    fh = open( result_file )
    array = []
    for j, line in enumerate(fh, start=-1):
        if j == -1:
            # skip header record
            continue
        in_row = line.split( ',' )
        if len(in_row) <= ATSAS_BEGIN:
            in_row += [ '' ] * (len(COLNAMES) - len(in_row))

        out_row = in_row[0:2]
        out_row += [ toFloat(v) for v in in_row[2:] ]

        array.append( out_row )

    fh.close()
    return array


def make_guinier_analysis_book(
            excel_is_available, wb, ws,
            book_file, serial_file, mc_vector, j0,
            applied_ranges=None,
            cmd_queue=None, parent=None, logger=None ):

    assert( applied_ranges is not None )

    ranges = []
    for range_ in applied_ranges:
        fromto_list = range_.get_fromto_list()
        ranges.append( [ fromto_list[0][0], fromto_list[-1][1] ] )

    put_info( (STREAM_GUINIER, 1), 1 )

    array = load_guinier_result( serial_file, mc_vector )
    put_info( (STREAM_GUINIER, 1), 2 )

    book = GuinierAnalysisResultBook( wb, ws, array, j0, parent=parent )
    put_info( (STREAM_GUINIER, 1), 3 )

    if excel_is_available:
        book.save( book_file )

    put_info( (STREAM_GUINIER, 1), 4 )

    if excel_is_available:
        book.add_annonations( book_file, ranges )
    else:
        if logger is not None:
            _, file = os.path.split(book_file)
            logger.warning( 'excel book formatting has been skipped for %s.', file )

    put_info( (STREAM_GUINIER, 1), 5 )

def make_zero_extrapolation_book(
            excel_is_available, wb, ws,
            book_file, mjn, indeces, c_rg_iz, qvector, data_matrix, j0,
            ze_array, param_array, orig_param_array,
            guinier_boundary,
            almerge_result, atsas_result,
            boundary_j, parent=None, lrf_info=None,
            logger=None ):

    prefix = book_file.replace( '.xlsx', '' )
    # np_savetxt( prefix + '-data_matrix.csv', data_matrix )
    # np_savetxt( prefix + '-ze_array.csv', ze_array )
    book = ZeroExtrapolationResultBook( wb, ws, mjn, indeces, c_rg_iz, qvector, data_matrix, j0,
                                        ze_array, param_array, orig_param_array,
                                        guinier_boundary,
                                        almerge_result, atsas_result,
                                        boundary_j, parent=parent, lrf_info=lrf_info )
    if excel_is_available:
        book.save( book_file )
        book.add_format_setting( book_file )
    else:
        if logger is not None:
            _, file = os.path.split(book_file)
            logger.warning( 'excel book formatting has been skipped for %s.', file )

def dump_zx_report_args( m, j, indeces, c_rg_iz, qvector, data_matrix, ze_array, param_array, orig_param_array, exz_array ):
    np_savetxt( 'indeces-%d-%d.csv' % (m, j), indeces )
    for k, list_k in enumerate( c_rg_iz ):
        for n, array in enumerate( list_k ):
            # print( (k,n), 'array.shape=', array.shape )
            # if (k,n) == (1,1):
            #    print( 'array=', array )
            if len( array.shape ) == 1:
                array_ = [ np.nan if a is None else a for a in array ]
            elif len( array.shape ) == 2:
                array_ = np.array( array )
                for p in range( array.shape[0] ):
                    for q in range( array.shape[1] ):
                        if array[p,q] is None:
                            array_[p,q] = np.nan
            else:
                assert( False )
            np_savetxt( 'c_rg_iz-%d-%d-%d-%d.csv' % (m, j, k, n), np.array(array_) )
    np_savetxt( 'qvector-%d-%d.csv' % (m, j), qvector )
    np_savetxt( 'data_matrix-%d-%d.csv' % (m, j), data_matrix )
    np_savetxt( 'ze_array-%d-%d.csv' % (m, j), ze_array )
    np_savetxt( 'param_array-%d-%d.csv' % (m, j), param_array )
    if orig_param_array is not None:
        np_savetxt( 'orig_param_array-%d-%d.csv' % (m, j), orig_param_array )
    if exz_array is not None:
        np_savetxt( 'exz_array-%d-%d.csv' % (m, j), exz_array )

def load_zx_report_args( folder, m, j ):
    indeces_csv = folder + '/indeces-%d-%d.csv' % ( m, j )
    indeces, _ = np_loadtxt( indeces_csv )

    c_rg_iz = []
    for k in range(3):
        list_ = []
        for n in range(2):
            c_rg_iz_csv =  folder + '/c_rg_iz-%d-%d-%d-%d.csv' % ( m, j, k, n )
            array = np.genfromtxt( c_rg_iz_csv, delimiter=',' )
            if len( array.shape ) == 1:
                array_ = np.array( [ None if np.isnan(a) else a for a in array ] )
            elif len( array.shape ) == 2:
                # array_ = np.array( array )
                array_= array
                for p in range( array.shape[0] ):
                    for q in range( array.shape[1] ):
                        if np.isnan( array[p,q] ):
                            array_[p,q] = None
            else:
                assert( False )
            # print( (k, n), 'array_=', array_ )
            list_.append( array_ )
        c_rg_iz.append( list_ )

    qvector_csv = folder + '/qvector-%d-%d.csv' % ( m, j )
    qvector, _ = np_loadtxt( qvector_csv )

    data_matrix_csv = folder + '/data_matrix-%d-%d.csv' % ( m, j )
    if os.path.exists( data_matrix_csv ):
        data_matrix, _ = np_loadtxt( data_matrix_csv )
    else:
        data_matrix = None

    ze_array_csv = folder + '/ze_array-%d-%d.csv' % ( m, j )
    if os.path.exists( ze_array_csv ):
        ze_array, _ = np_loadtxt( ze_array_csv )
    else:
        ze_array = None

    param_array_csv = folder + '/param_array-%d-%d.csv' % ( m, j )
    # param_array = np_loadtxt( param_array_csv )   # results in an error
    param_array = np.loadtxt( param_array_csv, delimiter=',' )
    print( 'param_array.shape=', param_array.shape )

    orig_param_array_csv = folder + '/orig_param_array-%d-%d.csv' % ( m, j )
    if os.path.exists( orig_param_array_csv ):
        orig_param_array, _ = np_loadtxt( orig_param_array_csv )
    else:
        orig_param_array = None

    exz_array_csv = folder + '/exz_array-%d-%d.csv' % ( m, j )
    if os.path.exists( exz_array_csv ):
        exz_array, _ = np_loadtxt( exz_array_csv )
    else:
        exz_array = None

    return indeces, c_rg_iz, qvector, data_matrix, ze_array, param_array, orig_param_array, exz_array

def make_zero_extrapolation_overlay_book(
            excel_is_available, wb, ws,
            book_file, mn, qvector, param_array_list, boundary_indeces,
            parent=None, lrf_info=None, logger=None ):

    overlay = ZeroExtrapolationOverlayBook( wb, ws, mn, qvector, param_array_list, parent=parent, lrf_info=lrf_info )
    if excel_is_available:
        overlay.save( book_file )
        overlay.add_format_setting( book_file, boundary_indeces=boundary_indeces )
    else:
        if logger is not None:
            _, file = os.path.split(book_file)
            logger.warning( 'excel book formatting has been skipped for %s.', file )

def make_summary_book(excel_is_available, wb, book_file, controller, debug=False):
    if debug:
        from OnTheFly.DebugDialog import DebugDialog
        dialog = DebugDialog(debug_info=[book_file, controller])
        dialog.show()

    summary = SummaryBook( wb, controller )

    if excel_is_available:
        summary.save( book_file )
        summary.add_format_setting( book_file )

    path = book_file.replace( '.xlsx', '.csv' )
    print( 'path=', path )
    summary.save_as_csv( path )
