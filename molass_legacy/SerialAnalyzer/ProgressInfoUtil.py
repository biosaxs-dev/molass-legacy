"""
    ProgressInfoUtil.py

    Copyright (c) 2017-2024, Masatsuyo Takahashi, KEK-PF
"""

"""
    stream      phase       max
    Base Corr.  0           100

    Guinier     0   autorg  num_files
                1   excel   openpyxl(), com()

    Zero-Ex     0   asc(0)  
                1   desc(0) 
                2   asc(1)  
                3   desc(1) 
                .
                1000  merge   com()
"""
from molass_legacy.SerialAnalyzer.AnalyzerUtil import get_analysis_ranges
from molass_legacy.KekLib.ProgressInfo import put_info
from ZeroExtrapolator import NUM_EXTRAPOLATION_POINTS
from molass_legacy.DataStructure.AnalysisRangeInfo import convert_to_paired_ranges

NUM_SHEET_TYPES = 3     # ( ASC,  DATGNOM-ASC ), ( DESC , DATGNOM-DESC ), OVERLAY,
STREAM_BASECOR  = 0
STREAM_GUINIER  = 1
STREAM_ZERO_EX  = 2
STREAM_BASECOR_MAX  = 10

def estimate_gunier( num_files, max_dict={} ):
    max_dict[ (STREAM_GUINIER, 0) ]  = num_files
    max_dict[ (STREAM_GUINIER, 1) ]  = 5
    return max_dict

def extimate_zero_extrapolation( ranges, max_dict={} ):
    num_zx_points = NUM_EXTRAPOLATION_POINTS

    for i, range_ in enumerate( ranges ):
        fromto_list = range_.get_fromto_list()
        lower, middle1 = fromto_list[0]
        if len(fromto_list) > 1:
            middle2, upper = fromto_list[1]
        else:
            middle2 = None
        max_dict[ (STREAM_ZERO_EX, i*NUM_SHEET_TYPES   ) ] = 1 + ( middle1 - lower ) + 1 + num_zx_points + 1 + 1
        if middle2 is not None:
            max_dict[ (STREAM_ZERO_EX, i*NUM_SHEET_TYPES+1 ) ] = 1 + ( upper - middle2 ) + num_zx_points + 1 + 1
        max_dict[ (STREAM_ZERO_EX, i*NUM_SHEET_TYPES+2 ) ] = 1

        # TODO: is this asymmetry ok?

    num_sheets      = len( ranges )*NUM_SHEET_TYPES
    max_dict[ (STREAM_ZERO_EX, 1000) ] = num_sheets
    max_dict[ (STREAM_ZERO_EX, 2000) ] = 1
    # print( 'max_dict=', max_dict )
    return max_dict

def estimate_init_max_dist( num_files, scattering_correction, ranges, zx_flag ):
    print( 'estimate_init_max_dist: scattering_correction=', scattering_correction )

    num_zx_points = NUM_EXTRAPOLATION_POINTS

    stream_basecorr_max = 0 if scattering_correction == 0 else STREAM_BASECOR_MAX
    max_dict = { (STREAM_BASECOR, 0):stream_basecorr_max }

    guinier_max_dict = estimate_gunier( num_files, max_dict=max_dict )

    if ranges is None:
        ranges = get_analysis_ranges()

    if zx_flag:
        applied_ranges, _ = convert_to_paired_ranges( ranges )
        init_max_dict = extimate_zero_extrapolation( applied_ranges, max_dict=guinier_max_dict )
    else:
        init_max_dict = guinier_max_dict

    # print( 'init_max_dict=', init_max_dict )
    return init_max_dict

class ProgressCallback:
    def __init__(self, max_progress, debug=True):
        self.debug = debug
        if debug:
            import logging
            self.logger = logging.getLogger('ProgressCallback')
        self.max_progress   = max_progress

    def __call__( self, i ):
        if i % 100 == 0 or i == self.max_progress:
            if i == self.max_progress:
                progress    = STREAM_BASECOR_MAX
            else:
                # this calculation is not accurate
                progress    = int( STREAM_BASECOR_MAX * i / self.max_progress )
            if self.debug:
                self.logger.info("put_info( (%d, 0), %d )", STREAM_BASECOR, progress)
            put_info( (STREAM_BASECOR, 0), progress )
