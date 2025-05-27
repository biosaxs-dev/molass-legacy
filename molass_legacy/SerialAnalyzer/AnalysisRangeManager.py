# coding: utf-8
"""
    AnalysisRangeManager.py

    Copyright (c) 2016-2020, SAXS Team, KEK-PF
"""
import numpy                as np
from molass_legacy.SerialAnalyzer.ElutionCurve           import ElutionCurve
from molass_legacy.DataStructure.AnalysisRangeInfo      import convert_to_paired_ranges
from PairedRangeLogger      import log_paired_ranges

DEBUG   = False

def get_analysis_ranges_from_sd( self, range_type, min_value, quality_array=None, conc_y=None, rg_array=None, ic_array=None ):
    # print( 'get_analysis_ranges_from_sd: min_quality=', min_quality )

    if conc_y is None:
        conc_y = self.conc_curve.y

    if range_type == 0:
        # ref_curve specification excludes invalid subobjects
        # which have inappropriate ref_curve values.
        curve = ElutionCurve( quality_array, low_quality=True )
    elif range_type == 1:
        curve = ElutionCurve( conc_y )
    else:
        assert( False )

    # print( 'curve.subobject_array=', curve.subobject_array )
    ranges = []
    for i, subobj in enumerate( curve.subobject_array ):
        ft, obj = subobj
        # print( 'subobj[%d]' % i, ft, obj.lower_abs, obj.middle_abs, obj.upper_abs )
        ranges.append( [ obj.lower_abs, obj.middle_abs, obj.upper_abs ] )

    return ranges

def get_analysis_ranges_for_exec( self, range_type, analysis_ranges, logger ):
    assert analysis_ranges is not None

    ranges = analysis_ranges
    ret = convert_to_paired_ranges( ranges )
    paired_ranges = ret[0]
    log_paired_ranges(self.logger, paired_ranges)

    return paired_ranges
