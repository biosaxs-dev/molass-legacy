# coding: utf-8
"""

    ファイル名：   MappingAutoHelper.py

    処理内容：

        濃度マッピング不具合の自動修正

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF

"""
from molass_legacy._MOLASS.SerialSettings     import get_setting
# from molass_legacy.SerialAnalyzer.ElutionCurve       import PEAK_INFO_RAW

PEAK_MATCH_ALLOW_RATIO  = 0.05

def get_add_info( name_t, x_t, x_f, scale, allow_t ):
    ret_info    = []

    debug = True
    if debug:
        print( '---- get_add_info for', name_t, scale, allow_t )
        print( 'x_t=', x_t )
        print( 'x_f=', x_f )

    for pf in x_f:
        pf_ = pf * scale
        found = False
        for pt in x_t:
            if abs( pt - pf_ ) < allow_t:
                found = True
        if not found:
            ret_info.append( pf_ )

    if debug: print( 'ret_info=', ret_info )

    return ret_info

def make_helper_info( mapper, exception ):
    if get_setting( 'enable_auto_helper' ) == 0:
        return None

    exc_info = exception.args[1]
    if exc_info[0] > 2:
        # can't cope with this situation
        return None

    print( '------------------------- make_helper_info: ', exc_info )
    a_info, x_info  = mapper.get_original_curve_info()

    a_size  = len(mapper.a_vector)
    x_size  = len(mapper.x_vector)
    a_allow = a_size * PEAK_MATCH_ALLOW_RATIO
    x_allow = x_size * PEAK_MATCH_ALLOW_RATIO

    ratio   = a_size / x_size
    print( 'ratio=', ratio )

    A       = ratio
    A_inv   = 1/A

    a_peak_x        = [ info[1] for info in a_info.peak_info ]
    x_peak_x        = [ info[1] for info in x_info.peak_info ]
    a_peak_x_raw    = [ info[1] for info in a_info.raw_info[PEAK_INFO_RAW] ]
    x_peak_x_raw    = [ info[1] for info in x_info.raw_info[PEAK_INFO_RAW] ]

    a_add_info      = get_add_info( 'a_peak_x',     a_peak_x,       x_peak_x,       A, a_allow )
    a_add_info_raw  = get_add_info( 'a_peak_x_raw', a_peak_x_raw,   x_peak_x_raw,   A, a_allow )

    x_add_info      = get_add_info( 'x_peak_x',     x_peak_x,       a_peak_x,       A_inv, x_allow )
    x_add_info_raw  = get_add_info( 'x_peak_x_raw', x_peak_x_raw,   a_peak_x_raw,   A_inv, x_allow )

    anyway_flag = False
    helper_info = [ ( [], a_add_info, a_add_info_raw ), ( [], x_add_info, x_add_info_raw ), mapper.flow_changes, anyway_flag ]

    num_info    = len(a_add_info) + len(a_add_info_raw) + len(x_add_info) + len(x_add_info_raw)
    if num_info == 0:
        helper_info = None
    else:
        mapper.logger.info( 'succeeded in making helper info' )

    return helper_info
