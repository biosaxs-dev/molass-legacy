# coding: utf-8
"""

    ファイル名：   AnalyzerUtil.py

    処理内容：

        連続測定データ全体解析起動 GUI の範囲設定

    Copyright (c) 2016-2020, SAXS Team, KEK-PF

"""
import os
from molass_legacy.KekLib.OurTkinter             import Tk
from molass_legacy._MOLASS.SerialSettings         import get_setting
from molass_legacy.KekLib.BasicUtils             import mkdirs_with_retry, get_caller_module
from molass_legacy._MOLASS.Version                import get_version_string
from molass_legacy._MOLASS.SerialSettings         import get_setting, set_setting
from molass_legacy.DataStructure.AnalysisRangeInfo      import AnalysisRangeInfo

def make_guinier_result_stamp( self ):

    input_smoothing         = get_setting( 'input_smoothing' )
    if input_smoothing == 1:
        params = 'Averaging ' + str( get_setting( 'num_curves_averaged' ) ) + 'curves\n'
        smoothing_description  = 'Smoothing Method on Elution Axis,' + params
    else:
        smoothing_description  = 'No Smoothing on Elution Axis\n'

    stamp = (   'Serial Analyzer Version,' + get_version_string() + '\n'
                + 'Intensity Data Folder,' + self.data_folder + '\n'
                + 'Absorbance Data Folder,' + self.conc_folder + '\n'
                + smoothing_description
                )
    return stamp

def exists_guinier_analysis_result( self, create=True ):
    if self.guinier_folder is None: return False
    if self.stamp_file is None: return False

    if not os.path.exists( self.guinier_folder ):
        if create:
            mkdirs_with_retry( self.guinier_folder )
        return False

    if not os.path.exists( self.stamp_file ):
        return False

    try:
        fh = open( self.stamp_file )
        stamp = fh.read()
        fh.close()
    except:
        stamp = ''

    return stamp == make_guinier_result_stamp( self )

def compute_conc_factor_util():
    return get_setting( 'path_length' ) / get_setting( 'extinction' )

def get_init_ranges( just_ranges=None ):
    ranges = []

    analysis_range_info = get_setting( 'analysis_range_info' )
    if analysis_range_info is None:
        manual_range_info = get_setting( 'manual_range_info' )
        if manual_range_info is not None:
            just_ranges = manual_range_info
        for range_ in just_ranges:
            # print('get_init_ranges(1): range_=', range_)
            ranges.append( [ 1, tuple( range_ ) ] )
    else:
        for range_ in analysis_range_info.get_ranges():
            # print('get_init_ranges(2): range_=', range_)
            top_x = int(range_[0].top_x + 0.5)
            ranges.append( [ 1, (range_[1][0], top_x, range_[-1][1]) ] )

    return ranges

def set_analysis_ranges( ranges ):
    set_setting( 'analysis_range_info', AnalysisRangeInfo(ranges) )

def get_analysis_ranges():
    analysis_range_info = get_setting( 'analysis_range_info' )
    assert analysis_range_info is not None
    return analysis_range_info.get_ranges()
