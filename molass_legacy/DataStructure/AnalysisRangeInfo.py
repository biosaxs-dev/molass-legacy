"""

    AnalysisRangeInfo.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF

    This class is intended to unify the following existing range_info's and the newly
    introduced decomp_editor's range_info.

        from the attribute range_info
            AnalyzerDialog.range_info[0]
            AnalyzerDialog.set_analysis_ranges()
                AnalyzerUtil.set_analysis_ranges()

        from get_setting( ... )
            Analyzer.get_analysis_ranges()
                AnalyzerUtil.get_analysis_ranges()

    How to unify

        1)  define 'analysis_range_info' as a temporary setting info.
            this ensures that it is cleared everytime when the in_folder gets changed.

        2)  a range info is a list of range record, which has either one of
            the following styles

            [ PeakInfo, [f1, t1], [f2, t2] ]  (new style)
            [ PeakInfo, [f1, t1] ]            (new style)

            where old-style info has been convereted as follows

            [ lower, middle, upper ]  => [ PeakInfo(k, middle), [lower, middle], [middle, upper] ] 
            [ lower, upper, upper ]   => [ PeakInfo(k, upper), [lower, upper] ] for a microfluidic range

"""
import copy
import numpy as np
from molass_legacy.DataStructure.PeakInfo import PeakInfo
from molass_legacy._MOLASS.SerialSettings import set_setting

class PairedRange:
    def __init__(self, peak_info, range1=None, range2=None):
        """
        normally, range1 is not None.
        however, range1 can be None in abnormal cases as 20161104/BL-10C/OA
        """
        self.peak_id    = peak_info.peak_id
        self.top_x      = peak_info.top_x
        self.elm_recs   = peak_info.elm_recs
        self.fromto_list = []
        for r in [range1, range2]:
            if r is not None:
                self.fromto_list.append(r)

    def get_fromto_list(self):
        return self.fromto_list

    def as_list(self, k):
        return [PeakInfo(k, self.top_x, self.elm_recs)] + self.fromto_list

    def __repr__(self):
        return 'PairedRange(%s,%s)' % ("PeakInfo(%d,%g,'elm_recs')" % (self.peak_id,self.top_x),
                                       ",".join([str(rec) for rec in self.fromto_list]))

    def __getitem__(self, index):
        return self.fromto_list[index]

    def get_log_str(self):
        if len(self.fromto_list) == 1:
            return 'range ' + str(self.fromto_list[0])
        else:
            return 'range pair ' + str(self.fromto_list)

    def update_range(self, ad, f, t):
        r = self.fromto_list[ad]
        r[0] = f
        r[1] = t

    def get_concatenated_range(self):
        return self.fromto_list[0][0], self.fromto_list[-1][-1]

def upgrade_ranges(ranges):
    ret_ranges =[]
    for k, range_ in enumerate(ranges):
        if range_.__class__.__name__ == "PairedRange":  # type(range_) seems to require a qualifier, i.e., AnalysisRangeInfo.PairedRange
            range_ = range_.as_list(k)
        if len(range_) < 2:
            # currently, this case happens only from make_default_analysis_range_info
            # with select_fix
            continue
        if np.isscalar(range_[1]):
            lower, middle, upper = range_
            info_ = PeakInfo( k, middle )
            if middle == upper:
                # i.e., for a microfluidic range
                range_ = [ info_, [lower, upper] ]
            else:
                range_ = [ info_, [lower, middle], [middle, upper] ]
            ret_ranges.append(range_)
        else:
            ret_ranges.append(range_)
    return ret_ranges

class AnalysisRangeInfo:
    def __init__(self, ranges, editor=None):
        self.ranges = upgrade_ranges(ranges)
        self.editor = editor

    def get_ranges(self):
        return self.ranges

    def get_old_style_ranges(self):
        ranges = []
        for range_ in self.ranges:
            n = len(range_)
            if n == 3:
                left_rec = range_[1]
                right_rec = range_[2]
                ranges.append( left_rec + [right_rec[1]] )
            elif n == 2:
                lower, upper = range_[1]
                middle = (lower + upper)//2
                ranges.append( [lower, middle, upper] )
            else:
                assert False
        return ranges

    def __repr__(self):
        return str(self.ranges)

def report_ranges_from_analysis_ranges(j0, analysis_ranges):
    return [ [ [j0+f, j0+t] for f, t in prange.get_fromto_list()] for prange in analysis_ranges ]

def convert_to_paired_ranges(in_ranges):
    # print( 'convert_to_paired_ranges: in_ranges=', in_ranges )

    ranges = upgrade_ranges(in_ranges)

    paired_ranges = []
    num_ranges = 0

    for k, range_ in enumerate(ranges):
        if len(range_) == 2:
            prange_ = PairedRange(range_[0], range_[1])
        else:
            prange_ = PairedRange(range_[0], range_[1], range_[2])
        paired_ranges.append( prange_ )
        num_ranges += len( prange_.fromto_list )

    return paired_ranges, num_ranges

def shift_paired_ranges(j0, paired_ranges):
    ret_ranges = []
    for r in paired_ranges:
        new_r = copy.deepcopy(r)
        new_r.fromto_list = [ [ ft-j0  for ft in rec] for rec in r.fromto_list]
        ret_ranges.append(new_r)
    return ret_ranges

def shift_range_from_to_by_x(x, f, t):
    return [min(max(0, int(v - x[0])), len(x)-1) for v in [f, t]]

def get_analysis_info_from_curve(curve, editor):
    paired_ranges = curve.get_default_paired_ranges()
    return AnalysisRangeInfo(paired_ranges, editor)

def set_default_analysis_range_info(x_curve):
    from RangeInfo import RangeEditorInfo
    paired_ranges = x_curve.get_default_paired_ranges()
    set_setting( 'range_type', 4 )
    analysis_range_info = AnalysisRangeInfo(paired_ranges, editor='RangeEditor')
    set_setting( 'analysis_range_info', analysis_range_info )
    ignorable_flags = [0] * len(paired_ranges)
    set_setting( 'range_editor_info', RangeEditorInfo(paired_ranges, ignorable_flags))
