"""
    RangeInfo.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF

    Eiher of these infos can be converted to an AnalysisRangeInfo

                                SerialSetting temporaries
                                -----------------------
        RangeEditorInfo    <=>   range_editor_info
        DecompEditorInfo   <=>   decomp_editor_info

             ↓

        AnalysisRangeInfo  <=>   analysis_range_info
"""
import copy

class SuperEditorInfo:
    def get_ranges(self):
        ranges = []
        for prange in self.paired_ranges:
            try:
                range_list = prange.get_fromto_list()
            except:
                range_list = prange[1:]
            ranges.append(range_list)
        return ranges

    def get_ignorable_flags(self):
        return self.ignorable_flags

    def __repr__(self):
        return "%s(%s,%s)" % (self.__class__.__qualname__,
                              self.paired_ranges,
                              repr(self.ignorable_flags),   # this will stringify, e.g., as 'array([False, ..., ])'
                              )

class RangeEditorInfo(SuperEditorInfo):
    def __init__(self, paired_ranges, ignorable_flags):
        self.paired_ranges = copy.deepcopy(paired_ranges)
        self.ignorable_flags = copy.deepcopy(ignorable_flags)

    def update(self, default_ranges):
        pass

class DecompEditorInfo(SuperEditorInfo):
    def __init__(self, paired_ranges, ignorable_flags):
        self.paired_ranges = copy.deepcopy(paired_ranges)
        self.ignorable_flags = copy.deepcopy(ignorable_flags)

    def update(self, control_info):
        pass

def shift_editor_ranges(j0, editor_ranges):
    ret_range_list = []
    for prange in editor_ranges:
        prange_ = []
        for f, t in prange:
            prange_.append([j0+f, j0+t])
        ret_range_list.append(prange_)

    return ret_range_list
