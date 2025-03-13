"""
    PeakInfo.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF
"""

class PeakInfo:
    def __init__(self, peak_id, top_x, elm_recs=None):
        self.peak_id    = peak_id
        self.top_x      = top_x

        self.elm_recs   = elm_recs
        """
        self.elm_recs will be
            either
                None
                    when constructed from old-style range
                    in AnalysisRangeInfo.upgrade_ranges
            or
                not None
                    when constructed
                    in DecompUtils.make_range_info_impl
        """

    def __repr__(self):
        elm_recs_ = None if self.elm_recs is None else ''.join([ str(e) for e in self.elm_recs ])
        return "%s(%d,%g,%s)" % (self.__class__.__qualname__, self.peak_id, self.top_x, elm_recs_)