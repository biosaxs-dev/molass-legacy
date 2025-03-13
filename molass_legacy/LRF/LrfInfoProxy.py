"""
    LrfInfoProxy.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier

class LrfInfoProxy:
    def __init__(self, A_data, need_bq=False, B_ratio=None, data=None, bq_bounds=None):
        self.boundary_j = None
        self.sg = sg = SimpleGuinier(A_data)
        self.need_bq_ = need_bq
        self.B_ratio = B_ratio
        self.Rg = sg.Rg
        self.basic_quality = sg.basic_quality
        self.data = data
        self.bq_bounds = bq_bounds

    def need_bq(self):
        return self.need_bq_
