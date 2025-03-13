"""
    Selective.PeakProxy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

class PeakProxy:
    """
    This is a proxy class for EmgPeak class.
    the instance should behave consistently in FitRecord.sort_fit_recs,
    so that the order of the fit_recs is the same as the order of the peaks.
    task: to unify the code of PeakProxy and EmgPeakProxy.
    """
    def __init__(self, top_x=None, top_y=None, area_prop=None, sign=1):
        self.top_x = top_x
        self.top_y = top_y
        self.area_prop = area_prop
        self.sign = sign