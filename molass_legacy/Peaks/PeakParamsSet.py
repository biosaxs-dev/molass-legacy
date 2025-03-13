"""
    Peaks/PeakParamsSet.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np

class PeakParamsSet:
    def __init__(self, uv_peaks, xr_peaks, a, b):
        self.uv_peaks = np.asarray(uv_peaks)
        self.xr_peaks = np.asarray(xr_peaks)
        self.a = a
        self.b = b
        self.items = [self.uv_peaks, self.xr_peaks, a, b]

    def __iter__(self):
        # this is for backward compatibility
        return iter(self.items)

    def __getitem__(self, item):
        # this is for backward compatibility
        return self.items[item]
