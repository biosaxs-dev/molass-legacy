"""
    SimplestMapper.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""

class SimplestMapper:
    def __init__(self, curve1, curve2):
        self.scale = len(curve2.x)/len(curve1.x)
        self.origin1 = curve1.primary_peak_x
        self.origin2 = curve2.primary_peak_x

    def map_1to2(self, i):
        return int(round(self.origin2 + self.scale * (i - self.origin1)))
