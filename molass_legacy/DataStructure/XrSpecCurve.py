"""
    XrSpecCurve.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""


class XrSpecCurve:
    def __init__(self, sd):
        D, E, qv, ecurve = sd.get_xr_data_separate_ly()
        self.x = qv
        j = ecurve.primary_peak_i
        self.y = D[:,j]

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y
