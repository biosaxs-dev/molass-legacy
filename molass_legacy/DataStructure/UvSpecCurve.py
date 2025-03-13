"""
    UvSpecCurve.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""


class UvSpecCurve:
    def __init__(self, sd, mpeaks=None):
        D, E, wv, ecurve = sd.get_uv_data_separate_ly()
        self.x = wv
        if mpeaks is None:
            j = ecurve.primary_peak_i
        else:
            xr_ecurve = sd.get_xray_curve()
            j = self.get_matching_peak_j(ecurve, xr_ecurve, mpeaks)
        self.y = D[:,j]

    def get_matching_peak_j(self, ecurve, xr_ecurve, mpeaks):
        j = ecurve.primary_peak_i
        print("ii=", mpeaks.ii)
        print("jj=", mpeaks.jj)
        return j

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y
