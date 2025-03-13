# coding: utf-8
"""
    Peaks.Peak.py

    EmgPeak successor

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from .PeakUtils import recognize_peaks
from .ElutionModels import egh

class Peak:
    def __init__(self, params):
        self.params = params
        self.opt_params = {
                "h" : params[0],
                "mu" : params[1],
                "sigma" : params[2],
                "tau" : params[3],
                }

    def get_model_y(self, x):
        return egh(x, *self.params)

def get_peaks(curve, **kwargs):
    x   = curve.x
    y   = curve.y

    params_list = recognize_peaks(x, y, num_peaks=len(curve.peak_info), debug=True)

    peaks = []
    for params in params_list:
        peaks.append(Peak(params))

    return peaks
