# coding: utf-8
"""
    NormalVectorDiff.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""

import numpy                as np
from scipy.interpolate      import UnivariateSpline

VERY_SMALL_VALUE = 1e-6

class NormalVectorDiff:
    def __init__(self, x, y1, spline1=None):
        self.x  = x
        self.y1 = y1
        if spline1 is None:
            spline1 = UnivariateSpline(x, y1, s=0)
        d1 = spline1.derivative(1)(x)
        self.ratio = 1/np.sqrt(1 + d1**2)

    def diff(self, y2):
        dy = np.abs( y2 - self.y1 )
        return dy * self.ratio