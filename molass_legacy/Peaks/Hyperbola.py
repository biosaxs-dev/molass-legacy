# coding: utf-8
"""

    Hyperbola.py

    Copyright (c) 2021, SAXS Team, KEK-PF

"""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt

"""
    see rotated-hyperbola.ipynb
"""

def compute_rotated_hyperbola_y(x, ptx, pty, y0, a, b, theta):
    x_ = x - ptx
    y_ = -a/b * np.sqrt(x_**2 + b**2) + y0 - pty
    return x_*np.sin(theta) + y_*np.cos(theta) + pty        

HBW_SCALE = 30

class RotatedHyperbola:
    def __init__(self, ptx, pty, a=None, b=None, hbw=None,deg=0):
        assert (b is not None) or (hbw is not None)
        if a is None:
            a = pty/hbw*HBW_SCALE
        y0 = pty + a
        if b is None:
            b = hbw/np.sqrt((y0/a)**2 - 1)
        self.ptx = ptx
        self.pty = pty
        self.x0 = ptx
        self.y0 = y0
        self.a = a
        self.b = b
        theta = deg*np.pi/180
        self.init_params = (ptx, pty, y0, a, b, theta)
        self.fitted_params = None
        self.spline = None

    def fit(self, x, y, method=None):
        def obj_func(p):
            return np.sum((compute_rotated_hyperbola_y(x, *p) - y)**2)
        result = minimize(obj_func, self.init_params, method=method)
        self.fitted_params = result.x
        print("self.fitted_params=", self.fitted_params)

    def __call__(self, x):
        params = self.init_params if self.fitted_params is None else self.fitted_params
        return compute_rotated_hyperbola_y(x, *params)

    def get_peak_top(self):
        ptx, pty, _, a = self.fitted_params[0:4]
        hbw = pty/a*HBW_SCALE
        x_ = np.linspace(ptx-hbw, ptx+hbw, 100)
        y_ = compute_rotated_hyperbola_y(x_, *self.fitted_params)
        n = np.argmax(y_)
        return np.array([x_[n], y_[n]])
