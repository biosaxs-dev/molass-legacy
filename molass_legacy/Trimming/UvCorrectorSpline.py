"""
    UvCorrectorSpline.py

    Copyright (c) 2021-2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
from ScipyUtils import get_spline
from molass_legacy.Peaks.ElutionModels import egh
import molass_legacy.KekLib.DebugPlot as plt
from .UvCorrector import UvCorrector

class UvCorrectorSpline(UvCorrector):
    def __init__(self, curve1, curve2):
        self.curve1 = curve1
        self.curve2 = curve2
        self.x = curve2.x
        self.y = curve2.y
        self.d1 = curve1.d1

    def derivative_curve(self, x, scale):
        return scale * self.d1(x)

    def correction_curve(self, x, *params):
        slope, intercept, scale = params
        return slope * x + intercept + self.derivative_curve(x, scale)

    def fit(self, peak_region, debug=False, fig_file=None):
        x = self.x
        y = self.y

        fit_slice = self.get_widest_possible_fit_slice(peak_region)

        fx = x[fit_slice]
        fy = y[fit_slice]
        slope, intercept = linregress(fx, fy)[0:2]

        pos_p0 = np.array([slope, intercept, +1])
        neg_p0 = np.array([slope, intercept, -1])

        popts = []
        errors = []
        for k, p0 in enumerate([pos_p0, neg_p0]):
            if debug:
                from .UvCorrectorIllust import debug_plot
                debug_plot("Initial Parameters", self, x, y, p0, fit_slice, fig_file=fig_file)

            try:
                popt, pcov = curve_fit(self.correction_curve, fx, fy, p0)
                error = np.std(self.correction_curve(fx, *p0) - fy)
                if debug:
                    debug_plot("Optimized Parameters", self, x, y, popt, fit_slice, init_params=p0, fig_file=fig_file)
            except Exception as exc:
                print(exc)
                popt = None
                error = np.inf

            popts.append(popt)
            errors.append(error)

        print("errors=", errors)

        self.popt = popts[0] if errors[0] < errors[1] else popts[1]
