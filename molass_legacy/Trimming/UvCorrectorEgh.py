"""
    UvCorrectorEgh.py

    Copyright (c) 2021-2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
from molass_legacy.Peaks.EghSupples import d_egh
from .UvCorrector import UvCorrector

class UvCorrectorEgh(UvCorrector):
    def __init__(self, curve1, curve2):
        self.curve1 = curve1
        self.curve2 = curve2
        self.x = curve2.x
        self.y = curve2.y

    def model_curve(self):
        pass

    def correction_curve(self, x, *params):
        slope, intercept, H, tR, sigma, tau = params
        return slope * x + intercept + d_egh(x, H, tR, sigma, tau)

    def fit(self, peak_region, debug=False, fig_file=None):
        x = self.x
        y = self.y

        fit_slice = self.get_widest_possible_fit_slice(peak_region)

        fx = x[fit_slice]
        fy = y[fit_slice]
        slope, intercept = linregress(fx, fy)[0:2]
        curve1 = self.curve1
        emg_peaks = curve1.get_emg_peaks()
        n = curve1.primary_peak_no
        emg_peak = emg_peaks[n]
        opt_params = emg_peak.opt_params
        H = opt_params[0]
        tR = opt_params[1]
        sigma = opt_params[2]
        tau = opt_params[3]
        pos_p0 = np.array([slope, intercept, +H, tR, sigma, tau])
        neg_p0 = np.array([slope, intercept, -H, tR, sigma, tau])

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
