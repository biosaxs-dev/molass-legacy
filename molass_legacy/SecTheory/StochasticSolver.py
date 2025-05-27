# coding: utf-8
"""
    SecTheory.StochasticSolver.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt
from .SecPDF import compute_standard_wCD

class CfDomain:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.area = np.sum(y)
        y_ = y/self.area
        self.N = N = 1024
        w, C, D = compute_standard_wCD(N)
        self.w = w
        self.C = C
        self.D = D
        cft = []
        for w_ in w:
            cft.append(np.sum(np.exp(1j*w_*x)*y_))
        self.cft = np.array(cft)

    def fit_impl(self, cf, init_params):
        self.cf = cf

        def obj_func(p):
            return np.sum(np.abs(cf(self.w, *p) - self.cft))

        opt_result = basinhopping(obj_func, init_params)
        print("opt params=", opt_result.x)
        return opt_result

    def get_spline(self, opt_params):
        N = self.N
        cft = self.cf(self.w[N//2:], *opt_params)
        cft = np.concatenate([cft[::-1].conj(), cft])
        pdfFFT = np.max([np.zeros(N), (self.C*np.fft.fft(self.D*cft)).real], axis=0)
        spline = UnivariateSpline(np.arange(N), pdfFFT*self.area, s=0)
        return spline

    def fit(self, cf, init_params):
        opt_result = self.fit_impl(cf, init_params)
        return self.get_spline(opt_result.x)
