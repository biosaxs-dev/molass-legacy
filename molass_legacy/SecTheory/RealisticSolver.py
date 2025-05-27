# coding: utf-8
"""
    SecTheory.RealisticSolver.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt
from .SecPDF import compute_standard_wCD

class RealisticCfDomain:
    def __init__(self, x, y, cf):
        self.x = x
        self.y = y
        self.cf = cf
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

    def fit(self, init_params):

        def obj_func(p):
            return self.cf.objective_func(self.w, p, self.cft)

        opt_result = basinhopping(obj_func, init_params)
        return opt_result

    def get_spline(self, opt_params):
        N = self.N
        cft = self.cf(self.w[N//2:], opt_params)
        cft = np.concatenate([cft[::-1].conj(), cft])
        pdfFFT = np.max([np.zeros(N), (self.C*np.fft.fft(self.D*cft)).real], axis=0)
        spline = UnivariateSpline(np.arange(N), pdfFFT*self.area, s=0)
        return spline

    def get_component_splines(self, opt_params):
        N = self.N
        half_w = self.w[N//2:]
        zeros = np.zeros(N)
        x = np.arange(N)
        model_cf = self.cf.model_cf

        rg_params, rp_params = self.cf.split_params(opt_params)
        ret_splines = []
        for rg, weight in rg_params:
            cft = model_cf(half_w, rg, *rp_params)
            cft = np.concatenate([cft[::-1].conj(), cft])
            pdfFFT = np.max([zeros, (self.C*np.fft.fft(self.D*cft)).real], axis=0)
            spline = UnivariateSpline(x, pdfFFT*weight*self.area, s=0)
            ret_splines.append(spline)
        return ret_splines
