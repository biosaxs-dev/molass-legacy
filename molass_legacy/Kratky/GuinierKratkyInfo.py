"""
    Kratky.GuinierKratkyInfo.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from scipy.stats import linregress
from molass_legacy.SerialAnalyzer.AnalyzerUtil import compute_conc_factor_util

class GuinierKratkyInfo:
    def __init__(self, optimizer, params, lrf_info, need_baseline=False):
        self.conc_factor = compute_conc_factor_util()   # consider a better place to do this

        separate_params = optimizer.split_params_simple(params)

        Pxr, Cxr, Puv, Cuv, mapped_UvD = lrf_info.matrices

        rg_params = lrf_info.get_valid_rgs(separate_params[2])
        self.rg_params = rg_params
        rg_ = np.average(rg_params)

        qv = optimizer.qvector

        q = 1.3/rg_
        i = bisect_right(qv, q*2)
        self.gslice = gslice = slice(0, i)
        self.qv2 = qv**2
        self.qv2_ = qv2_ = self.qv2[gslice]

        nc = len(rg_params)
        rgs = []
        I0s = []
        glny_s = []
        qrgs = []
        qrgnys = []
        nc_ = nc
        if need_baseline:
            nc_ += 1

        for k, cy in enumerate(Pxr.T[0:nc_]):
            glny_ = np.log(cy[gslice])
            slope, intercept = linregress(qv2_, glny_)[0:2]
            Rg = np.sqrt(-3*slope)
            rgs.append(Rg)
            I0 = np.exp(intercept)
            I0s.append(I0)
            qrg = qv*Rg
            qrgny = qrg**2*cy/I0
            glny_s.append(glny_ - intercept)
            qrgs.append(qrg)
            qrgnys.append(qrgny)

        self.rgs = rgs
        self.I0s = I0s
        self.Cuv = Cuv
        self.glny_s = glny_s
        self.qrgs = qrgs
        self.qrgnys = qrgnys

    def compute_adjacent_deviation_ratios(self):
        rgs = np.array(self.rgs)
        guinier_devs = np.abs(rgs[:-1] - rgs[1:])
        guinier_devs = guinier_devs/np.sum(guinier_devs)

        qrgnys = self.qrgnys
        kratky_devs = []
        for k, (y1, y2) in enumerate(zip(qrgnys[:-1], qrgnys[1:])):
            kratky_devs.append(np.sqrt(np.mean((y1 - y2)**2)))
        kratky_devs = np.array(kratky_devs)/np.sum(kratky_devs)

        devs = np.sqrt(guinier_devs**2 + kratky_devs**2)

        return devs/np.sum(devs)

    def get_molecular_masses(self):
        masses = []
        for k, i0 in enumerate(self.I0s):
            c = np.max(self.Cuv[k,:]) * self.conc_factor   # conc_factor ok?
            masses.append(i0/c)
        return masses
