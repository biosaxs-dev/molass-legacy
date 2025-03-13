# coding: utf-8
"""
    UnfoldingModel.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import numpy as np
from lmfit import Parameters, minimize

RT = 1

class UnfoldingModel:
    def __init__(self, params=None):
        if params is None:
            params = Parameters()
            params.add('G', value=4)
            params.add('m', value=1)
            params.add('af', value=0.5)
            params.add('bf', value=-0.01)
            params.add('au', value=0.3)
            params.add('bu', value=-0.02)
        self.parems = params

    def compute_Pf(self, params, x):
        if params is None:
            params = self.parems
        G = params['G']
        m =  params['m']
        Pf = 1/(1 + np.exp(-(G-m*x)/RT))
        return Pf

    def compute_yf(self, params, Pf, x):
        if params is None:
            params = self.parems

        af = params['af']
        bf = params['bf']
        yf = (af + bf*x)*Pf
        return yf

    def compute_yu(self, params, Pf, x):
        if params is None:
            params = self.parems

        au = params['au']
        bu = params['bu']
        yu = (au + bu*x)*(1-Pf)
        return yu

    def fit(self, x, y):
        params = Parameters()
        params.add('G', value=2)
        params.add('m', value=2)
        params.add('af', value=0)
        params.add('bf', value=0)
        params.add('au', value=0)
        params.add('bu', value=0)

        def obj_func(p):
            Pf = self.compute_Pf(p, x)
            yf = self.compute_yf(p, Pf, x)
            yu = self.compute_yu(p, Pf, x)
            return yf+yu - y

        res = minimize(obj_func, params, args=())
        return res
