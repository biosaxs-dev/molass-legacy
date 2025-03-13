# coding: utf-8
"""
    SimpleUnfolding.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import logging
import numpy as np
from scipy import optimize
from lmfit import Parameters, minimize

MAX_ITERATION_FMIN_CG = 10000

"""
constants borrowed from PyFolding.constants.py
"""
# constants
IDEAL_GAS_CONSTANT_KCAL = 1.987204118E-3
TEMPERATURE_CELSIUS = 25.

# calculated constants
ZERO_KELVIN = 273.15
TEMPERATURE_KELVIN = ZERO_KELVIN + TEMPERATURE_CELSIUS
RT = IDEAL_GAS_CONSTANT_KCAL * TEMPERATURE_KELVIN
# print('RT=', RT)    # 0.5924849

def proportion_folded(G, m, x):
    return 1/(1 + np.exp(-(G-m*x)/RT))

class SimpleUnfolding:
    def __init__(self, params=None):
        self.logger = logging.getLogger(__name__)
        if params is None:
            params = Parameters()
            params.add('G', value=4)
            params.add('m', value=1)
            params.add('af', value=0.5)
            params.add('bf', value=-0.01)
            params.add('au', value=0.3)
            params.add('bu', value=-0.02)
        self.params = params

    def compute_Pf(self, params, x):
        if params is None:
            params = self.params
        G = params['G']
        m = params['m']
        pf = proportion_folded(G, m, x)
        return pf

    def compute_yf(self, params, pf, x):
        if params is None:
            params = self.params

        af = params['af']
        bf = params['bf']
        yf = (af + bf*x)*pf
        return yf

    def compute_yu(self, params, pf, x):
        if params is None:
            params = self.params

        au = params['au']
        bu = params['bu']
        pu = (1-pf)
        yu = (au + bu*x)*pu
        return yu

    def fit(self, x, y, G_init=1):
        # use UnfoldingModel to get init pramas
        mx = (x[0] + x[-1])/2
        init_params = Parameters()
        init_params.add('G', value=G_init, min=0, max=100)
        m_init = G_init/mx
        init_params.add('m', value=m_init, min=0, max=100)
        init_params.add('af', value=y[0], min=0, max=10)
        init_params.add('bf', value=0, min=-1, max=1)
        init_params.add('au', value=y[-1], min=0, max=10)
        init_params.add('bu', value=0, min=-1, max=1)
        # print('init params=', [(k, p.value) for k, p in init_params.items()])

        def obj_func(p):
            pf = self.compute_Pf(p, x)
            yf = self.compute_yf(p, pf, x)
            yu = self.compute_yu(p, pf, x)
            return yf+yu - y

        min_error = None
        opt_method = None
        opt_result = None
        # for method in ['leastsq', 'least_squares', 'cg', 'tnc', 'dual_annealing']:
        for method in ['leastsq']:
            res = minimize(obj_func, init_params, args=(), method=method)
            error = self.compute_error(res.params)
            if min_error is None or error < min_error:
                min_error = error
                opt_method = method
                opt_result = res

        # print('opt_method=', opt_method)
        G = opt_result.params['G']
        m = opt_result.params['m']
        self.logger.info("initial parameters (G=%.3g, m=%.3g) have been optimized to (G=%.3g, m=%.3g).",
                            G_init, m_init, G, m)
        return opt_result

    def compute_error(self, params):
        error = 0
        for p0, p1 in zip(self.params.items(), params.items()):
            error += (p0[1].value - p1[1].value)**2
        return error

    def make_data(self, dependency, x):
        pf = self.compute_Pf(None, x)
        yf = self.compute_yf(None, pf, x)
        yu = self.compute_yu(None, pf, x)
        return [yf, yu]

    def make_conc_vector(self, x):
        yf, yu = self.make_data(1, x)
        return yf + yu
