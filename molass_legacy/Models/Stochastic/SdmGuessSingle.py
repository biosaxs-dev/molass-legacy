"""
    Models.RateTheory.EdmBasicStudy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from molass_legacy.Models.RateTheory.RobustEDM_Utils import compute_moments
from molass_legacy.Models.Stochastic.DispersivePdf import dispersive_monopore_pdf
from .DispersivePdf import N0

def guess_sdm_params(x, y, timescale=0.25):
    M = compute_moments(x, y)
    print("M=", M)

    t0_init = 50
    rho = 0.5
    me = 1.5
    mp = 1.5

    def moment_objective(p):
        K, N, t0  = p
        T = K/N
        n1 = N*(1 - rho)**me
        t1 = T*(1 - rho)**mp
        M1_ = t0 + n1*t1
        M2_ = 2*n1*t1**2 + M1_**2/N0
        M3_ = 6*n1*t1**2*(N0*t1 + n1*t1 + t0)/N0
        return np.log((M1_ - M[1])**2) + np.log((M2_ - M[2])**2) + np.log((M3_ - M[3])**2)
    
    res = minimize(moment_objective, [500, 1000, t0_init], method='Nelder-Mead')

    K_init, N_init, t0_init2 = res.x

    def scale_objective(p):
        K, N, t0, scale  = p
        T = K/N
        n1 = N*(1 - rho)**me
        t1 = T*(1 - rho)**mp
        y_ = scale*dispersive_monopore_pdf(x, n1, t1, N0, t0, timescale=timescale)
        return np.sum((y_ - y)**2)

    res = minimize(scale_objective, [K_init, N_init, t0_init2, M[0]], method='Nelder-Mead')
    K, N, t0, scale = res.x

    return np.array([K, N, t0, N0, rho, me, mp, scale])