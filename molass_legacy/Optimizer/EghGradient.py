# coding: utf-8
"""
    Optimizer.EghGradient.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.Models.ElutionCurveModels import egh

PARAMS = ['h', 'mu', 'sigma', 'tau']    # 'a' not used
NUM_PARAMS = len(PARAMS)

def compute_gradient_impl(M, P, rank, x, pv):
    print('compute_gradient_impl')
    pv_ = pv.reshape((rank, NUM_PARAMS))
    DP = np.zeros((NUM_PARAMS, len(x)))
    C_list = []
    for h, mu, sigma, tau in pv_:
        fv = egh(x, h, mu, sigma, tau)
        dh = egh(x, 1, mu, sigma, tau)
        x_mu = x-mu
        denom1 = 2*sigma**2 + tau*x_mu
        denom2 = denom1**2
        dm = fv * (-tau*x_mu**2/denom2 + 2*x_mu/denom1)
        ds = fv * 4*sigma*x_mu**2/denom2
        dt = fv * x_mu**3/denom2
        C_list.append(fv)
        DP += np.array([dh, dm, ds, dt])    # ok?
    C = np.array(C_list)
    DC = 2 * P.T @ (P@C - M)
    print('DC.shape=', DC.shape)
    print('DP.shape=', DP.shape)
    ret = (DC @ DP.T).flatten()
    print('ret.shape=', ret.shape)
    return ret
