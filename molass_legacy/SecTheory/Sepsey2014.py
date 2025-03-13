# coding: utf-8
"""
    SecTheory.Sepsey2014.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy import integrate

def rho(rg, rp):
    return 0 if rg > rp else rg/rp

def np_(nperm, me, rho_):
    return nperm*(1 - rho_)**me

def tp_(tperm, mp, rho_):
    return tperm*(1 - rho_)**mp

def lognormal_pore_cf_iterate(w, rp0, sigma, nperm, me, tperm, mp, rg):
    # 2014, A. Sepsey. Eq.(21)
    z = 0
    denom_sigma2 = 2*sigma**2
    log_rp0 = np.log(rp0)
    for r in range(10, 200):
        if r < rg:
            continue

        rho_ = rg/r
        np_ = nperm*(1 - rho_)**me
        tp_ = tperm*(1 - rho_)**mp
        z += np_/r * np.exp(- (np.log(r) - log_rp0)**2/denom_sigma2) * (1/(1 - 1j*w*tp_) - 1)
    z /= np.sqrt(2*np.pi)*sigma
    return np.exp(z)

def lognormal_pore_cf(w, rp0, sigma, nperm, me, tperm, mp, rg):
    # 2014, A. Sepsey. Eq.(21)
    denom_sigma2 = 2*sigma**2
    log_rp0 = np.log(rp0)
    def integrand(r):
        rho_ = rg/r
        np_ = nperm*(1 - rho_)**me
        tp_ = tperm*(1 - rho_)**mp
        return np_/r * np.exp(- (np.log(r) - log_rp0)**2/denom_sigma2) * (1/(1 - 1j*w*tp_) - 1)
    val, err = integrate.quad_vec(integrand, rg, 100)
    return np.exp(np.sqrt(2*np.pi)*sigma*val)
