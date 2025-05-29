"""
    Models.Stochastic.LognormalPoreFunc.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.stats import lognorm
from scipy.integrate import quad_vec
from molass_legacy.SecTheory.SecPDF import FftInvPdf
from molass_legacy.KekLib.IntegrateUtils import complex_quadrature_vec
from molass_legacy.Models.Stochastic.LognormalUtils import compute_mode, compute_stdev
from molass_legacy.Models.Stochastic.ParamLimits import PORESIZE_INTEG_LIMIT

def Ksec(Rg, r, m):
    return np.power(1 - min(1, Rg/r), m)

def distr_func(r, mu, sigma):
    return lognorm.pdf(r, sigma, scale=np.exp(mu))

def integrand_impl(r, w, N, T, me, mp, mu, sigma, Rg):
    return distr_func(r, mu, sigma)*N*Ksec(Rg, r, me)*(1/(1 - w*1j*T*Ksec(Rg, r, mp)) - 1)

def lognormal_pore_cf(w, N, T, me, mp, mu, sigma, Rg, x0, const_rg_limit=False):
    if const_rg_limit:
        max_rg = PORESIZE_INTEG_LIMIT
    else:
        mode = compute_mode(mu, sigma)
        stdev = compute_stdev(mu, sigma)
        max_rg = min(PORESIZE_INTEG_LIMIT, mode + 5*stdev)

    # note that integrand_impl is a vector function because w is a vector
    integrated = complex_quadrature_vec(lambda r: integrand_impl(r, w, N, T, me, mp, mu, sigma, Rg), Rg, max_rg)[0]
    return np.exp(integrated + 1j*w*x0)     # + 1j*w*x0 may not be correct. reconsider

lognormal_pore_pdf = FftInvPdf(lognormal_pore_cf)

def lognormal_pore_func(x, scale, N, T, me, mp, mu, sigma, Rg, x0):
    return scale*lognormal_pore_pdf(x - x0, N, T, me, mp, mu, sigma, Rg, 0)  # not always the same as below
    # return scale*lognormal_pore_pdf(x, N, T, me, mp, mu, sigma, Rg, x0)

def compute_trvec(N, T, t0, me, mp, mu, sigma, rgv):
    from molass_legacy.Models.Stochastic.LognormalUtils import compute_mode, compute_stdev
    mode = compute_mode(mu, sigma)
    stdev = compute_stdev(mu, sigma)
    rmin = max(0, mode - 3*stdev)
    rmax = min(PORESIZE_INTEG_LIMIT, mode + 3*stdev)
    ones = np.ones(len(rgv))
    """
    including t0 in lambda seems to make a latent bug appear in scipy 1.11.4 or 1.12.0 in in Python 3.11.8
    excluding t0 from lambda is better, anyway
    """
    return t0 + quad_vec(lambda r: N*T*distr_func(r,mu,sigma)*np.power(1 - np.min([ones, rgv/r], axis=0), me+mp), rmin, rmax)[0]

if False:
    rv_list = []
    def distr_func_debug(r, mu, sigma):
        v = lognorm.pdf(r, sigma, scale=np.exp(mu))
        rv_list.append((r,v))
        return v

    def plot_rv_list():
        import molass_legacy.KekLib.DebugPlot as plt
        rv_array = np.array(rv_list)
        print("rv_array.shape=", rv_array.shape)
        print("r=", rv_array[:,0])
        print("v=", rv_array[:,1])
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("compute_trvec debug")
            ax.plot(*rv_array.T)
            fig.tight_layout()
            plt.show()