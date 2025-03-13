"""
    Models/Stochastic/SecModelUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.ElutionModelUtils import compute_4moments
from molass_legacy.Models.ModelUtils import compute_area_props
from Experiment.ColumnTypes import get_all_poresizes
from SecTheory.BasicModels import robust_single_pore_pdf
from molass_legacy.Models.Stochastic.LognormalPoreFunc import lognormal_pore_func

DEFAULT_SIGMA = 0.1
SIGMA_LOWER_BOUND = 0.05
NUM_ENTRIES_LIST = [1000, 2000, 3000, 4000, 5000]
DEFAULT_NUM_ENTRIES = NUM_ENTRIES_LIST[1]
DEFAULT_ENTRY_TIME = 0.5

def compute_tRv(N, T, x0, m, poresize, rgs):
    rho = rgs/poresize
    rho[rho > 1] = 1
    return x0 + N*T*(1 - rho)**m

def guess_monopore_params_from_rgdist(x, y, peak_rgs, peak_trs, props, logger=None, debug=True):
    if logger is None:
        logger = logging.getLogger(__name__)
    print("guess_secparams_from_rgdist")

    N = DEFAULT_NUM_ENTRIES
    T = DEFAULT_ENTRY_TIME

    W, M1, M2, M3 = compute_4moments(x, y)

    s = np.sqrt(M2)
    init_x0 = M1 - 8*s
    me = 1.5
    mp = 1.5
    m = me + mp

    max_rg = np.max(peak_rgs)
    poresizes = np.asarray(get_all_poresizes())
    possible_poresizes = poresizes[poresizes > max_rg + 20]
    logger.info("trying for rgs=%s against possible_poresizes: %s", str(peak_rgs), str(possible_poresizes))

    dev_list = []
    for poresize in possible_poresizes:
        for N in NUM_ENTRIES_LIST:
            def x0_objective(p):
                T_, x0_ = p
                trv_ = compute_tRv(N, T_, x0_, m, poresize, peak_rgs)
                dev = np.sum((trv_ - peak_trs)**2)
                return dev
            res = minimize(x0_objective, [T, init_x0], method="Nelder-Mead")
            dev_list.append((poresize, N, *res.x, res.fun))

    dev_list = np.asarray(dev_list)
    k = np.argmin(dev_list[:,-1])

    poresize = dev_list[k,0]
    N = dev_list[k,1]
    T = dev_list[k,2]
    x0 = dev_list[k,3]
    print("poresize, N, T, x0 = (%.3g, %d, %.3g, %.3g)" % (poresize, N, T, x0))
    rho = peak_rgs/poresize
    rho[rho > 1] = 1
    niv = N*(1 - rho)**me
    tiv = T*(1 - rho)**mp
    t = x - x0

    evaluate_props = False

    def scales_objective(scales, title=None):
        cy_list = []
        for ni, ti, scale in zip(niv, tiv, scales):
            cy = scale*robust_single_pore_pdf(t, ni, ti)
            cy_list.append(cy)
        ty = np.sum(cy_list, axis=0)
        fv1 = np.sum((ty - y)**2)
        if evaluate_props:
            props_ = compute_area_props(cy_list)
            fv2 = np.sum((props_ - props)**2)
        else:
            fv2 = 0
        if title is not None:
            print(title, fv1, fv2)
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title(title)
                ax.plot(x, y)
                for cy in cy_list:
                    ax.plot(x, cy, ":")
                ax.plot(x, ty, ":", color="red")
                fig.tight_layout()
                plt.show()
        return fv1 + fv2

    num_components = len(peak_rgs)
    init_scales = np.ones(num_components)/num_components
    if debug:
        scales_objective(init_scales, title="before minimize")
    bounds = [(0, None)]*num_components
    res = minimize(scales_objective, init_scales, bounds=bounds, method="Nelder-Mead")
    if debug:
        scales_objective(res.x, title="after minimize")
    return np.concatenate([[N, T, x0, me, mp, poresize], res.x])

def guess_lognormalpore_params_from_rgdist(x, y, peak_rgs, peak_trs, props, debug=True):
    logger = logging.getLogger(__name__)
    print("guess_lognormalpore_params_from_rgdist")
    monopore_params = guess_monopore_params_from_rgdist(x, y, peak_rgs, peak_trs, props, logger=logger, debug=debug)
    N, T, x0, me, mp, poresize = monopore_params[:6]
    init_sigma = DEFAULT_SIGMA
    init_mu = np.log(poresize)

    def mu_sigma_scales_objective(p, title=None):
        mu, sigma = p[0:2]
        cy_list = []
        for rg, scale in zip(peak_rgs, p[2:]):
            cy = lognormal_pore_func(x, scale, N, T, me, mp, mu, sigma, rg, x0)
            cy_list.append(cy)
        ty = np.sum(cy_list, axis=0)
        fv = np.sum((ty - y)**2)
        if title is not None:
            print(title, fv)
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title(title)
                ax.plot(x, y)
                for cy in cy_list:
                    ax.plot(x, cy, ":")
                ax.plot(x, ty, ":", color="red")
                fig.tight_layout()
                plt.show()
        return fv
    
    num_components = len(peak_rgs)
    bounds = [(0, None), (SIGMA_LOWER_BOUND, None)] + [(0, None)]*num_components
    init_scales = np.ones(num_components)/num_components    # consider area
    init_params = np.concatenate([[init_mu, init_sigma], init_scales])
    if debug:
        print("init_params =", init_params)
        mu_sigma_scales_objective(init_params, title="before LNP minimize")
    res = minimize(mu_sigma_scales_objective, init_params, bounds=bounds)
    if debug:
        print("res.x =", res.x)
        mu_sigma_scales_objective(res.x, title="after LNP minimize")
    opt_params = np.concatenate([[N, T, x0, me, mp], res.x]) 
    return opt_params
                                                                        

