"""
    RgSecUpdater.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from SecTheory.SecEstimator import guess_initial_secparams

USE_BASIN_HOPPING = True
if USE_BASIN_HOPPING:
    from scipy.optimize import basinhopping
else:
    from scipy.optimize import minimize

def update_optimizer_impl(optimizer, params, rg_params, rg_qualities):
    print("update_optimizer_impl: rg_params=", rg_params, "rg_qualities=", rg_qualities)

    separate_params = optimizer.split_params_simple(params)
    separate_params[2] = rg_params

    optimizer.init_rgs = rg_params

    init_xr_params = separate_params[0]
    Npc, rp, tI, t0, P, m = guess_initial_secparams(init_xr_params, rg_params)
    trs = init_xr_params[:,1]
    mu, sigma = init_xr_params[0,[1,2]]

    bounds = [(tI, mu-sigma), (0, P*2), (0, 3)]
    init_params = [t0, P, m]
    rhos = rg_params/rp
    rhos[rhos > 1] = 1

    def fixed_poresize_objective(p):
        t0_, P_, m_ = p
        trs_ = t0_ + P_*np.power(1 - rhos, m_)
        return np.sum(((trs_ - trs)*rg_qualities)**2)

    objective_func = fixed_poresize_objective

    if USE_BASIN_HOPPING:
        minimizer_kwargs = dict(method="Nelder-Mead", bounds=bounds)
        ret = basinhopping(objective_func, init_params, minimizer_kwargs=minimizer_kwargs)
    else:
        ret = minimize(objective_func, init_params, method="Nelder-Mead", bounds=bounds)

    rp_ = rp
    t0_, P_, m_ = ret.x

    init_sec_params = np.array([Npc, rp_, tI, t0_, P_, m_])
    separate_params[7] = init_sec_params

    optimizer.init_separate_params = separate_params

    x = np.concatenate([init_xr_params.flatten(),*separate_params[1:]])
    optimizer.init_params = x
    optimizer.update_bounds(x)

    return x
