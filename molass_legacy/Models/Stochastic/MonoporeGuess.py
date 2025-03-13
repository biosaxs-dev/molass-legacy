"""
    Models.Stochastic.MonoporeGuess.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from importlib import reload
import molass_legacy.KekLib.DebugPlot as plt
from SecTheory.BasicModels import robust_single_pore_pdf
from molass_legacy.Models.ModelUtils import compute_raw_moments1, compute_area_props
from molass_legacy.Models.Stochastic.ParamLimits import MAX_PORESIZE

EVALUATE_MOMENT3 = False
M1_WEIGHT = 9
M2_WEIGHT = 1
M3_WEIGHT = 0

def guess_monopore_params_using_moments(x, y, egh_moments_list, peak_rgs, props, monopore_params=None, debug=False):
    N, T, x0, me, mp, poresize = monopore_params[0:6]

    def column_objective(p, debug=False):
        N_, T_, x0_, poresize_ = p
        rhov = peak_rgs/poresize_
        rhov[rhov > 1] = 1
        dev_list = []
        for k, (M, rho) in enumerate(zip(egh_moments_list, rhov)):
            M1_ = x0_ + N_ * T_ * (1 - rho)**(me + mp)
            M2_ = 2 * N_ * T_**2 * (1 - rho)**(me + 2*mp)
            if EVALUATE_MOMENT3:
                M3_ = 6 * N * T**3 * (1 - rho)**(me + 3*mp)
                dev_list.append(M1_WEIGHT*(M1_ - M[0])**2 + M2_WEIGHT*(M2_ - M[1])**2 + M3_WEIGHT*(M3_ - M[2])**2)
            else:
                dev_list.append(M1_WEIGHT*(M1_ - M[0])**2 + M2_WEIGHT*(M2_ - M[1])**2)
            if debug:
                print([k], "M1, M1_ = %.3g, %.3g" % (M[0], M1_))
                print([k], "M2, M2_ = %.3g, %.3g" % (M[1], M2_))
                if EVALUATE_MOMENT3:
                    print([k], "M3, M3_ = %.3g, %.3g" % (M[2], M3_))
        return np.sum(np.asarray(dev_list)*props)

    init_params = [N, T, x0, poresize]
    if debug:
        print("init_params=", init_params)
        print("before minimize: fv=", column_objective(init_params, debug=True))

    bounds = [(300, 5000), (0.01, 10), (x0-50, x0+50), (0, MAX_PORESIZE)]
    res = minimize(column_objective, init_params, bounds=bounds, method='Nelder-Mead')
    if debug:
        print("res.x=", res.x)
        print("after minimize: fv=", column_objective(res.x, debug=True))

    N, T, x0, poresize = res.x

    use_nt_ratio = False 
    use_multi_factor_fv = False
    if use_multi_factor_fv:
        raw_moments = np.array([M[0] for M in egh_moments_list])

    def x0_scales_objective(p, title=None):
        if use_nt_ratio:
            K, ratio, x0_ = p[0:3]  # K = N*T, ratio = N/T
            N_ = np.sqrt(K*ratio)   # T = N/ratio, K = N*T = N**2 / ratio
            T_ = np.sqrt(K/ratio)   # N = T*ratio, K = N*T = T**2 * ratio
        else:
            N_, T_, x0_ = p[0:3]
        scales_ = p[3:]
        rhov = peak_rgs/poresize
        rhov[rhov > 1] = 1
        cy_list = []
        for k, (rho, scale) in enumerate(zip(rhov, scales_)):
            ni_ = N_*(1 - rho)**me
            ti_ = T_*(1 - rho)**mp
            cy = scale*robust_single_pore_pdf(x - x0_, ni_, ti_)
            cy_list.append(cy)    
        ty = np.sum(cy_list, axis=0)
        if use_multi_factor_fv:
            raw_moments_ = compute_raw_moments1(x, cy_list)
            props_ = compute_area_props(cy_list)

        if title is not None:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title(title)
                ax.plot(x, y, 'o')
                for cy in cy_list:
                    ax.plot(x, cy, '-')
                ax.plot(x, ty, ':', color="red")
                fig.tight_layout()
                plt.show()
        if use_multi_factor_fv:
            dev = np.sum((ty - y)**2)**2 + 0.00005*np.sum((raw_moments_ - raw_moments)**2)**2 + 0.00005*np.sum((props_ - props)**2)**2
        else:
            dev = np.sum((ty - y)**2)
        return dev

    num_components = len(peak_rgs)
    scales = monopore_params[6:]
    if use_nt_ratio:
        K = N*T
        ratio = N/T
        init_params = np.concatenate([[K, ratio, x0], scales])
    else:
        init_params = np.concatenate([[N, T, x0], scales])
    if debug:
        print("init_params=", init_params)
        print("before minimize: fv=", x0_scales_objective(init_params, title="before x0_scales_objective minimize"))

    bounds = [(300, 5000), (0.01, 10), (x0-50, x0+50)] + [(0, 10)]*num_components
    res = minimize(x0_scales_objective, init_params, method='Nelder-Mead', bounds=bounds)
    if debug:
        print("res.x=", res.x)
        print("after minimize: fv=", x0_scales_objective(res.x, title="after x0_scales_objective minimize"))
    if use_nt_ratio:
        K, ratio, x0 = res.x[0:3]
        N = np.sqrt(K*ratio)
        T = np.sqrt(K/ratio)
    else:
        N, T, x0 = res.x[0:3]
    scales = res.x[3:]
    monopore_params = np.concatenate([[N, T, x0, me, mp, poresize], scales])
    return monopore_params
