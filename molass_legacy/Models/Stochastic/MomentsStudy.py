"""
    Models.Stochastic.MomentsStudy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.integrate import quad
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.Stochastic.MomentUtils import to_moderate_props
from SecTheory.BasicModels import robust_single_pore_pdf
from molass_legacy.Models.Stochastic.LognormalPoreFunc import distr_func, lognormal_pore_func
from molass_legacy.Models.Stochastic.LognormalUtils import compute_mu_sigma, compute_mode, compute_mu_sigma_from_mean, compute_mean_from_mode
from molass_legacy.Models.Stochastic.ParamLimits import (
    USE_K, KT_BOUND, N_BOUND,
    MAX_PORESIZE, M1_WEIGHT, M2_WEIGHT, PORESIZE_BOUNDS, PORESIZE_MEAN_BOUND, PORESIZE_STDEV_BOUND)

def moments_study_impl(arg_lrf_src, return_rgs=False, use_unreliables=True, correct_entirely=True, progress_cb=None, show_result=False, debug=False):
    from importlib import reload
    import Selective.DataFilter
    reload(Selective.DataFilter)
    from Selective.DataFilter import DataFilter
    filter = DataFilter(arg_lrf_src)
    lrf_src = filter.get_filtered_src()

    logger = logging.getLogger(__name__)

    x = lrf_src.xr_x
    y = lrf_src.xr_y
    rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = lrf_src.compute_rgs(keep_num_components=False, debug=False)
    peaks = lrf_src.get_peaks()
    egh_moments_list = lrf_src.get_egh_moments_list()
    if use_unreliables:
        from importlib import reload
        import Models.Stochastic.MonoporeMoments as module
        reload(module)
        from molass_legacy.Models.Stochastic.MonoporeMoments import study_monopore_moments_impl
        ret = study_monopore_moments_impl(lrf_src, debug=debug)
        if ret is None:
            return
        mnp_params, temp_rgs, unreliable_indeces, params_scaler = ret
        if len(unreliable_indeces) > 0:
            if correct_entirely:
                corrected_rgs = temp_rgs
            else:
                corrected_rgs = peak_rgs.copy()
                corrected_rgs[unreliable_indeces] = peak_rgs[unreliable_indeces]*0.1 + temp_rgs[unreliable_indeces]*0.9
            logger.info("peak_rgs=%s are replaced by corrected_rgs=%s with unreliable_indeces=%s", peak_rgs, corrected_rgs, unreliable_indeces)
            peak_rgs = corrected_rgs
    else:
        mnp_params = monopore_study(x, y, peaks, peak_rgs, props, egh_moments_list, logger=logger, debug=debug)
        unreliable_indeces = np.array([], dtype=int)
    if mnp_params is None:
        return
    if progress_cb is not None:
        progress_cb(1)
    lnp_params = lnpore_study(x, y, peaks, peak_rgs, props, egh_moments_list, mnp_params, params_scaler, unreliable_indeces, logger=logger, progress_cb=progress_cb, show_result=show_result, debug=debug)
    if progress_cb is not None:
        progress_cb(10)
    if return_rgs:
        return lnp_params, peak_rgs
    else:
        return lnp_params

def monopore_study(x, y, peaks, peak_rgs, props, egh_moments_list, logger=None, progress_cb=None, debug=False):
    num_peaks = len(peaks)
    m_props = to_moderate_props(props)
    if debug:
        print("monopore_study: num_peaks=", num_peaks, "peak_rgs=", peak_rgs, "props=", props, "m_props=", m_props)
    me = 1.5
    mp = 1.5

    use_K = True
    use_basinhopping = True                     # it seems that global optimization is required for this problem
    kt_scale = 100 if use_K else 0.1
    param_scales = np.array([100, kt_scale, 10, 10])    # scales to normalize the parameters for basinhopping

    def mnp_column_objective(p, debug=False):
        if use_basinhopping:
            p *= param_scales
        N_, kt, x0_, poresize_ = p
        if USE_K:
            T_ = kt/N_
        else:
            T_ = kt
        rhov = peak_rgs/poresize_
        rhov[rhov > 1] = 1
        dev_list = []
        for k, (M, rho) in enumerate(zip(egh_moments_list, rhov)):
            M1_ = x0_ + N_ * T_ * (1 - rho)**(me + mp)
            M2_ = 2 * N_ * T_**2 * (1 - rho)**(me + 2*mp)
            dev_list.append(M1_WEIGHT*(M1_ - M[0])**2 + M2_WEIGHT*(M2_ - M[1])**2)
            if debug:
                print([k], "M1, M1_ = %.3g, %.3g" % (M[0], M1_))
                print([k], "M2, M2_ = %.3g, %.3g" % (M[1], M2_))
        return np.sum(np.asarray(dev_list) * m_props)   # seems better not to weight with props

    if USE_K:
        kt = 500
    else:
        kt = 0.25
    init_params = [2000, kt, 0, 200]
    eghM = egh_moments_list[0]
    x0_upper = eghM[0] - 5*np.sqrt(eghM[1])
    bounds = [N_BOUND, KT_BOUND, (-1000, x0_upper), PORESIZE_BOUNDS]

    if use_basinhopping:
        init_params_ = np.array(init_params)/param_scales
        bounds_ = [(lower/scale, upper/scale) for (lower, upper), scale in zip(bounds, param_scales)]
        res = basinhopping(mnp_column_objective, init_params_, niter=100,
                           minimizer_kwargs=dict(method='Nelder-Mead', bounds=bounds_))
        res.x *= param_scales   # do not forget to scale back the result
    else:
        res = minimize(mnp_column_objective, init_params, bounds=bounds, method='Nelder-Mead')

    if debug:
        print("mnp_column_objective: res.x=", res.x)

    max_m2 = np.max([M[1] for M in egh_moments_list])
    score = np.sqrt(res.fun)/num_peaks/max_m2
    print("max_m2=", max_m2,  "score = ", score)
    """
    Ald     153    0.14
    OA_Ald  155    0.19
    """

    N, kt, x0, poresize = res.x
    if USE_K:
        T = kt/N
    else:
        T = kt
    rhov = peak_rgs/poresize
    rhov[rhov > 1] = 1
    abort = False

    def mnp_scales_objective(p, title=None):
        debug = title is not None
        x0_ = p[0]
        scales = p[1:]
        cy_list = []
        if debug:
            moments = []
        for k, (rho, scale) in enumerate(zip(rhov, scales)):
            ni_ = N * (1 - rho)**me
            ti_ = T * (1 - rho)**mp
            if debug:
                M1_ = x0_ + ni_ * ti_
                M2_ = 2 * ni_ * ti_**2
                moments.append([M1_, M2_])
            cy = scale * robust_single_pore_pdf(x - x0_, ni_, ti_)
            cy_list.append(cy)
        ty = np.sum(cy_list, axis=0)
        dev = np.sum((ty - y)**2)
        if debug:
            from importlib import reload
            import Models.Stochastic.MonoporeUtils as module
            reload(module)
            from molass_legacy.Models.Stochastic.MonoporeUtils import plot_monopore_moments_state
            mnp_params = N, T, x0_, me, mp, poresize
            ret = plot_monopore_moments_state(title, x, y, peak_rgs, mnp_params, x0_upper, cy_list, ty, moments, egh_moments_list)
            if not ret:
                nonlocal abort
                abort = True
        return dev
    
    area = np.sum(x*y)/np.sum(y)
    scale = 0.5*area*np.max(y)/num_peaks    # task: better to simplify this scale, though it is irrelevant here.
    print("area = ", area)
    init_params = np.concatenate([[x0], props*scale])
    if debug:
        mnp_scales_objective(init_params, title="before MNP minimize")
        if abort:
            return
    bounds = [(-1000, 1000)] + [(0, 10)]*num_peaks
    res = minimize(mnp_scales_objective, init_params, bounds=bounds, method='Nelder-Mead')
    if debug:
        mnp_scales_objective(res.x, title="after MNP minimize")
        if abort:
            return

    x0 = res.x[0]
    scales = res.x[1:]
    return np.concatenate([(N, T, x0, me, mp, poresize), scales])

def lnpore_study(x, y, peaks, peak_rgs, props, egh_moments_list, mnp_params, params_scaler, unreliable_indeces, logger=None, progress_cb=None, show_result=False, debug=False):
    from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
    num_peaks = len(peaks)
    m_props = to_moderate_props(props)
    logger.info("lnpore_study: num_peaks=%d, mnp_params=%s", num_peaks, mnp_params)
    me = 1.5
    mp = 1.5
    N, T, init_x0, me, mp, poresize = mnp_params[0:6]

    nur = len(unreliable_indeces)
    temp_rgs = peak_rgs.copy()

    def lnp_column_objective(p, debug=False):
        N_, TK_, x0_, mean_, stdev_ = p[0:5]
        if USE_K:
            T_ = TK_/N_
        else:
            T_ = TK_
        mu_, sigma_ = compute_mu_sigma_from_mean(mean_, stdev_)
        if nur > 0:
            temp_rgs[unreliable_indeces] = p[5:]

        dev_list = []
        for k, (M, rg) in enumerate(zip(egh_moments_list, temp_rgs)):
            M1_ = x0_ + N_ * T_    * quad(lambda r : distr_func(r, mu_, sigma_) * (1 - min(1, rg/r))**(me +   mp), rg, MAX_PORESIZE)[0]
            M2_ =   2 * N_ * T_**2 * quad(lambda r : distr_func(r, mu_, sigma_) * (1 - min(1, rg/r))**(me + 2*mp), rg, MAX_PORESIZE)[0]
            dev_list.append(M1_WEIGHT*(M1_ - (M[0]))**2 + M2_WEIGHT*(M2_ - M[1])**2)
            if debug:
                print([k], "N_=%d, T_=%.3g, x0_=%.3g, rg=%.3g" % (N_, T_, x0_, rg))
                print([k], "mean_=%.3g, stdev_=%.3g, mu_=%.3g, sigma_=%.3g" % (mean_, stdev_, mu_, sigma_))
                print([k], "M1, M1_ = %.3g, %.3g" % (M[0], M1_))
                print([k], "M2, M2_ = %.3g, %.3g" % (M[1], M2_))
        return np.sum(np.asarray(dev_list)*m_props)

    stdev = poresize*0.1
    mean = compute_mean_from_mode(poresize, stdev)
    # init_N = min(2000, N)   # too large initial N can be hamful?
    init_N = N
    # init_T = N*T/init_N
    if USE_K:
        init_KT = N*T
    else:
        init_KT = T
    init_params = [init_N, init_KT, init_x0, mean, stdev] + list(temp_rgs[unreliable_indeces])
    if debug or True:
        print("init_params for lnp_column_objective = ", init_params)
        lnp_column_objective(init_params, debug=True)

    mnp_bounds = params_scaler.get_bounds()
    # mnp_bounds[0][1] = 1500     # on what condition?
    bounds = np.vstack([mnp_bounds[0:3], PORESIZE_MEAN_BOUND, PORESIZE_STDEV_BOUND] + [(10, 100)]*nur)

    res = minimize(lnp_column_objective, init_params, bounds=bounds, method='Nelder-Mead')
    if debug or True:
        logger.info("res.x from lnp_column_objective = %s", res.x)
        lnp_column_objective(res.x, debug=True)

    if progress_cb is not None:
        progress_cb(5)

    N, KT, x0, mean, stdev = res.x[0:5]
    if nur > 0:
        temp_rgs[unreliable_indeces] = res.x[5:]
        logger.info("temp_rgs have been updated to %s with unreliable_indeces=%s",  temp_rgs, unreliable_indeces)

    if USE_K:
        T = KT/N
    else:
        T = KT
    mu, sigma = compute_mu_sigma_from_mean(mean, stdev)
    mode = compute_mode(mu, sigma)
    
    logger.info("mode=%.3g from LNP optimized mu=%.3g, sigma=%.3g", mode, mu, sigma)
    logger.info("N for lnp_scales_objective=%d", N)

    abort = False

    def lnp_scales_objective(p, title=None):
        x0_ = p[0]
        scales = p[1:]
        rhov = peak_rgs/poresize
        rhov[rhov > 1] = 1
        cy_list = []
        for k, (rg, scale) in enumerate(zip(temp_rgs, scales)):
            cy = lognormal_pore_func(x, scale, N, T, me, mp, mu, sigma, rg, x0_)
            cy_list.append(cy)
        ty = np.sum(cy_list, axis=0)
        dev = np.sum((ty - y)**2)
        if title is not None:
            from importlib import reload
            import Models.Stochastic.LognormalUtils as lnp_utils
            reload(lnp_utils)
            from molass_legacy.Models.Stochastic.LnporeUtils import plot_lognormal_fitting_state
            lnp_params = np.concatenate([(N, T, x0, me, mp, mu, sigma), scales])
            ret = plot_lognormal_fitting_state(x, y, lnp_params, peak_rgs, title=title)
            if not ret:
                nonlocal abort
                abort = True
        return dev
    
    init_params = np.concatenate([[x0], mnp_params[6:]])
    logger.info("init_params for lnp_scales_objective=%s", init_params)
    if debug:
        title = "Lognormal PSD Elution Components for %s (before minimize)" % get_in_folder()
        lnp_scales_objective(init_params, title=title)
        if abort:
            print("aborting LNP study")
            return
    bounds = [(x0-100, x0+100)] + [(0, 10)]*num_peaks
    res = minimize(lnp_scales_objective, init_params, bounds=bounds, method='Nelder-Mead')
    if debug or show_result:
        print("showing result of lnp_scales_objective = ", res.x)
        title = "Lognormal PSD Elution Components for %s (after minimize)" % get_in_folder()
        lnp_scales_objective(res.x, title=title)
    x0 = res.x[0]

    return np.concatenate([(N, T, x0, me, mp, mu, sigma), res.x[1:]])