"""
    Models.Stochastic.MonoporeMoments.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from scipy.optimize import minimize, basinhopping
from importlib import reload
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.BasicUtils import Struct
from molass_legacy.Models.Stochastic.MomentUtils import to_moderate_props
from molass_legacy.Models.Stochastic.RgReliability import determine_unreliables
from SecTheory.BasicModels import robust_single_pore_pdf
from molass_legacy.Models.Stochastic.ParamLimits import USE_K, M1_WEIGHT, M2_WEIGHT, BASINHOPPING_SCALE, MonoporeParamsScaler

def collate_monopore_moments():
    print("collate_monopore_moments")

def study_monopore_moments_impl(lrf_src,
                                use_basinhopping = True,        # it seems that global optimization is required for this problem
                                use_timescale = True,           # timescale parameter, which will be removed eventually,
                                                                # is used to help the optimizer by relating the changes of t0 and T
                                keep_num_components=False,
                                want_num_components=4,          # this should be eventually want_num_components=None
                                trust_all_rgs=False,            # used for illustration
                                return_init_params=False,       # uses for animation
                                return_init_moments=False,      # used for illustration
                                return_step1_moments=False,     # used for illustration
                                return_step2_moments=False,     # used for illustration
                                trust_max_num=None,
                                debug=False):
    if debug:
        import Models.Stochastic.MonoporeUtils as module
        reload(module)
        import Models.Stochastic.ExclusionCurves as module
        reload(module)

    from molass_legacy.Models.Stochastic.MonoporeUtils import guess_monopore_colparams
    from molass_legacy.Models.Stochastic.ExclusionCurves import compute_simply_conformant_rgs

    print('study_monopore_moments_impl')
    logger = lrf_src.logger

    rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = lrf_src.compute_rgs(keep_num_components=keep_num_components,
                                                                                              want_num_components=want_num_components,  debug=False)
    logger.info("study_monopore_moments_impl entry: peak_rgs=%s", peak_rgs)
    peaks = lrf_src.get_peaks()
    num_peaks = len(peaks)
    egh_moments_list = lrf_src.get_egh_moments_list()
    m_props = to_moderate_props(props)

    me = 1.5
    mp = 1.5

    if trust_all_rgs:
        unreliable_indeces = np.array([], dtype=int)
    else:
        unreliable_indeces = determine_unreliables(peak_rgs, qualities, props, trust_max_num=trust_max_num)
    nur = len(unreliable_indeces)

    use_conformant_rgs = False

    adjust_rgs_at_start_only = False
    u_indeces = np.array([], dtype=int) if (use_conformant_rgs or adjust_rgs_at_start_only) else unreliable_indeces
    num_timescales = 1 if use_timescale else 0
    params_scaler = MonoporeParamsScaler(egh_moments_list, 0, peak_rgs, u_indeces, num_timescales=num_timescales, allow_near_t0=False)

    temp_rgs = np.asarray(peak_rgs).copy()
    logger.info("initial input info: temp_rgs=%s, unreliable_indeces=%s", temp_rgs, unreliable_indeces)
    egh_moments_array = np.asarray(egh_moments_list)
    if num_peaks > 1:
        eghm1_min_diff = np.min(np.abs(np.diff(egh_moments_array[:,0])))
        if debug:
            print("eghm1_min_diff=", eghm1_min_diff)
        MIN_DIFF_PENALTY_SCALE = 1e6
    
    apply_diff_penalty = True
    def mnp_column_flex_objective(p, return_moments=False, debug=False):
        if use_basinhopping:
            p = params_scaler.scale_back(p)
        N_, KT_, x0_, poresize_ = p[0:4]
        if USE_K:
            T_ = KT_/N_
        else:
            T_ = KT_
        if use_timescale:
            T_ *= p[-1]
            x0_ *= p[-1]
        if not adjust_rgs_at_start_only:
            if nur > 0:
                if use_conformant_rgs:
                    temp_rgs[unreliable_indeces] = compute_simply_conformant_rgs(N_, KT_, x0_, poresize_, 3, peak_trs[unreliable_indeces], debug=debug, rgs=peak_rgs[unreliable_indeces])
                else:
                    temp_rgs[unreliable_indeces] = p[4:4+nur]

        if len(temp_rgs) > 1:
            # Rg's should be in descending order
            order_penalty = max(0, np.max(np.diff(temp_rgs))) * MIN_DIFF_PENALTY_SCALE
        else:
            order_penalty = 0

        rhov = temp_rgs/poresize_
        rhov[rhov > 1] = 1
        dev_list = []
        moments_list = []
        for k, (M, rho) in enumerate(zip(egh_moments_list, rhov)):
            M1_ = x0_ + N_ * T_ * (1 - rho)**(me + mp)
            M2_ = 2 * N_ * T_**2 * (1 - rho)**(me + 2*mp)
            dev_list.append(M1_WEIGHT*(M1_ - M[0])**2 + M2_WEIGHT*(M2_ - M[1])**2)
            if return_moments:
                moments_list.append([M1_, M2_])
            else:
                moments_list.append(M1_)
            if debug:
                print([k], "M1, M1_ = %.3g, %.3g" % (M[0], M1_))
                print([k], "M2, M2_ = %.3g, %.3g" % (M[1], M2_))

        if return_moments:
            return moments_list

        if apply_diff_penalty and num_peaks > 1:
            m1_min_diff = np.min(np.abs(np.diff(moments_list)))
            diff_penalty = max(0, abs(1 - m1_min_diff/eghm1_min_diff) - 0.1)**2 * MIN_DIFF_PENALTY_SCALE
        else:
            diff_penalty = 0
        fv = np.sum(np.asarray(dev_list) * m_props) + diff_penalty + order_penalty   # seems better not to weight with props
        if debug:
            print("m_props=", m_props)
            print("order_penalty=", order_penalty)
            print("diff_penalty=", diff_penalty)
            print("fv=", fv)
        return fv

    if USE_K:
        kt_init = np.sqrt(np.sum(egh_moments_array[:,1])) * 30
        # kt_init = 500
    else:
        kt_init = 0.5
    t0_init = np.min([M[0] - np.sqrt(M[1])*20 for M in egh_moments_list])
    colparams = [1000, kt_init, t0_init, 200]

    plot_info = Struct(x=lrf_src.xr_x, y=lrf_src.xr_y, model=lrf_src.model, peaks=peaks)
    if num_peaks > 3:
        ret = guess_monopore_colparams(peak_rgs, qualities, peak_trs, colparams, debug=debug, plot_info=plot_info)
        if ret is None:
            return
        colparams = list(ret)
        print("colparams=", colparams)

    rgs = peak_rgs[unreliable_indeces]
    if use_conformant_rgs:
        conformant_rgs = []
    else:
        trs = peak_trs[unreliable_indeces]
        ret = compute_simply_conformant_rgs(*colparams, 3, trs, debug=debug, rgs=rgs, plot_info=plot_info)
        if ret is None:
            return
        temp_rgs[unreliable_indeces] = ret
        conformant_rgs = list(ret)
        logger.info("using conformant Rg's %s from trs=%s, for unreliable peaks at start only", conformant_rgs, trs)

    timescale = [1] if use_timescale else []
    param_rgs = [] if adjust_rgs_at_start_only else conformant_rgs
    init_params = colparams + param_rgs + timescale
 
    if return_init_params:
        return init_params

    if use_basinhopping:
        init_params_ = params_scaler.scale(init_params)
        if return_init_moments:
            return mnp_column_flex_objective(init_params_, return_moments=True)

        bounds_ = [(0, BASINHOPPING_SCALE)] * len(init_params)
        if debug:
            print("mnp_column_objective:init_params=", init_params)
            print("mnp_column_objective:init_params_=", init_params_)
            print("mnp_column_objective:bounds=", params_scaler.get_bounds())
            print("mnp_column_objective:bounds_=", bounds_)
            mnp_column_flex_objective(init_params_, debug=True)

        minimizer_kwargs = dict(method='Nelder-Mead', bounds=bounds_)
        # minimizer_kwargs = dict(method='trust-constr', bounds=bounds_)
        # minimizer_kwargs = dict(method='SLSQP', bounds=bounds_)
        # minimizer_kwargs = dict(method='COBYLA', bounds=bounds_)
        # minimizer_kwargs = dict(bounds=bounds_)
        res = basinhopping(mnp_column_flex_objective, init_params_,
                           # stepsize=0.5,
                           niter=100,
                           minimizer_kwargs=minimizer_kwargs)

        mnp_column_flex_objective(res.x, debug=True)
        if return_step1_moments:
            return mnp_column_flex_objective(res.x, return_moments=True)

        res.x = params_scaler.scale_back(res.x)     # do not forget to scale back the result
    else:
        bounds = params_scaler.get_bounds()
        if debug:
            print("mnp_column_objective:init_params=", init_params)
            print("mnp_column_objective:bounds=", bounds)
        
        res = minimize(mnp_column_flex_objective, init_params, bounds=bounds, method='Nelder-Mead')
        if return_step1_moments:
            return mnp_column_flex_objective(res.x, return_moments=True)

    if debug:
        print("mnp_column_objective: res.x=", res.x)
    
    x = lrf_src.xr_x
    y = lrf_src.xr_y

    N, KT, x0, poresize = res.x[0:4]
    if USE_K:
        T = KT/N
    else:
        T = KT
    if use_timescale:
        T *= res.x[-1]
        x0 *= res.x[-1]
    if nur > 0:
        if use_conformant_rgs:
            temp_rgs[unreliable_indeces] = compute_simply_conformant_rgs(N, KT, x0, poresize, 3, peak_trs[unreliable_indeces])
        else:
            temp_rgs[unreliable_indeces] = res.x[4:4+nur]
    rhov = temp_rgs/poresize
    rhov[rhov > 1] = 1
    abort = False

    def mnp_scales_objective(p, return_cy_list=False, return_moments=False, title=None):
        debug = title is not None
        T_ = T*p[-1]
        x0_ = p[0] * p[-1]
        scales = p[1:1+num_peaks]
        cy_list = []
        if debug or return_moments:
            moments_list = []
        for k, (rho, scale) in enumerate(zip(rhov, scales)):
            ni_ = N * (1 - rho)**me
            ti_ = T_ * (1 - rho)**mp
            if debug or return_moments:
                M1_ = x0_ + ni_ * ti_
                M2_ = 2 * ni_ * ti_**2
                moments_list.append([M1_, M2_])
            cy = scale * robust_single_pore_pdf(x - x0_, ni_, ti_)
            cy_list.append(cy)
        if return_cy_list:
            return cy_list
        if return_moments:
            return moments_list
        ty = np.sum(cy_list, axis=0)
        dev = np.sum((ty - y)**2)
        if debug:
            from molass_legacy.Models.Stochastic.MonoporeUtils import plot_monopore_moments_state
            title += "; FV=%.3g" % dev
            mnp_params = np.concatenate([[N, T, x0_, me, mp, poresize], scales])
            bounds = params_scaler.get_bounds()
            t0_upper = bounds[2][1]
            ret = plot_monopore_moments_state(title, x, y, temp_rgs, mnp_params, t0_upper, cy_list, ty, moments_list, egh_moments_list)
            if not ret:
                nonlocal abort
                abort = True
        return dev
    
    x0_ones = np.concatenate([[x0], np.ones(num_peaks)])
    cy_list = mnp_scales_objective(x0_ones, return_cy_list=True)
    init_scales = []
    for cy, h in zip(cy_list, peaks[:,0]):
        init_scales.append(h/np.max(cy))
    timescale = [1] if use_timescale else []
    init_params = np.concatenate([[x0], init_scales, timescale])
    if debug:
        mnp_scales_objective(init_params, title="before MNP scales optimize with unreliables %s minimize" % unreliable_indeces)
        if abort:
            return
    bounds = [(-1000, 1000)] + [(0, 10)]*num_peaks + [(0, 2)]
    print("init_params=", init_params)
    print("bounds=", bounds)
    res = minimize(mnp_scales_objective, init_params, bounds=bounds, method='Nelder-Mead')
    if debug:
        mnp_scales_objective(res.x, title="after MNP scales optimize with unreliables %s minimize" % unreliable_indeces)
        if abort:
            return

    if return_step2_moments:
        return mnp_scales_objective(res.x, return_moments=True), unreliable_indeces

    x0 = res.x[0]
    if use_timescale:
        x0 *= res.x[-1]
        T *= res.x[-1]
    scales = res.x[1:1+num_peaks]
    mnp_params = np.concatenate([(N, T, x0, me, mp, poresize), scales])
    if debug and use_timescale:
        temp_params = np.concatenate([[x0], scales, timescale])
        mnp_scales_objective(temp_params, title="timescale proof")
        if abort:
            return
    return mnp_params, temp_rgs, unreliable_indeces, params_scaler