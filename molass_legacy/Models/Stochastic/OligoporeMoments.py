"""
    Models.Stochastic.OligoporeMoments.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from scipy.optimize import minimize, basinhopping
from importlib import reload
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.BasicUtils import Struct
from molass_legacy.Models.Stochastic.MomentUtils import compute_egh_moments, to_moderate_props
from molass_legacy.Models.Stochastic.RgReliability import determine_unreliables
from molass_legacy.Models.Stochastic.ParamLimits import USE_K, M1_WEIGHT, M2_WEIGHT, PORESIZE_BOUNDS, BASINHOPPING_SCALE, N0_INIT, OligoporeParamsScaler
from molass_legacy.Models.Stochastic.OligoporeImpl import USE_DISPERSIVE, oligopore_pdf

M1_WEIGHT = 9
M2_WEIGHT = 1
PENALTY_SCALE = 1e6
PORESIZE_LIMIT = 1000

def sec_oligopore_pdf_proof(lrf_src):
    from SecTheory.BasicModels import robust_single_pore_pdf

    N, T, t0, me, mp, poresize = 1000, 0.5, 50, 1.5, 1.5, 100

    pszs = np.ones(3) * poresize
    pszp = np.ones(3)/3

    rg = 30
    rhov = rg/pszs
    ni_ = N * (1 - rhov)**me
    ti_ = T * (1 - rhov)**mp
    x = np.arange(300)
    y = robust_single_pore_pdf(x - t0, ni_[0], ti_[0])
    N0 = N0_INIT if USE_DISPERSIVE else None

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title("Proof of sec_oligopore_pdf compared to robust_single_pore_pdf")
        ax.plot(x, y)
        nt_pairs = [(ni_, ti_) for ni_, ti_ in zip(ni_, ti_)]
        py = oligopore_pdf(x - t0, nt_pairs, pszp, 0, N0=N0)
        ax.plot(x, py)
        py = oligopore_pdf(x, nt_pairs, pszp, t0, N0=N0)
        ax.plot(x, py)        
        ax.axvline(t0, color="red")
        # ax.axvline(x0, color="gray")
        fig.tight_layout()
        ret = plt.show()
    return ret

def study_oligopore_moments_impl(lrf_src,
                                 keep_num_components=False,
                                 want_num_components=None,
                                 trust_max_num=None,
                                 return_guess_params=False,
                                 debug=False):
    if debug:
        import Models.Stochastic.OligoporeUtils as module
        reload(module)
    from molass_legacy.Models.Stochastic.OligoporeUtils import guess_oligopore_colparams, plot_oligopore_conformance

    print('study_oligopore_moments_impl')

    if False:
        ret = sec_oligopore_pdf_proof(lrf_src)
        if not ret:
            return

    all_peaks = lrf_src.xr_peaks
    rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = lrf_src.compute_rgs(keep_num_components=keep_num_components,
                                                                                              want_num_components=want_num_components,
                                                                                              debug=False)
    print("peak_rgs=", peak_rgs)

    peaks = all_peaks[indeces]
    num_peaks = len(peaks)
    egh_moments_list = compute_egh_moments(peaks)
    m_props = to_moderate_props(props)

    # num_pszs = 3        # i.e., Tripore
    num_pszs = 2        # i.e., Dipore
    me = 1.5
    mp = 1.5

    unreliable_indeces = determine_unreliables(peak_rgs, qualities, props, trust_max_num=trust_max_num)
    nur = len(unreliable_indeces)

    use_basinhopping = True         # it seems that global optimization is required for this problem
    params_scaler = OligoporeParamsScaler(num_pszs, egh_moments_list, 0, peak_rgs, unreliable_indeces, dispersive=USE_DISPERSIVE)

    # scales to normalize the parameters for basinhopping
    # param_scales = np.array([100, 0.1, 50, 100] + [10]*num_pszs + [0.1]*num_pszs + [10]*nur)  

    temp_rgs = np.asarray(peak_rgs).copy()
    egh_moments_array = np.asarray(egh_moments_list)

    sep = 0
    slice1 = slice(0, 3)
    sep += 3
    slice2 = slice(sep, sep + num_pszs)
    sep += num_pszs
    slice3 = slice(sep, sep + num_pszs-1)
    sep += num_pszs-1
    slice4 = slice(sep, sep + nur)
    sep += nur
    N0_index = -1
    def split_params(p):
        N_, KT_, x0_ = p[slice1]
        if USE_K:
            T_ = KT_/N_
        else:
            T_ = KT_
        props = p[slice3]
        props_ = np.concatenate([props, [1 - np.sum(props)]])
        N0 = p[N0_index] if USE_DISPERSIVE else None
        return (N_, T_, x0_), p[slice2], props_, p[slice4], N0

    def plp_column_flex_objective(p, debug=False):
        if use_basinhopping:
            p = params_scaler.scale_back(p)
        (N_, T_, x0_), pszv_, pszp_, p_rgs, N0_ = split_params(p)
        temp_rgs[unreliable_indeces] = p_rgs
        
        psz_order_penalty = min(0, np.min(np.diff(pszv_)))**2 *PENALTY_SCALE    # poresizes should be in ascending order
        negative_penalty = min(0, np.min(pszp_))**2 *PENALTY_SCALE

        dev_list = []
        for k, (M, rg) in enumerate(zip(egh_moments_list, temp_rgs)):
            rhov = rg/pszv_
            rhov[rhov > 1] = 1
            M1_ = x0_ + N_ * T_ * np.sum(pszp_ * (1 - rhov)**(me + mp))
            if N0_ is None:
                M2_ = 2 * N_ * T_**2 * np.sum(pszp_ * (1 - rhov)**(me + 2*mp))
            else:
                M2_ = np.sum( 2 * N_ * T_**2 * pszp_ * (1 - rhov)**(me + 2*mp ) + (pszp_ * ((1 - rhov)**(me + mp) + x0_))**2/N0_ )
            dev_list.append(M1_WEIGHT*(M1_ - M[0])**2 + M2_WEIGHT*(M2_ - M[1])**2)
            if debug:
                print([k], "M1, M1_ = %.3g, %.3g" % (M[0], M1_))
                print([k], "M2, M2_ = %.3g, %.3g" % (M[1], M2_))
        if debug:
            print("pszv_, pszp_=", pszv_, pszp_)
            print("psz_order_penalty=", psz_order_penalty)
            print("negative_penalty=", negative_penalty)

        return np.sum(np.asarray(dev_list) * m_props) + psz_order_penalty + negative_penalty

    if USE_K:
        kt_init = np.sqrt(np.sum(egh_moments_array[:,1])) * 30
    else:
        kt_init = 0.5
    t0_init = np.min([M[0] - np.sqrt(M[1])*20 for M in egh_moments_list])
    if num_pszs == 2:
        poresizes_init = [80, 200]
    else:
        poresizes_init = [70, 100, 200]
    assert len(poresizes_init) == num_pszs

    colparams = [1000, kt_init, t0_init] + poresizes_init + list(np.ones(num_pszs-1)/num_pszs)

    plot_info = Struct(x=lrf_src.xr_x, y=lrf_src.xr_y, model=lrf_src.model, peaks=peaks)
    if num_peaks > 2:
        ret = guess_oligopore_colparams(num_pszs, peak_rgs, qualities, egh_moments_list, colparams, m_props, debug=debug, plot_info=plot_info)
        if ret is None:
            return
        if return_guess_params:
            return ret
        N, K, t0, pszs, pszp = ret
        colparams = np.concatenate([[N, K, t0], pszs, pszp[:-1]])   #  remember that pszp has an extra value to make it sum up to unity
        print("colparams=", colparams)

    init_params_list = [colparams,
                        peak_rgs[unreliable_indeces],
                        [N0_INIT] if USE_DISPERSIVE else [],
                        ]

    init_params = np.concatenate(init_params_list)

    if use_basinhopping:
        init_params_ = params_scaler.scale(init_params)
        bounds_ = [(0, BASINHOPPING_SCALE)] * len(init_params)
        plp_column_flex_objective(init_params_, debug=True)
        res = basinhopping(plp_column_flex_objective, init_params_, niter=100,
                           minimizer_kwargs=dict(method='Nelder-Mead', bounds=bounds_))
        plp_column_flex_objective(res.x, debug=True)
        res.x = params_scaler.scale_back(res.x)     # do not forget to scale back the result
    else:
        bounds = params_scaler.get_bounds()
        res = minimize(plp_column_flex_objective, init_params, bounds=bounds, method='Nelder-Mead')

    if debug:
        print("mnp_column_objective: res.x=", res.x)
    
    x = lrf_src.xr_x
    y = lrf_src.xr_y

    (N, T, x0), pszv, pszp, p_rgs, N0 = split_params(res.x)
    temp_rgs[unreliable_indeces] = p_rgs
    abort = False

    def plp_scales_objective(p, return_cy_list=False, title=None):
        debug = title is not None
        # x0_ = p[0]
        scales = p
        cy_list = []
        if debug:
            moments = []
        for k, (rg, scale) in enumerate(zip(temp_rgs, scales)):
            rhov = rg/pszv
            rhov[rhov > 1] = 1
            if debug:
                M1_ = x0 + N * T * np.sum(pszp * (1 - rhov)**(me + mp))
                if N0 is None:
                    M2_ = 2 * N * T**2 * np.sum(pszp * (1 - rhov)**(me + 2*mp ))
                else:
                    M2_ = np.sum( 2 * N * T**2 * pszp * (1 - rhov)**(me + 2*mp ) + (pszp * ((1 - rhov)**(me + mp) + x0))**2/N0 )
                moments.append([M1_, M2_])
            ni_ = N * (1 - rhov)**me
            ti_ = T * (1 - rhov)**mp
            nt_pairs = [(ni_, ti_) for ni_, ti_ in zip(ni_, ti_)]
            cy = scale * oligopore_pdf(x, nt_pairs, pszp, x0, N0=N0)
            cy_list.append(cy)
        if return_cy_list:
            return cy_list
        ty = np.sum(cy_list, axis=0)
        dev = np.sum((ty - y)**2)
        if debug:
            def plot_details():
                import Models.Stochastic.OligoporeUtils
                reload(Models.Stochastic.OligoporeUtils)
                from molass_legacy.Models.Stochastic.OligoporeUtils import plot_oligopore_moments_state
                params_list = [(N, T, x0), pszv, pszp, scales]      # not including me, mp here
                if N0 is not None:
                    params_list += [[N0]]
                olgp_params = np.concatenate(params_list)
                ret = plot_oligopore_moments_state(x, y, cy_list, ty, egh_moments_list, moments, split_params, olgp_params, temp_rgs, params_scaler, title=title)

            with plt.Dp(button_spec=['OK', 'Cancel'],
                        extra_button_specs=[("Plot Details", plot_details)]
                        ):
                fig, ax = plt.subplots()
                ax.set_title(title)
                ax.plot(x, y, color='orange', label='data')
                for k, cy in enumerate(cy_list):
                    ax.plot(x, cy, ":", label='component-%d' % k)
                ax.plot(x, ty, ":", color='red', label='model total')
                fig.tight_layout()
                ret = plt.show()
            if not ret:
                nonlocal abort
                abort = True
        return dev
    
    cy_list = plp_scales_objective(np.ones(num_peaks), return_cy_list=True)
    init_scales = []
    for cy, h in zip(cy_list, peaks[:,0]):
        init_scales.append(h/np.max(cy))
    init_params = np.array(init_scales)
    if debug:
        plp_scales_objective(init_params, title="before MNP with unreliables %s minimize" % unreliable_indeces)
        if abort:
            return
    # bounds = [(-500, 500)] + [(0, 10)]*num_peaks
    bounds = [(0, 10)]*num_peaks
    if True:
        res = minimize(plp_scales_objective, init_params, bounds=bounds, method='Nelder-Mead')
    else:
        res = basinhopping(plp_scales_objective, init_params, niter=100,
                           minimizer_kwargs=dict(method='Nelder-Mead', bounds=bounds))        
    if debug:
        plp_scales_objective(res.x, title="after MNP with unreliables %s minimize" % unreliable_indeces)
        if abort:
            return

    # x0 = res.x[0]
    scales = res.x
    params_list = [(N, T, x0, me, mp), pszv, pszp, scales]
    if N0 is not None:
        params_list += [[N0]]

    oligopore_params = np.concatenate(params_list)
    if debug:
        K = N*T
        ret = plot_oligopore_conformance(N, K, t0, pszs, pszp, plot_info, rg_info=(temp_rgs, peak_trs, True))
        if not ret:
            return

    return oligopore_params, temp_rgs, unreliable_indeces