"""
    Models.Stochastic.BasinhoppingAnimation.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from molass_legacy.Models.Stochastic.MomentUtils import to_moderate_props
from molass_legacy.Models.Stochastic.ParamLimits import USE_K, M1_WEIGHT, M2_WEIGHT, BASINHOPPING_SCALE, MonoporeParamsScaler
from molass_legacy.Models.Stochastic.MonoporeMoments import study_monopore_moments_impl
from molass_legacy.Models.Stochastic.MonoporeUtils import plot_monopore_curves
from SecTheory.BasicModels import robust_single_pore_pdf
from molass_legacy.Models.Stochastic.MomentsCollation import plot_collation_state_impl
from molass_legacy.Models.Stochastic.RgReliability import determine_unreliables

def basinhopping_moment_animation_impl(lrf_src,  debug=False):
    print("moments_collation_illust_impl")

    rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = lrf_src.compute_rgs(debug=False)
    peaks = lrf_src.get_peaks()
    num_peaks = len(peaks)
    model = lrf_src.model
    egh_moments_list = lrf_src.get_egh_moments_list()
    m_props = to_moderate_props(props)

    x = lrf_src.xr_x
    y = lrf_src.xr_y

    egh_cy_list = []
    for params in peaks:
        cy = model(x, params)
        egh_cy_list.append(cy)

    def plot_collation_state(ax, moments_list):
        plot_collation_state_impl(x, y, egh_cy_list, egh_moments_list, ax, moments_list)

    unreliable_indeces = determine_unreliables(peak_rgs, qualities, props, trust_max_num=None)
    nur = len(unreliable_indeces)

    use_timescale = True
    num_timescales = 1 if use_timescale else 0
    params_scaler = MonoporeParamsScaler(egh_moments_list, 0, peak_rgs, unreliable_indeces, num_timescales=num_timescales)

    temp_rgs = np.asarray(peak_rgs).copy()
    egh_moments_array = np.asarray(egh_moments_list)
    if num_peaks > 1:
        eghm1_min_diff = np.min(np.abs(np.diff(egh_moments_array[:,0])))
        if debug:
            print("eghm1_min_diff=", eghm1_min_diff)
        MIN_DIFF_PENALTY_SCALE = 1e6

    ret = study_monopore_moments_impl(lrf_src,
                                        # keep_num_components=True,     # this is required for backward compatibility
                                        # trust_max_num=2,              # do not trust Rg's exceeding this number to get possibly better fit
                                        debug=False)
    if ret is None:
        return

    monopore_params, temp_rgs, unreliable_indeces, params_scaler = ret
    N, T, t0, me, mp, poresize = monopore_params[0:6]

    init_moments_list  = study_monopore_moments_impl(lrf_src, return_init_moments=True)

    apply_diff_penalty = True
    def mnp_column_flex_objective(p, return_moments=False, debug=False):
        p = params_scaler.scale_back(p)
        N_, KT_, x0_, poresize_ = p[0:4]
        if USE_K:
            T_ = KT_/N_
        else:
            T_ = KT_
        if use_timescale:
            T_ *= p[-1]
            x0_ *= p[-1]
        if nur > 0:
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

    if True:
        temp_params = np.asarray(study_monopore_moments_impl(lrf_src, return_init_params=True))
    else:
        temp_params = np.concatenate([[N, N*T, t0, poresize], temp_rgs[unreliable_indeces], [1]])
    temp_params_ = params_scaler.scale(temp_params)
    v = np.linspace(0, 9, 10)
    xx, yy = np.meshgrid(v, v)
    zz = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            KT_ = xx[i,j]
            t0_ = yy[i,j]
            temp_params_[[1,2]] = KT_, t0_
            zz[i,j] = mnp_column_flex_objective(temp_params_)
    
    with plt.Dp():
        fig = plt.figure(figsize=(18,5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132, projection="3d")
        ax3 = fig.add_subplot(133)
        fig.suptitle("Basinhopping Animation using %s" % get_in_folder(), fontsize=20)
        ax1.set_title("Animation in Elution Curves")
        plot_collation_state(ax1, init_moments_list)
        ax2.set_title("Animation in $(T,t_0,FV)$ 3D Space")
        ax2.plot_surface(xx, yy, zz)
        ax3.set_title("Animation in $(T,t_0,FV)$ Contour Map")
        ax3.contour(xx, yy, zz)
        fig.tight_layout()
        plt.show()

def basinhopping_elution_animation_impl(lrf_src,  debug=False):
    print("moments_collation_illust_impl")

    rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = lrf_src.compute_rgs(debug=False)
    peaks = lrf_src.get_peaks()
    num_peaks = len(peaks)
    model = lrf_src.model
    egh_moments_list = lrf_src.get_egh_moments_list()
    x = lrf_src.xr_x
    y = lrf_src.xr_y

    ret = study_monopore_moments_impl(lrf_src,
                                        # keep_num_components=True,     # this is required for backward compatibility
                                        # trust_max_num=2,              # do not trust Rg's exceeding this number to get possibly better fit
                                        debug=False)
    if ret is None:
        return

    monopore_params, temp_rgs, unreliable_indeces, params_scaler = ret
    N, T, t0, me, mp, poresize = monopore_params[0:6]

    rhov = temp_rgs/poresize
    rhov[rhov > 1] = 1

    def T_t0_analysis_objective(p, title=None):
        debug = title is not None
        T_ = p[0]
        t0_ = p[1]
        scales = p[2:1+num_peaks]
        cy_list = []

        for k, (rho, scale) in enumerate(zip(rhov, scales)):
            ni_ = N * (1 - rho)**me
            ti_ = T_ * (1 - rho)**mp
            cy = scale * robust_single_pore_pdf(x - t0_, ni_, ti_)
            cy_list.append(cy)
        ty = np.sum(cy_list, axis=0)
        dev = np.sum((ty - y)**2)
        return dev

    init_params = np.concatenate([[T, t0], monopore_params[6:]])

    with plt.Dp():
        fig = plt.figure(figsize=(18,5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132, projection="3d")
        ax3 = fig.add_subplot(133)
        fig.suptitle("Basinhopping Animation using %s" % get_in_folder(), fontsize=20)
        ax1.set_title("Animation in Elution Curves")
        ax1.plot(x, y, color="orange")
        plot_monopore_curves(ax1, x, monopore_params, temp_rgs)
        ax2.set_title("Animation in $(T,t_0,FV)$ 3D Space")
        ax3.set_title("Animation in $(T,t_0,FV)$ Contour Map")
        fig.tight_layout()
        plt.show()