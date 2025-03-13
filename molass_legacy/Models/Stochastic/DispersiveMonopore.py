"""
    Models.Stochastic.DispersiveMonopore.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from molass_legacy.Models.Stochastic.ParamLimits import USE_K, MAX_PORESIZE
from molass_legacy.Models.Stochastic.RgReliability import determine_unreliables
from molass_legacy.Models.Stochastic.DispersivePdf import N0

M1_WEIGHT = 6
M2_WEIGHT = 2
M3_WEIGHT = 2

assert USE_K

def guess_params_using_moments(x, y, egh_moments_list, peak_rgs, qualities, props,
                               fronting=False,
                               secon_trial=False,
                               trust_max_num=None,
                               exec_spec=None,
                               avoid_vanishing=True, debug=False):
    from molass_legacy.Models.Stochastic.ParamLimits import BASINHOPPING_SCALE
    if debug:
        from importlib import reload    
        import Models.Stochastic.DispersiveLimits
        reload(Models.Stochastic.DispersiveLimits)
        import Models.Stochastic.DispersiveUtils
        reload(Models.Stochastic.DispersiveUtils)
        import Models.Stochastic.MobileDispersion
        reload(Models.Stochastic.MobileDispersion)
    from molass_legacy.Models.Stochastic.DispersiveLimits import DispersiveParamsScaler
    from molass_legacy.Models.Stochastic.DispersiveUtils import NUM_SDMCOL_PARAMS, NUM_SDMCUV_PARAMS, compute_elution_curves, investigate_sdm_params
    from molass_legacy.Models.Stochastic.MobileDispersion import DispersionRatio

    PENALTY_SCALE = 1e3

    me = 1.5
    mp = 1.5

    num_components = len(peak_rgs)
    in_folder = get_in_folder()
    poresize_bounds = None
    unreliable_indeces = None

    if exec_spec is None:
        if in_folder.find('data01') > 0:
            unreliable_indeces = [1,2]
            poresize_bounds = 80, 120
            N0 = 10000
        elif in_folder.find('data02') > 0:
            unreliable_indeces = [1,2]
            poresize_bounds = 80, 120
            N0 = 5000
        elif in_folder.find('BSA_201') > 0:
            unreliable_indeces = [0]
            poresize_bounds = 75, 80
            # N0 = 50000      # best with tI-bound - 2000
            N0 = 50000
        elif in_folder.find('BSA_202') > 0:
            unreliable_indeces = [0]
            poresize_bounds = 75, 80
            # N0 = 14400/4
            N0 = 7000       # best with N0 = 50000, tI-bound - 2000
        elif in_folder.find('FER_OA_301') > 0:
            unreliable_indeces = [0,2]
            poresize_bounds = 75, 80
            # N0 = 50000      # best with tI-bound - 2000
            N0 = 50000
        elif in_folder.find('FER_OA_302') > 0:
            unreliable_indeces = [0,2]
            poresize_bounds = 75, 80
            # N0 = 14400/4
            N0 = 7000       # best with N0 = 50000, tI-bound - 2000
        elif in_folder.find('ALD_OA001') > 0:
            unreliable_indeces = [1]
            poresize_bounds = 75, 80
            N0 = 50000
        elif in_folder.find('ALD_OA002') > 0:
            unreliable_indeces = [1]
            poresize_bounds = 75, 80
            N0 = 50000
        else:
            unreliable_indeces = []
            poresize_bounds = 75, 80
            if fronting:
                # as in 20230303/HasA
                N0 = 30000
            else:
                N0 = 50000
    else:
        unreliable_indeces = exec_spec['unreliable_indeces']
        poresize_bounds = exec_spec['poresize_bounds']
        # N0 = exec_spec['init_N0']
        if fronting:
            # as in 20230303/HasA
            poresize_bounds = 75, 300
            N0 = 30000
        else:
            N0 = 50000

    if unreliable_indeces is None:
        unreliable_indeces = determine_unreliables(peak_rgs, qualities, props, trust_max_num=trust_max_num, debug=True)

    print("unreliable_indeces=", unreliable_indeces)
    num_unreliables = len(unreliable_indeces)
    use_basinhopping = True         # it seems that global optimization is required for this problem
    params_scaler = DispersiveParamsScaler(egh_moments_list, 0, peak_rgs, unreliable_indeces,
                                           poresize_bounds=poresize_bounds,
                                           fronting=fronting,
                                           )
    # 
    # bounds = [(300, 2000), (300, 5000), (x0-500, x0+500), (0, MAX_PORESIZE), (10000, 1e6), (-500, x0)]
    bounds = params_scaler.get_bounds()
    N = np.average(bounds[0])
    NT = np.average(bounds[1])
    # x0 = np.min([M[0] - np.sqrt(M[1])*10 for M in egh_moments_list])
    x0 = np.average(bounds[2])
    poresize = np.average(bounds[3])
    sigma0 = np.sqrt(np.percentile([M[1] for M in egh_moments_list], 30))
    t0 = min(500, np.sqrt(N0)*sigma0)
    if fronting:
        t0 = 0
    print("N0=", N0, "t0=", t0)
    tI = x0 - t0
    init_colparams = np.concatenate([[N, NT, x0, poresize, tI], peak_rgs[unreliable_indeces]])
    target_moments_list = [(M[0], np.sqrt(M[1]), np.sign(M[2])*np.power(abs(M[2]), 1/3)) for M in egh_moments_list]
    main_third_moment = target_moments_list[np.argmax(props)][2]
    print("main_third_moment=", main_third_moment)
    zeros_rg_diff = np.zeros(num_components-1)
    add_third_moment = True

    def optimize_moments(init_colparams, use_elution=False):
        N, NT, x0, poresize, tI = init_colparams[0:NUM_SDMCOL_PARAMS]

        # dratio_spline = DispersionRatio(10, poresize*0.8, N, NT/N, poresize, t0, N0)

        print("init_colparams=", init_colparams)

        temp_rgs = np.asarray(peak_rgs).copy()
        def moments_objective(p, debug=False, title=None):
            if use_basinhopping:
                p = params_scaler.scale_back(p)
            N_, K_, x0_, poresize_, tI_ = p[0:NUM_SDMCOL_PARAMS]
            p_rgs = p[NUM_SDMCOL_PARAMS:]
            temp_rgs[unreliable_indeces] = p_rgs
            penalty = PENALTY_SCALE * np.sum(np.min([zeros_rg_diff, temp_rgs[:-1] - temp_rgs[1:]], axis=0)**2)
            T_ = K_/N_
            rhov = temp_rgs/poresize_
            rhov[rhov > 1] = 1
            dev_list = []
            if debug:
                moments_list = []
            for k, (M, rho) in enumerate(zip(target_moments_list, rhov)):
                ni = N_ * (1 - rho)**me
                ti = T_ * (1 - rho)**mp
                M1_ = x0_ + ni*ti
                M2_ = np.sqrt(2 * ni * ti**2 + (M1_ - tI_)**2/N0)
                moments_dev = M1_WEIGHT*(M1_ - M[0])**2 + M2_WEIGHT*(M2_- M[1])**2
                if add_third_moment:
                    M3_ = 6*ni*ti**2*(N0*ti + ni*ti + x0_)/N0
                    M3_ = np.sign(M3_)*np.power(abs(M3_), 1/3)
                    moments_dev += M3_WEIGHT*(M3_ - M[2])**2
                dev_list.append(moments_dev)
                if debug:
                    print([k], "M1, M1_ = %.4g, %.4g" % (M[0], M1_))
                    print([k], "M2, M2_ = %.3g, %.3g" % (M[1], M2_))
                    moments_list.append([M1_, M2_])

            if debug:
                from molass_legacy.Models.Stochastic.DispersivePdf import dispersive_monopore_pdf
                print("moments_list=", moments_list)
                print("dev_list=", dev_list)
                print("p=", p)
                print("N_, T_, N0, tI_=", N_, T_, N0, tI_)
                with plt.Dp():
                    fig, ax = plt.subplots()
                    if title is not None:
                        ax.set_title(title)
                    ax.plot(x, y)
                    def plot_moments(ax, moments, color=None):
                        m1 = moments[0]
                        ax.axvline(x=m1, color=color)
                        s = moments[1]
                        ax.axvspan(m1-s, m1+s, color=color, alpha=0.3)

                    for moments, egh_moments in zip(moments_list, target_moments_list):
                        plot_moments(ax, moments, color='yellow')
                        plot_moments(ax, egh_moments, color='green')
                    ax.axvline(x=x0_, color='red')
                    scales = props * np.sum(y)
                    x_ = x - tI_
                    print("x_[0]=", x_[0])
                    t0_ = x0_ - tI_
                    for k, (rho, scale) in enumerate(zip(rhov, scales)):
                        ni_ = N_ * (1 - rho)**me
                        ti_ = T_ * (1 - rho)**mp
                        print([k], ni_, ti_, N0, t0_)
                        cy = scale*dispersive_monopore_pdf(x_, ni_, ti_, N0, t0_)
                        ax.plot(x, cy, ":")
 
                    rv = np.linspace(20, poresize, 100)
                    rhov_ = rv/poresize_
                    rhov_[rhov_ > 1] = 1
                    trv = x0_ + N_ * T_ * (1 - rhov_)**(me + mp)

                    axt = ax.twinx()
                    axt.grid(False)
                    axt.plot(trv, rv, color='yellow')
                    ehg_m1 = [M[0] for M in egh_moments_list]
                    axt.plot(ehg_m1, temp_rgs, 'o')

                    fig.tight_layout()
                    ret = plt.show()
                    if not ret:
                        if title is None:
                            assert False
                        return

            return np.sum(np.asarray(dev_list)*props) + penalty

        if use_basinhopping:
            init_colparams = params_scaler.scale(init_colparams)
        if debug:
            print("init_colparams=", init_colparams)
            ret = moments_objective(init_colparams, debug=True, title="before moments minimize")
            if ret is None:
                return

        if use_basinhopping:
            bounds_ = [(0, BASINHOPPING_SCALE)] * len(init_colparams)
            minimizer_kwargs = dict(method='Nelder-Mead', bounds=bounds_)
            res = basinhopping(moments_objective, init_colparams, minimizer_kwargs=minimizer_kwargs)
        else:
            res = minimize(moments_objective, init_colparams, bounds=bounds, method='Nelder-Mead')

        if debug:
            print("res.x=", res.x)
            ret = moments_objective(res.x, debug=True, title="after moments minimize")
            if ret is None:
                return
        if use_basinhopping:    
            res.x = params_scaler.scale_back(res.x)
        temp_rgs[unreliable_indeces] = res.x[NUM_SDMCOL_PARAMS:]

        init_scales = props * np.sum(y)
        # init_cuvparams = dratio_spline.covert_to_curveparams(peak_rgs, props, target_moments_list, res.x, init_scales, debug_info=(x, y))
        init_cuvparams = np.concatenate([res.x[:NUM_SDMCOL_PARAMS-1], [N0, res.x[NUM_SDMCOL_PARAMS-1]], init_scales])
        if init_cuvparams is None:
            return
        print("init_cuvparams=", init_cuvparams)

        if debug:
            ret = investigate_sdm_params(x, y, init_cuvparams, num_unreliables, temp_rgs, bounds)
            if not ret:
                return

        return res.x, temp_rgs, bounds, init_cuvparams

    # first trial
    temp_colparams = init_colparams
    for k in range(1):
        print([k], "temp_colparams=", temp_colparams)
        opt_info = optimize_moments(temp_colparams)
        if opt_info is None:
            return
        temp_colparams, temp_rgs, bounds, init_cuvparams = opt_info

    # second trial
    if secon_trial:
        use_basinhopping = False
        opt_info = optimize_moments(temp_colparams, use_elution=True)
        if opt_info is None:
                return
        temp_colparams, temp_rgs, bounds, init_cuvparams = opt_info
        
    cuvparams = init_cuvparams[0:NUM_SDMCUV_PARAMS]

    def elution_objective(p, debug=False, title=None):
        # p_rgs = p[NUM_SDMCOL_PARAMS:NUM_SDMCOL_PARAMS+num_unreliables]
        # temp_rgs[unreliable_indeces] = p_rgs
        cy_list, ty = compute_elution_curves(x, np.concatenate([cuvparams, p]), temp_rgs)

        if debug:
            x0_ = cuvparams[2]
            with plt.Dp():
                fig, ax = plt.subplots()
                if title is not None:
                    ax.set_title(title)
                ax.plot(x, y)
                for cy in cy_list:
                    ax.plot(x, cy, ":")
                ax.plot(x, ty, ":", color="red")
                ax.axvline(x=x0_, color='red')
                # ax.axvline(x=tI_, color='gray')
                fig.tight_layout()
                ret = plt.show()
                if not ret:
                    return

        return np.sum((y - ty)**2)

    # init_scales = props * np.sum(y)
    init_scales = init_cuvparams[-num_components:]
    if debug:
        print("init_scales=", init_scales)
        ret = elution_objective(init_scales, debug=True, title="before elution minimize")
        if ret is None:
            return

    if avoid_vanishing:
        min_scale = np.min(init_scales)*0.5
    else:
        min_scale = 0
    scale_bounds = [(min_scale, 10)]*len(init_scales)
    res = minimize(elution_objective, init_scales, bounds=scale_bounds, method='Nelder-Mead')
    
    if debug:
        print("res.x=", res.x)
        ret = elution_objective(res.x, debug=True, title="after elution minimize")
        if ret is None:
            return
    
    ret_params = np.concatenate([cuvparams, res.x])
    return ret_params, temp_rgs, bounds