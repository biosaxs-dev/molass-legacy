"""
    Models.Stochastic.PairedAnalysisImpl.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from importlib import reload
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.Stochastic.DispersiveUtils import NUM_SDMCUV_PARAMS

def paired_analysis_impl(names, num_components, info_list, rg_info_list, unreliable_indeces,
                         sdm_params_info_list, mapping_list, mapping, moments_list, exec_spec
                         ):
    import SSDC.SsdcAnalysisUtils
    reload(SSDC.SsdcAnalysisUtils)
    from SSDC.SsdcAnalysisUtils import plot_ssdc_result
    print("paired_analysis_impl")

    for k, (sdm_params, temp_rgs, temp_bouns) in enumerate(sdm_params_info_list):
        print([k], sdm_params)
        print([k], temp_rgs)
        print([k], temp_bouns)
        N, K, x0, poresize, N0, tI = sdm_params[0:NUM_SDMCUV_PARAMS]
        print([k], poresize, N0, tI)

    thick_params, thick_rgs, thick_bounds = sdm_params_info_list[0]
    thick_N, thick_K, thick_x0, thick_poresize, thick_N0, thick_tI = thick_params[0:NUM_SDMCUV_PARAMS]
    thick_T = thick_K/thick_N

    thin_params, thin_rgs, thin_bounds = sdm_params_info_list[1]
    thin_N, thin_K, thin_x0, thin_poresize, thin_N0, thin_tI = thin_params[0:NUM_SDMCUV_PARAMS]
    thin_T = thin_K/thin_N


    init_rgs = np.average([thick_rgs, thin_rgs], axis=0)
    temp_rgs = init_rgs.copy()
    print("temp_rgs=", temp_rgs)
    print("mapping=", mapping)
    a, b = mapping

    me = 1.5
    mp = 1.5

    thick_M1_tv = [M[0] for M in moments_list[0]]
    thick_M2_tv = [np.sqrt(M[1]) for M in moments_list[0]]
    thin_M1_tv  = [M[0] for M in moments_list[1]]
    thin_M2_tv  = [np.sqrt(M[1]) for M in moments_list[1]]

    minimal = False
    use_basinhoppng = False
    if minimal:
        unreliable_indeces = np.array([], dtype=int)
    else:
        unreliable_indeces = np.array(unreliable_indeces, dtype=int)

    def paired_objective(p):
        poresize = p[0]
        if minimal:
            thick_tI_, thin_N0_ = p[1:3]
            thick_N_ = thick_N
            thick_T_ = thick_T
            thin_N_ = thin_N
            thin_T_ = thin_T
            thick_x0_ = thick_x0
            thin_x0_ = thin_x0
        else:
            thick_N_, thick_K_, thick_x0_, thick_tI_ = p[1:5]
            thick_T_ = thick_K_/thick_N_
            thin_N_, thin_K_, thin_x0_, thin_N0_ = p[5:9]
            thin_T_ = thin_K_/thin_N_
        temp_rgs[unreliable_indeces] = p[9:]

        rhov = temp_rgs/poresize
        rhov[rhov > 1] = 1
        thin_tI_ = thick_tI_*a + b
        thick_M1v = thick_x0_ + thick_N_*thick_T_*(1 - rhov)**(me+mp)
        thin_M1v  =  thin_x0_ +  thin_N_* thin_T_*(1 - rhov)**(me+mp)
        thick_M2v = np.sqrt(2*thick_N_*thick_T_**2*(1 - rhov)**(me+2*mp) + (thick_M1v - thick_tI_)**2/thick_N0)
        thin_M2v  = np.sqrt(2* thin_N_* thin_T_**2*(1 - rhov)**(me+2*mp) + (thin_M1v  - thin_tI_ )**2/ thin_N0_)
        return (  np.log(np.sum((thick_M1v - thick_M1_tv)**2)) + np.log(np.sum((thick_M2v - thick_M2_tv)**2))
                + np.log(np.sum(( thin_M1v -  thin_M1_tv)**2)) + np.log(np.sum(( thin_M2v -  thin_M2_tv)**2))
                )
    init_poresize = (thick_poresize + thin_poresize)/2
    poresize_bounds = exec_spec["poresize_bounds"]
    if minimal:
        init_params = np.array([init_poresize, thick_tI, thin_N0])
        bounds = [poresize_bounds, (-3000, 0), (3600, thick_N0)]
    else:
        init_params = np.concatenate([[init_poresize,
                                    thick_N, thick_K, thick_x0, thick_tI,
                                     thin_N,  thin_K,  thin_x0, thin_N0],
                                    init_rgs[unreliable_indeces]])
        print("thick_bounds[4]=", thick_bounds[4])
        bounds = [poresize_bounds,
                (300, 3000), (300, 2000), (thick_x0 - 100, thick_x0 + 100),
                thick_bounds[4],    # better widen?
                (300, 3000), (300, 2000), (thin_x0 - 100, thin_x0 + 100), (900, thick_N0)] + [(10, 100)]*len(unreliable_indeces)

    if use_basinhoppng:
        minimizer_kwargs = dict(method='Nelder-Mead', bounds=bounds)
        res = basinhopping(paired_objective, init_params, minimizer_kwargs=minimizer_kwargs)        
    else:
        res = minimize(paired_objective, init_params, method='Nelder-Mead', bounds=bounds)

    p = res.x
    poresize = p[0]
    if minimal:
        thick_tI, thin_N0 = p[1:3]
    else:
        thick_N, thick_K, thick_x0, thick_tI = p[1:5]
        thin_N, thin_K, thin_x0, thin_N0 = p[5:9]
        # temp_rgs[unreliable_indeces] = p[9:]
    thin_tI = thick_tI*a + b
    opt_thick_params = np.concatenate([[thick_N, thick_K, thick_x0, poresize, thick_N0, thick_tI], thick_params[NUM_SDMCUV_PARAMS:]])
    opt_thin_params  = np.concatenate([[thin_N,  thin_K,  thin_x0,  poresize,  thin_N0,  thin_tI],  thin_params[NUM_SDMCUV_PARAMS:]])
    print(thick_params)
    print(opt_thick_params)
    print(thin_params)
    print(opt_thin_params)
    opt_sdm_params_info_list = [(opt_thick_params, temp_rgs), (opt_thin_params, temp_rgs)]
    plot_ssdc_result(names, info_list, rg_info_list, unreliable_indeces,
                     opt_sdm_params_info_list, mapping_list, mapping)