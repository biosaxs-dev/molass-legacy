"""
    Models.Stochastic.ColumnWidthAnalysis.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from datetime import datetime
from importlib import reload

BH_NITER = 100

def estimate_column_parameters_separately(moment_rg_file, column_type, debug=False):
    from molass_legacy.Batch.LiteBatch import LiteBatch
    from molass_legacy.Models.Stochastic.DatablockUtils import load_datablock_list, get_block_lrf_src
    from molass_legacy.Models.Stochastic.MomentUtils import compute_egh_moments
    from molass_legacy.Models.Stochastic.DispersiveMonopore import guess_params_using_moments

    in_folders, block_list = load_datablock_list(moment_rg_file, column_type)
    batch = LiteBatch()
    params_list = []
    for in_folder, block in zip(in_folders, block_list):
        lrf_src = get_block_lrf_src(batch, in_folder)
        rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = lrf_src.compute_rgs(debug=False)
        peaks = lrf_src.xr_peaks[indeces,:]
        egh_moments_list = compute_egh_moments(peaks)
        ret = guess_params_using_moments(lrf_src.xr_x, lrf_src.xr_y, egh_moments_list, peak_rgs, qualities, props, debug=debug)
        if ret is None:
            return
        sdm_params, temp_rgs = ret
        colparams = np.concatenate([sdm_params, temp_rgs])
        params_list.append(colparams)
    return params_list

def estimate_column_parameters(moment_rg_file, column_type, params_list=None, rg_curves=None, debug=False):
    from molass_legacy.KekLib.BasicUtils import Struct
    from molass_legacy.Models.Stochastic.DatablockUtils import load_datablock_list
    from molass_legacy.Models.Stochastic.MomentUtils import compute_egh_moments, to_moderate_props
    from molass_legacy.Models.Stochastic.RgReliability import determine_unreliables
    from molass_legacy.Models.Stochastic.ParamLimits import (USE_K, M1_WEIGHT, M2_WEIGHT,
                                               MNP_BOUNDS, PORESIZE_BOUNDS, N0_BOUND, TIMESCALE_BOUND,
                                               BASINHOPPING_SCALE, N0_INIT, ParamsScaler)
    if debug:
        import Models.Stochastic.OligoporeUtils as module
        reload(module)
    from molass_legacy.Models.Stochastic.OligoporeUtils import guess_oligopore_colparams

    print('estimate_oligopore_distribution')

    assert USE_K

    T0_BOUND = MNP_BOUNDS[2].copy()

    class MultidatasetParamsScaler(ParamsScaler):
        def __init__(self, num_experiments, num_timescales=0):
            self.params_list = params_list
            MNP_BOUNDS_ = MNP_BOUNDS.copy()
    
            bounds_list = (MNP_BOUNDS_[0:2]
                            + [PORESIZE_BOUNDS]          # allowing the upper bound to reach MAX_PORESIZE can cause problems? 
                            + [N0_BOUND]
                            + [T0_BOUND] * num_experiments
                            + [T0_BOUND] * num_experiments
                            + [TIMESCALE_BOUND]*num_timescales)

            self.bounds = np.array(bounds_list)
            self.set_scales()

    in_folders, block_list = load_datablock_list(moment_rg_file, column_type)

    use_basinhopping = True
    use_timescales = True
    num_blocks = len(in_folders)

    sep1 = 4
    sep2 = sep1 + num_blocks
    sep3 = sep2 + num_blocks
    def split(p):
        N, K, poresize, N0 = p[0:sep1]
        x0v = p[sep1:sep2]
        tIv = p[sep2:sep3]
        if use_timescales:
            tscales = p[sep3:]
        else:
            tscales = np.array([])
        return N, K/N, poresize, N0, x0v, tIv, tscales

    num_timescales = num_blocks if use_timescales else 0
    scaler = MultidatasetParamsScaler(num_blocks, num_timescales=num_timescales)

    me = 1.5
    mp = 1.5

    def multi_dataset_objective(p):
        if use_basinhopping:
            p = scaler.scale_back(p)
        N, T, poresize, N0, x0v, tIv, tscales = split(p)

        t0_upper_penalty = 0
        dev = 0
        for k, (block, x0, tI) in enumerate(zip(block_list, x0v, tIv)):
            if use_timescales:
                T_ = T*tscales[k]
                x0_ = x0*tscales[k]
            else:
                T_ = T
                x0_ = x0
            for M1, M2, rg, quality in block:
                t0_upper_penalty += min(0, M1 - np.sqrt(M2) - x0_)**2    # i.e., M1 - np.sqrt(M2) > t0 must always hold
                rho = min(1, rg/poresize)
                M1_ = x0_ + N * T_ * (1 - rho)**(me + mp)
                M2_ = 2 * N * T_**2 * (1 - rho)**(me + 2*mp) + (M1_ - tI)**2/N0
                dev += quality * np.log(M1_WEIGHT*(M1_ - M1)**2 + M2_WEIGHT*(M2_ - M2)**2)
                # dev += np.log(M1_WEIGHT*(M1_ - M1)**2 + M2_WEIGHT*(M2_ - M2)**2)

        return dev + t0_upper_penalty*1e6

    if use_timescales:
        if column_type == '1':
            timescales = [1, 1, 1, 1, 1]
        else:
            timescales = [1, 1, 1, 0.6, 0.6]
    else:
        timescales = []

    if params_list is None:
        init_x0v = []
        for i, block in enumerate(block_list):
            x0_init = np.min([M[0] - np.sqrt(M[1])*20 for M in block[:,1:3]])
            init_x0v.append(x0_init)

        t0_upper = np.max(block[:,1] - np.sqrt(block[:,2]))     # 

        print("init_t0v=", init_t0v)
        init_params = np.concatenate([[1000, 1000], [60, 200], [0.5], init_t0v, timescales])
    else:
        params_array = np.array(params_list)
        # print("params_array=", params_array)
        init_t0v = params_array[:,2]
        weighted_array = compute_weighted_array(params_array)
        weighted_avarage = np.average(weighted_array, axis=0)
        N, K = weighted_avarage[0:2]
        init_pszv = weighted_avarage[3:3+num_pszs]
        init_pszp = np.average(params_array[:,3+num_pszs:3+2*num_pszs-1], axis=0)
        print("weighted_avarage=", weighted_avarage)
        print("init_pszv=", init_pszv)
        print("init_pszp=", init_pszp)
        print("init_t0v=", init_t0v)
        init_params = np.concatenate([[N, K], init_pszv, init_pszp, init_t0v, timescales])
        t0_upper = np.max(init_t0v)

    if use_basinhopping:
        print("init_params=", init_params)
        print("bounds=", bounds)
        bounds_ = [(0, BASINHOPPING_SCALE)] * len(init_params)
        minimizer_kwargs = dict(method='Nelder-Mead', bounds=bounds_)
        minima_counter = 0
        def minima_callback(x, f, accept):
            nonlocal minima_counter
            if minima_counter % 10 == 0:
                print("%s: minima_counter=%d" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), minima_counter))
            minima_counter += 1

        init_params_ = scaler.scale(init_params)

        res = basinhopping(multi_dataset_objective, init_params_, niter=BH_NITER,
                           minimizer_kwargs=minimizer_kwargs, callback=minima_callback)
        res.x = scaler.scale_back(res.x)
    else:
        res = minimize(multi_dataset_objective, init_params, method='Nelder-Mead', bounds=bounds)
    print('res.x=', res.x)

    N, T, pszv, pszp, t0v, tscales = split(res.x)

    import molass_legacy.KekLib.DebugPlot as plt

    extra_button_specs=[("Plot Results", lambda : plot_results(N, T, pszv, pszp, t0v, tscales, in_folders, block_list, params_list, rg_curves=rg_curves)),]
    with plt.Dp(button_spec=["OK", "Cancel"], extra_button_specs=extra_button_specs):
        fig, ax = plt.subplots()
        plt.show()

def rounded_list(fmt, values):
    return ', '.join([fmt % v for v in values])

def plot_results(N, T, pszv, pszp, t0v, tscales, in_folders, block_list, params_list, rg_curves=None):
    import Models.Stochastic.OligoporePlotImpl as module
    reload(module)
    from molass_legacy.Models.Stochastic.OligoporePlotImpl import plot_results_impl
    try:
        plot_results_impl(N, T, pszv, pszp, t0v, tscales, in_folders, block_list, params_list, rg_curves=rg_curves)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(None, "plot_results_impl failure: ")

def load_csv_values(path):
    params_list = []
    with open(path) as fh:
        for line in fh:
            values = [float(v) for v in line[:-1].split(',')]
            params_list.append(values)
    return params_list

def compute_weighted_array(params_array):
    pv = params_array[:,-2]
    qv = params_array[:,-1]
    target_array = params_array[:,:-2].T
    return (target_array*pv + target_array*qv).T

def test_weighted_array():
    from molass_legacy.SerialAnalyzer.DataUtils import get_local_path
    TODO = get_local_path('TODO')
    params_array = np.array(load_csv_values(TODO + r'\20240425\params_list.csv'))
    print("params_array=", params_array)
    print("weighted_average=", compute_weighted_array(params_array))

if __name__ == "__main__":
    import sys
    sys.path.append("../lib")
    import os
    import seaborn as sns
    sns.set_theme()
    from molass_legacy.KekLib.ChangeableLogger import Logger
    from molass_legacy.KekLib.BasicUtils import clear_dirs_with_retry
    from molass_legacy.SerialAnalyzer.DataUtils import get_local_path
    from molass_legacy.Models.Stochastic.DatablockUtils import get_rg_curves

    TODO = get_local_path('TODO')
    column_type = '2'

    # test_weighted_array()
    # exit()

    # os.makedirs('temp', exist_ok=True)
    clear_dirs_with_retry(['temp'])
    logger = Logger("temp/psd-estimate.log")

    moment_rg_file = TODO + r'\20240425\rgs.csv'
    if True:    
        params_list = estimate_column_parameters_separately(moment_rg_file, column_type, debug=True)

        with open('temp/params_list-%s.csv' % column_type, 'w') as fh:
            for params in params_list:
                fh.write(','.join(['%g' % v for v in params]) + '\n')
    else:
        rg_folders_root = TODO + '\\20240523\\rg_folders'
        rg_curves = get_rg_curves(rg_folders_root, column_type)
        params_list = load_csv_values(TODO + r'\20240614\params_list-%s.csv' % column_type)
        estimate_column_parameters(moment_rg_file, column_type, params_list=params_list, rg_curves=rg_curves)