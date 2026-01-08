"""
    Optimizer.OptimizerMain.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""
import os
import numpy as np
from molass_legacy.Baseline.BaselineUtils import create_xr_baseline_object
from .FuncImporter import import_objective_function

def create_optimizer_from_job(in_folder=None, n_components=None, class_code=None, trimming_txt=None, shared_memory=None,
                              need_settings=False):
    if need_settings:
        pass

    from molass.Bridge.OptimizerInput import OptimizerInput
    fullopt_input = OptimizerInput(in_folder=in_folder, trimming_txt=trimming_txt, legacy=True)
    dsets = fullopt_input.get_dsets()
    x_shifts_file = trimming_txt.replace('trimming.txt', 'x_shifts.txt')
    if os.path.exists(x_shifts_file):
        x_shifts = np.loadtxt(x_shifts_file, dtype=int)
        dsets.apply_x_shifts(x_shifts)
    fullopt_class = import_objective_function(class_code)
    uv_base_curve = fullopt_input.get_base_curve()      # uv_base_curve comes from FullOptInput.get_sd_from_folder()
    xr_base_curve = create_xr_baseline_object()
    qvector, wvector = fullopt_input.get_spectral_vectors()
    optimizer = fullopt_class(dsets, n_components,
                uv_base_curve=uv_base_curve,
                xr_base_curve=xr_base_curve,
                qvector=qvector,
                wvector=wvector,
                shared_memory=shared_memory)
    return optimizer

def optimizer_main(in_folder, trimming_txt=None, n_components=3,
                   solver=None,
                   init_params=None, real_bounds=None,
                   drift_type=None, niter=100, seed=None,
                   callback=True, class_code='F0000', shared_memory=None,
                   nnn=0,
                   legacy=True,
                   xr_only=False,
                   debug=True):

    optimizer = create_optimizer_from_job(in_folder=in_folder,
                                          n_components=n_components,
                                          class_code=class_code,
                                          trimming_txt=trimming_txt,
                                          shared_memory=shared_memory)
    optimizer.set_xr_only(xr_only)

    if seed is None:
        seed = np.random.randint(100000, 999999)
    strategy = optimizer.get_strategy()
    if strategy.trust_initial_baseline():
        baseline_fixed = True
        from molass_legacy.Optimizer.FixedBaselineOptimizer import FixedBaselineOptimizer
        fb_optimizer = FixedBaselineOptimizer(optimizer)
        result = fb_optimizer.solve(init_params, real_bounds=real_bounds, niter=niter, seed=seed, debug=debug)
    else:
        baseline_fixed = False
        if strategy.baseline_first():
            if nnn == 0:
                from molass_legacy.Optimizer.BaselineOptimizer import BaselineOptimizer
                baseline_optimizer = BaselineOptimizer(optimizer)
                baseline_indeces = optimizer.get_baseline_indeces()
                result = baseline_optimizer.solve(init_params, baseline_indeces)
            else:
                baseline_fixed = True
                result = optimizer.solve(init_params, real_bounds=real_bounds, niter=niter, seed=seed, callback=callback, method=solver,
                                        baseline_fixed=baseline_fixed, debug=debug)
        else:
            if strategy.is_strategic(nnn):
                from molass_legacy.Optimizer.StrategicOptimizer import StrategicOptimizer
                temp_params = init_params
                indeces_list = strategy.get_indeces_list(nnn)
                assert len(indeces_list) == 1   # for now
                open_mode = "w"                 # "callback.txt" open mode
                for indeces in indeces_list:
                    strategic_optimizer = StrategicOptimizer(optimizer, indeces)
                    # task: add method option
                    result = strategic_optimizer.solve(temp_params, real_bounds=real_bounds, niter=niter, seed=seed, open_mode=open_mode, debug=debug)
                    temp_params = result.x
                    open_mode = "a"
            else:
                result = optimizer.solve(init_params, real_bounds=real_bounds, niter=niter, seed=seed, callback=callback, method=solver, debug=debug)
    fig_info = [in_folder, None, result]
    if debug:
        optimizer.objective_func(result.x, plot=True, fig_info=fig_info)
