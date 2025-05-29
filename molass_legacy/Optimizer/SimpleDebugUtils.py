"""
    Optimizer.SimpleDebugUtils.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
from .FuncImporter import import_objective_function

def copycreate_optimizer(optimizer):
    class_code = optimizer.__class__.__name__
    optimizer_class = import_objective_function(class_code)
    new_optimizer = optimizer_class(
        optimizer.dsets,
        optimizer.n_components,
        uv_base_curve=optimizer.uv_base_curve,
        xr_base_curve=optimizer.xr_base_curve,
        qvector=optimizer.qvector,
        wvector=optimizer.wvector,
        composite=optimizer.composite,
        )
    return new_optimizer

def debug_optimizer(optimizer, debug_params=None, plot=False):
    new_optimizer = copycreate_optimizer(optimizer)
    if debug_params is None:
        debug_params = optimizer.init_params
    new_optimizer.prepare_for_optimization(debug_params)
    new_optimizer.objective_func(debug_params, plot=plot)