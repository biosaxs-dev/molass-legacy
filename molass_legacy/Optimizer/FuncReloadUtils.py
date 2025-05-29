"""
    Optimizer.FuncReloadUtils.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
from .FuncImporter import import_objective_function

def reload_optimizer(old_optimizer, prepare=True, init_params=None):
    func_code = old_optimizer.get_name()
    func_class = import_objective_function(func_code)

    optimizer = func_class(
                old_optimizer.dsets,
                old_optimizer.n_components,
                uv_base_curve=old_optimizer.uv_base_curve,
                xr_base_curve=old_optimizer.xr_base_curve,
                qvector=old_optimizer.qvector,    # trimmed sd
                wvector=old_optimizer.wvector,
                )

    if old_optimizer.is_stochastic():
        optimizer.params_type.set_estimator(old_optimizer.params_type.estimator)

    if prepare:
        if init_params is None:
            init_params = old_optimizer.init_params
        optimizer.prepare_for_optimization(init_params)

    return optimizer