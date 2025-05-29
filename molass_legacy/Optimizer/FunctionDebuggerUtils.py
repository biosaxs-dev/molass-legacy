"""
    Optimizer.FunctionDebuggerUtils.py

    Copyright (c) 2023-2025, SAXS Team, KEK-PF
"""
from molass_legacy.Baseline.BaselineUtils import create_xr_baseline_object
from .FuncImporter import import_objective_function

def create_optimizer_for_debug(dsets, n_components, optinit_info, init_params, composite=None, prepare=True):
    print("create_optimizer_for_debug: prepare=", prepare)

    # task-remark-begin unify the coding below with Optimizer.JobStateInfo.py
    uv_base_curve = optinit_info.treat.get_base_curve()
    class_code = optinit_info.class_code
    fullopt_class = import_objective_function(class_code)
    sd = optinit_info.sd
    xr_base_curve = create_xr_baseline_object()
    new_optimizer = fullopt_class(dsets, n_components,
                                uv_base_curve=uv_base_curve,
                                xr_base_curve=xr_base_curve,
                                qvector=sd.qvector,
                                wvector=sd.lvector,
                                composite=composite,
                                )
    if prepare:
        new_optimizer.prepare_for_optimization(init_params)
    # task-remark-end

    return new_optimizer
