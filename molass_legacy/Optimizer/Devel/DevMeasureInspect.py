"""
    Optimizer.Devel.DevMeasureInspect.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def dev_measure_inspect(optimizer, work_params):
    from importlib import reload
    import Distance.DeviationMeasure
    reload(Distance.DeviationMeasure)
    from Distance.DeviationMeasure import feature_deviation
    import Distance.NormalizedRmsd
    reload(Distance.NormalizedRmsd)
    from Distance.NormalizedRmsd import normalized_rmsd

    print("dev_measure_inspect")
    lrf_info1 = optimizer.objective_func(work_params, return_lrf_info=True)
    temp_params = work_params.copy()
    temp_params[2] = 0.3
    lrf_info2 = optimizer.objective_func(temp_params, return_lrf_info=True)
    
    adjust_target = 1
    uv_adjust = adjust_target - np.log10(optimizer.uv_norm1)
    xr_adjust = adjust_target - np.log10(optimizer.xr_norm1)
    print("uv_norm1=", optimizer.uv_norm1)
    print("xr_norm1=", optimizer.xr_norm1)
    print("uv_adjust=", uv_adjust)
    print("xr_adjust=", xr_adjust)

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        dev1 = feature_deviation(lrf_info1.xr_ty, lrf_info1.y, optimizer.xr_curve.max_y, optimizer.xr_norm1, debug=True, fig_info=(fig,ax1))
        dev2 = feature_deviation(lrf_info2.xr_ty, lrf_info2.y, optimizer.xr_curve.max_y, optimizer.xr_norm1, debug=True, fig_info=(fig,ax2))
        print("dev1=", dev1)
        print("dev2=", dev2)
        dev3 = normalized_rmsd(lrf_info1.uv_ty, lrf_info1.y, adjust=uv_adjust)
        dev4 = normalized_rmsd(lrf_info1.xr_ty, lrf_info1.y, adjust=xr_adjust)
        print("dev3=", dev3)
        print("dev4=", dev4)
        fig.tight_layout()
        plt.show()