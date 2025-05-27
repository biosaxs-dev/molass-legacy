"""
    Optimizer.Devel.SimpleInspect.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Optimizer.BasicOptimizer import UV_XR_RATIO_ALLOW

def simple_inspect_impl(optimizer, work_params):
    from importlib import reload
    import Optimizer.XrUvScaleRatio
    reload(Optimizer.XrUvScaleRatio)
    from molass_legacy.Optimizer.XrUvScaleRatio import xruv_scale_ratio_penalty
    print("simple_inspect_impl")
    xr_params, xr_baseparams, rg_params, (a, b), uv_params, uv_baseparams, (c, d), sdmcol_params = optimizer.split_params_simple(work_params)

    print("real_bounds=", optimizer.real_bounds)

    uv_xr_ratio = uv_params/xr_params
    average = np.average(uv_xr_ratio)
    std = np.std(uv_xr_ratio)
    ratio_deviation = max(0, std/average - UV_XR_RATIO_ALLOW)**2
    print("np.std(uv_xr_ratio) = ", std)
    print("np.average(uv_xr_ratio) = ", average)
    print("ratio_deviation = ", ratio_deviation)
    print("penalty = ", xruv_scale_ratio_penalty(xr_params, uv_params))

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.plot(uv_xr_ratio, "o")
        ax.axhline(np.average(uv_xr_ratio), color="red")
        ax.axhspan(average - std, average + std, color="red", alpha=0.2)
        fig.tight_layout()
        plt.show()