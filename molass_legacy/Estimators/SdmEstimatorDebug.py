"""
    Estimators.SdmEstimatorDebug.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def components_plot_debug_impl(optimizer, params):
    print("components_plot_debug_impl")

    lrf_info = optimizer.objective_func(params, return_lrf_info=True)

    def plot_components(ax, x, y, C, ty):
        ax.plot(x, y, label="data")
        for k, cy in enumerate(C):
            ax.plot(x, cy, ":", label="component-%d" % k)
        ax.plot(x, ty, ":", color="red", label="model total")

    with plt.Dp():
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
        fig.suptitle("components_plot_debug_impl")
        ax1, ax2 = axes[0,0], axes[0,1]
        ax3, ax4 = axes[1,0], axes[1,1]
        plot_components(ax1, lrf_info.uv_x, lrf_info.uv_y, lrf_info.matrices[3], lrf_info.uv_ty)
        plot_components(ax2, lrf_info.x, lrf_info.y, lrf_info.matrices[1], lrf_info.xr_ty)
        plot_components(ax3, lrf_info.uv_x, lrf_info.uv_y, lrf_info.scaled_uv_cy_array, lrf_info.uv_ty)
        plot_components(ax4, lrf_info.x, lrf_info.y, lrf_info.scaled_xr_cy_array, lrf_info.xr_ty)
        fig.tight_layout()
        ret = plt.show()

    return ret