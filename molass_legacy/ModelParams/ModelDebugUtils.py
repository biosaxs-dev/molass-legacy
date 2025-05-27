"""
    ModelDebugUtils.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import molass_legacy.KekLib.DebugPlot as plt

def model_plot(x, y, xr_cy_list, xr_ty, uv_x, uv_y, uv_cy_list, uv_ty):

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))

        ax1.plot(uv_x, uv_y)
        for k, cy in enumerate(uv_cy_list):
            ax1.plot(uv_x, cy, ":", label="c=%d"%k)
        ax1.plot(uv_x, uv_ty, ":", color="red", label="c-total")
        ax1.legend()

        ax2.plot(x, y)
        for k, cy in enumerate(xr_cy_list):
            ax2.plot(x, cy, ":", label="c=%d"%k)
        ax2.plot(x, xr_ty, ":", color="red", label="c-total")
        ax2.legend()

        fig.tight_layout()
        plt.show()
