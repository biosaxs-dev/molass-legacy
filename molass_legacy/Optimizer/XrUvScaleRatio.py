"""
    Optimizer.XrUvScaleRatio.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np

ALLOW_RATIO = 1.5       # > 1.0 for OA_Ald
PENALTY_SCALE = 0.1     # too large value may cause the optimization early stages to fail

def xruv_scale_ratio_penalty(xr_scales, uv_scales, debug=False):
    uv_xr_ratio = uv_scales/xr_scales
    average = np.average(uv_xr_ratio)
    std = np.std(uv_xr_ratio)
    dev = np.abs(uv_xr_ratio - average)
    exv = dev[dev > std*ALLOW_RATIO]
    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        print("average=", average)
        print("dev=", dev)
        allow = std*ALLOW_RATIO
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("UV/Xray Ratio Debug", fontsize=16)
            ax.bar(np.arange(len(uv_xr_ratio)), uv_xr_ratio)
            ax.axhline(average)
            ax.axhspan(average - allow, average + allow, color="yellow", alpha=0.3)
            fig.tight_layout()
            plt.show()
    if len(exv) > 0:
        penalty = PENALTY_SCALE*np.sum(exv)
    else:
        penalty = 0.0
    return penalty