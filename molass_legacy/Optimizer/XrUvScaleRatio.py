"""
    Optimizer.XrUvScaleRatio.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np

RATIO_ALLOW = 1.3       # > 1.0 for OA_Ald
PENALTY_SCALE = 0.1     # too large value may cause the optimization early stages to fail

def xruv_scale_ratio_penalty(xr_scales, uv_scales, log_ratio=True, debug=False):
    if log_ratio:
        uv_xr_ratio = np.log(uv_scales/xr_scales)
    else:
        uv_xr_ratio = uv_scales/xr_scales
    average = np.average(uv_xr_ratio)
    std = np.std(uv_xr_ratio)
    dev = np.abs(uv_xr_ratio - average)
    exv = dev[dev > std*RATIO_ALLOW]
    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        print("average=", average)
        print("dev=", dev)
        allow = std*RATIO_ALLOW
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title(f"UV/Xray Ratio Debug: RATIO_ALLOW={RATIO_ALLOW}", fontsize=16)
            ax.bar(np.arange(len(uv_xr_ratio)), uv_xr_ratio)
            ax.axhline(average)
            ax.axhspan(average - std, average + std, color="green", alpha=0.1, label="1 std dev")
            ax.axhspan(average - allow, average + allow, color="yellow", alpha=0.2, label="allowance")
            ax.legend()
            fig.tight_layout()
            plt.show()
    if len(exv) > 0:
        penalty = PENALTY_SCALE*np.sum(exv)
    else:
        penalty = 0.0
    return penalty