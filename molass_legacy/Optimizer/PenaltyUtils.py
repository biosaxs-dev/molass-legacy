"""
    Optimizer.PenaltyUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from molass_legacy.Optimizer.XrUvScaleRatio import xruv_scale_ratio_penalty
PENALTY_SCALE = 1e3
UV_B_ALLOW_RATIO = 0.1

def compute_mapping_penalty(uv_curve, xr_curve, init_mapping, mapping, uv_x_size, xr_scales, uv_scales, debug=False):
    a_init, b_init = init_mapping
    a, b = mapping
    ratio = a/a_init
    a_deviation = min(0, ratio - 0.99)**2 + max(0, ratio - 1.05)**2
    b_allowance = uv_x_size * UV_B_ALLOW_RATIO
    b_deviation = max(0, abs(b - b_init) - b_allowance)**2

    ratio_penalty = xruv_scale_ratio_penalty(xr_scales, uv_scales, debug=debug)
    mapping_penalty = PENALTY_SCALE * (a_deviation + b_deviation ) + ratio_penalty

    if debug:
        print("ratio=", ratio)
        print("a_deviation=", a_deviation)
        print("b_allowance=", b_allowance)
        print("ratio_penalty=", ratio_penalty)
        print("mapping_penalty=", mapping_penalty)
        xr_x = xr_curve.x
        import molass_legacy.KekLib.DebugPlot as plt
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("compute_mapping_penalty: debug")
            ax.set_xlabel("XR")
            ax.set_ylabel("UV")
            ax.plot(xr_x, xr_x*a_init + b_init, label='initial mapping')
            ax.plot(xr_x, xr_x*a + b, ':', label='current mapping')
            fig.tight_layout()
            plt.show()

    return mapping_penalty