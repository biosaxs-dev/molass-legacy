"""
    Distance.DeviationMeasure.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

VALID_Y_RATIO = 0.03
WANTED_RATIO_LIMIT = 0.5
FEATURE_SCALE = 2

def feature_deviation(y1, y2, max_y2, y2_norm, debug=False, fig_info=None):
    diff = y1 - y2
    base = 0.7*y1 + 0.3*y2
    valid = y2 > max_y2*VALID_Y_RATIO
    ratio = np.abs(diff[valid])/base[valid]
    wanted_ratio = ratio[ratio > WANTED_RATIO_LIMIT]
    if len(wanted_ratio) > 0:
        fdev = np.mean(wanted_ratio)
    else:
        fdev = WANTED_RATIO_LIMIT
    ret_dev = np.log10(np.linalg.norm(diff)/y2_norm)/FEATURE_SCALE + np.log10(fdev)*FEATURE_SCALE

    if debug:
        print("fdev=", fdev)
        print("ret_dev=", ret_dev)
        x = np.arange(len(y2))
        x_ = x[valid]
        def debug_plot(fig, ax):
            ax.set_title("feature_deviation debug")
            ax.plot(x, y1, label="y1")
            ax.plot(x, y2, label="y2")
            ax.plot(x, base, label="base")
            ax.plot(x, diff, label="diff")
            axt = ax.twinx()
            axt.grid(False)
            axt.plot(x_, ratio, label="ratio")
            ax.legend()
            axt.legend(loc="center right")

        if fig_info is None:
            with plt.Dp():
                fig, ax = plt.subplots()
                debug_plot(fig, ax)
                fig.tight_layout()
                plt.show()
        else:
            fig, ax = fig_info
            debug_plot(fig, ax)
    
    return ret_dev
