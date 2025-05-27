"""
    FlowChangePrep.py

    Copyright (c) 2021-2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.stats import linregress
from molass_legacy.UV.PlainCurve import check_diehardness, flattenable

REMAINING_FLATTEN_MARGIN = 50   # required in 20211021 to modify the positions from integral
REMAINING_FLATTEN_PROPORTIONS = [0.05, 0.95]    # check at least for 20180617 to narrow, 20211021 to widen
MODIFY_REMAINING = True
REMAINING_RATIO_TEST = 0.1
REMAINING_RATIO_LIMIT = 0.15    # > 0.285 for 20170304, < 0.098 for 20181127

def flatten_remaining_peak(peak_region, y, debug=False):
    j1_, j2_ = peak_region.get_peak_ends(REMAINING_FLATTEN_PROPORTIONS)[1]
    j1 = max(0, j1_-REMAINING_FLATTEN_MARGIN)
    j2 = min(len(y)-1, j2_+REMAINING_FLATTEN_MARGIN)
    ret_y = y.copy()
    n = j2-j1
    w = np.arange(n)/(n-1)
    ret_y[j1:j2] = y[j1]*(1-w) + y[j2]*w
    if debug:
        import molass_legacy.KekLib.DebugPlot as plt

        with plt.Dp():
            x = np.arange(len(y))
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.plot(x, ret_y, ":")
            ymin, ymax = ax.get_ylim()
            ax.set_ylim()
            for j in [j1_, j2_]:
                ax.plot([j, j], [ymin, ymax], ":", color="blue")
            for j in [j1, j2]:
                ax.plot([j, j], [ymin, ymax], ":", color="red")
            fig.tight_layout()
            plt.show()
            # exit()
    return ret_y

def get_easier_curve_y(a_curve, a_curve2, peak_region, logger, debug=False):
    x = a_curve2.x
    y = a_curve2.y

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt

        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("get_easier_curve_y entry")
            ax1.plot(a_curve.x, a_curve.y)
            ax2.plot(x, y)
            fig.tight_layout()
            plt.show()

    y_for_gy = y
    if MODIFY_REMAINING:
        y_modified = False
        diehard, max_k, ratio = check_diehardness(a_curve, a_curve2)
        if flattenable(ratio):
            # task: unify this and Uv.UvPreRecog.py
            pk_i = a_curve.primary_peak_i
            pk_y = np.average(a_curve2.y[pk_i-20:pk_i+21])
            remaining_ratio = pk_y/a_curve.primary_peak_y
            print("remaining_ratio=", remaining_ratio)
            if remaining_ratio > REMAINING_RATIO_TEST:
                outside = peak_region.get_outside()
                slope, intercept = linregress(x[outside], y[outside])[0:2]
                remaining_ratio = (pk_y - (slope*pk_i + intercept))/a_curve.primary_peak_y
                print("corrected remaining_ratio=", remaining_ratio)
                if remaining_ratio > REMAINING_RATIO_LIMIT:
                    # as in 20180219
                    y_for_gy = flatten_remaining_peak(peak_region, y)
                    y_modified = True
                    logger.info("remaining peak has been flattened due to remaining_ratio=%.3g > %.3g.", remaining_ratio, REMAINING_RATIO_LIMIT)
        else:
            # as in 20211021 where ratio == 1.54
            y_for_gy = np.zeros(len(y))
            y_modified = True
    else:
        y_modified = False

    first_peak, ret_y = peak_region.get_first_peak()
    last_peak = a_curve.peak_info[-1]

    if debug:
        print("diehard, max_k, ratio=", diehard, max_k, ratio)
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("get_easier_curve_y return")
            ax1.plot(a_curve.x, a_curve.y)
            ax2.plot(x, y)
            ax2.plot(x, y_for_gy)
            fig.tight_layout()
            plt.show()

    return y_modified, y_for_gy, first_peak, last_peak
