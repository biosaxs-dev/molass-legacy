"""
    Trimming.OptViewRange.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Batch.LiteBatch import LiteBatch
from molass_legacy.Peaks.MomentsUtils import compute_moments
from molass_legacy.DataStructure.LPM import get_corrected

BAD_SN_NEGATIVE_THRESHOLD = 0.3

def correct_largely_negative_values(y, num_peaks):
    y = get_corrected(y)
    where_negetive = np.where(y < 0)[0]
    negetve_ratio = len(where_negetive)/len(y)
    print("negetve_ratio=", negetve_ratio)
    if num_peaks == 1 or negetve_ratio > BAD_SN_NEGATIVE_THRESHOLD:
        # as in 20250212/REF
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("applied correct_largely_negative_values due to negetve_ratio=%.3g", negetve_ratio)          
        m = np.argmax(y)
        y_ = y.copy()
        left_negative = where_negetive[where_negetive < m]
        if len(left_negative) > 0:
            y_[:left_negative[-1]] = 0
        right_negative = where_negetive[where_negetive > m]
        if len(right_negative) > 0:
            y_[right_negative[0]:] = 0
    else:
        y_ = y
    return y_

class OptViewRange:
    def __init__(self, x, y, num_peaks, upper=None, smooth=False):
        if smooth:
            from molass_legacy.KekLib.SciPyCookbook import smooth
            y = smooth(y)
        if upper is None:
            x_ = x
            y_ = y
        else:
            y = correct_largely_negative_values(y, num_peaks)
            max_y = np.max(y)
            upper_part = y > max_y * upper
            x_ = x[upper_part]
            y_ = y[upper_part]
        self.x_ = x_
        self.y_ = y_
        self.moments = compute_moments(x_, y_)

    def get_range(self, sigmas=5):
        m1 = self.moments[1]
        hw = sigmas * np.sqrt(self.moments[2])
        return m1 - hw, m1 + hw
    
def get_opt_view_range(caller, debug=True):
    print("get_opt_view_range")
    for keystr in ["uv_restrict_list", "xr_restrict_list"]:
        print(keystr, get_setting(keystr))

    lb = LiteBatch()
    lb.prepare(caller.serial_data, debug=False)
    uv_x, uv_y, xr_x, xr_y, baselines = lb.get_curve_xy(return_baselines=True, debug=False)
    uv_y_ = uv_y - baselines[0]
    xr_y_ = xr_y - baselines[1]
    ranges = []
    ranges_xy = []
    for x, y in [(uv_x, uv_y_), (xr_x, xr_y_)]:
        range_ = OptViewRange(x, y, upper=0.03)
        ranges.append(range_.get_range())
        ranges_xy.append((range_.x_, range_.y_))

    if debug:
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            ax1.plot(uv_x, uv_y)
            ax1.plot(uv_x, baselines[0])
            ax1.plot(*ranges_xy[0], 'o', color='yellow')
            ax1.axvspan(*ranges[0], alpha=0.2)
            ax2.plot(xr_x, xr_y)
            ax2.plot(xr_x, baselines[1])
            ax2.plot(*ranges_xy[1], 'o', color='yellow')
            ax2.axvspan(*ranges[1], alpha=0.2)
            fig.tight_layout()
            plt.show()