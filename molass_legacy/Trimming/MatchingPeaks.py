"""
    MatchingPeaks.py

    Copyright (c) 2020-2022, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from bisect import bisect_right
from itertools import combinations

ALLOWANCE_RATIO = 0.05

def compute_matching_indeces(size, uv_top_x, xr_top_x, logger=None):
    """
    e.g.,
                   input           output (return)
        uv_top_x = [189, 625]   => [   0, 1, None]
        xr_top_x = [624, 800]   => [None, 0,    1]
    """
    ret_list1 = []
    ret_list2 = []

    if len(uv_top_x) >= len(xr_top_x):
        list_more = uv_top_x
        list_less = xr_top_x
        reversed_ = False
    else:
        list_more = xr_top_x
        list_less = uv_top_x
        reversed_ = True

    less_indeces = list(range(len(list_less)))
    greater_indeces = list(range(len(list_more)))

    ret_pairs = []
    start_j = 0
    matched = set()
    unmatched_peaks = []
    for i in greater_indeces:
        found = False
        for j in less_indeces[start_j:]:
            dist = abs(list_more[i] - list_less[j])
            if dist/size < ALLOWANCE_RATIO:
                ret_pairs.append((i,j))
                start_j = j + 1
                found = True
                matched.add(j)
                break
        if not found:
            ret_pairs.append((i,None))
            unmatched_peaks.append(list_more[i])

    if len(matched) < len(list_less):
        unmatched = list(set(less_indeces) - matched)
        for k in unmatched:
            peak = list_less[k]
            i = bisect_right(list_more, peak)
            ret_pairs.insert(i, (None, k))
            unmatched_peaks.append(list_less[k])

    if len(unmatched_peaks) > 0 and logger is not None:
        logger.warning("compute_matching_indeces detected unmatched peaks in : %s", str(sorted(unmatched_peaks)))

    list1, list2 = list(zip(*ret_pairs))
    if reversed_:
        list2, list1 = list1, list2
    return list1, list2

class MatchingPeaks:
    def __init__(self, cs, uv_curve_, xr_curve_, debug=False):
        self.logger = logging.getLogger(__name__)

        self.uv_curve = uv_curve = cs.a_curve
        self.xr_curve = xr_curve = cs.x_curve

        A = cs.slope
        B = cs.intercept

        uv_peaks = uv_curve.get_emg_peaks()
        xr_peaks = xr_curve.get_emg_peaks()
        uv_top_x = [int(p.top_x) for p in uv_peaks]
        xr_top_x = [int(p.top_x*A + B)  for p in xr_peaks]
        ii, jj = compute_matching_indeces(len(uv_curve.y), uv_top_x, xr_top_x, self.logger)
        print("ii=", ii)
        print("jj=", jj)

        ret_uv_peaks = []
        ret_xr_peaks = []
        for i, j in zip(ii, jj):
            if i is None or j is None:
                # raise PeakMathcingError()
                """
                seems to be degraded after compute_matching_indeces changed
                for Mapping.FewerPointMapper.py, which is not used yet.
                """
                continue
            ret_uv_peaks.append(uv_peaks[i])
            ret_xr_peaks.append(xr_peaks[j])

        if debug:
            self.debug_plot_emg_peaks("get_matching_peak_lists", xr_peaks, ret_xr_peaks, uv_peaks, ret_uv_peaks)

        self.ii = ii
        self.jj = jj
        self.uv_peaks = ret_uv_peaks
        self.xr_peaks = ret_xr_peaks

    def debug_plot_emg_peaks(self, title, xr_emg_peaks, ret_xr_peaks, uv_emg_peaks, ret_uv_peaks):
        import molass_legacy.KekLib.DebugPlot as plt
        from molass_legacy.Elution.CurveUtils import simple_plot

        plt.push()
        xr_ecurve = self.xr_curve
        uv_ecurve = self.uv_curve

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,10))

        fig.suptitle(title)

        def emg_peak_plot(ax, ecurve, color, emg_peaks):
            simple_plot(ax, ecurve, color=color, legend=False)
            x = ecurve.x
            for k, epeak in enumerate(emg_peaks):
                y = epeak.get_model_y(x)
                ax.plot(x, y, ':', label=str(k))
            ax.legend()

        emg_peak_plot(axes[0,0], xr_ecurve, "orange", xr_emg_peaks)
        emg_peak_plot(axes[0,1], xr_ecurve, "orange", ret_xr_peaks)
        x = uv_ecurve.x
        emg_peak_plot(axes[1,0], uv_ecurve, "blue", uv_emg_peaks)
        emg_peak_plot(axes[1,1],  uv_ecurve, "blue", ret_uv_peaks)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        plt.show()
        plt.pop()

    def get_matching_peak_lists(self):
        return self.uv_peaks, self.xr_peaks
