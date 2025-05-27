"""
    RobustPeaks.py

    Copyright (c) 2021-2023, SAXS Team, KEK-PF
"""
import logging
from bisect import bisect_right
import numpy as np
from scipy.stats import linregress
from molass_legacy.KekLib.SciPyCookbook import smooth
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
import molass_legacy.KekLib.DebugPlot as plt

COUNT_FOR_BREAK = 3
MIN_HEIGHT_RATIO = 0.05
EXTEND_HEIGHT_RATIO = 0.02
EXTREME_POS_RATIO = 0.95        # < 0.99 for 20201007_2

def guess_peak_width(x, y, pt, min_width, debug=False):
    from scipy.stats import linregress

    r_values = []
    last_r_value = None
    dec_count = 0
    for i in range(min_width, pt-min_width):
        x_ = x[pt-i:pt]
        y_ = y[pt-i:pt]
        slope, intercept, r_value, p_value, std_err = linregress(x_, y_)
        r_values.append(r_value)
        if last_r_value is not None and r_value < last_r_value:
            dec_count += 1
            if dec_count > COUNT_FOR_BREAK:
                # i.e., more than COUNT_FOR_BREAK consecutive decrease
                break
        else:
            dec_count = 0
        last_r_value = r_value

    left_width = min_width + np.argmax(r_values)

    if debug:
        plt.push()
        fig, ax = plt.subplots()
        ax.set_title("Left of %d" % pt)
        ax.plot(x, y)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
        p = pt - left_width
        ax.plot([p, p], [ymin, ymax], ':', color='red')
        axt = ax.twinx()
        axt.grid(False)
        f = pt - min_width - len(r_values)
        t = f + len(r_values)
        axt.plot(x[f:t], list(reversed(r_values)), color='C1')
        plt.show()
        plt.pop()

    r_values = []
    last_r_value = None
    inc_count = 0
    for i in range(min_width, len(x)-pt-min_width):
        x_ = x[pt:pt+i]
        y_ = y[pt:pt+i]
        slope, intercept, r_value, p_value, std_err = linregress(x_, y_)
        r_values.append(r_value)
        if last_r_value is not None and r_value > last_r_value:
            inc_count += 1
            if inc_count > COUNT_FOR_BREAK:
                # i.e., more than COUNT_FOR_BREAK consecutive increase
                break
        else:
            inc_count = 0
        last_r_value = r_value

    right_width = min_width + np.argmin(r_values)

    if debug:
        plt.push()
        fig, ax = plt.subplots()
        ax.set_title("Right of %d" % pt)
        ax.plot(x, y)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
        p = pt + right_width
        ax.plot([p, p], [ymin, ymax], ':', color='red')
        axt = ax.twinx()
        axt.grid(False)
        f = pt + min_width
        t = f + len(r_values)
        axt.plot(x[f:t], r_values, color='C1')
        plt.show()
        plt.pop()

    return left_width, right_width

def fix_bubble(pt, y):
    ny = y[pt-5:pt+5]
    h = ny[-1] - ny[0]
    by = ny[0] + np.arange(10)*h/9
    ny_ = ny - by
    pp = np.where(ny_ > abs(h)/2)[0]
    pp_ = pt-5+pp
    y[pp_] -= ny_[pp]

    if False:
        print("abs(h)/2=", abs(h)/2)
        print("pp=", pp)
        plt.push()
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,7))
        ax1.plot(ny_)
        ax2.plot(y)
        ymin, ymax = ax2.get_ylim()
        for p in pp_[[0,-1]]:
            ax2.plot([p, p], [ymin, ymax], ':', color='red')
        fig.tight_layout()
        plt.show()
        plt.pop()

    return pp_[[0,-1]]

def recognize_peaks_roughly(x, y, logger=None, num_peaks=None, min_ratio=MIN_HEIGHT_RATIO, min_width=10, check_bubble=False, return_ignore_info=False, debug=False):
    # min_width=10 is too small for 20200630_8

    y_copy = y.copy()
    pt = np.argmax(y_copy)
    pos_ratio = pt/len(y_copy)
    ignore_info = None
    if pos_ratio > EXTREME_POS_RATIO:
        from molass_legacy.KekLib.BasicUtils import Struct
        from GeometryUtils import rotated_argmin
        # like ths last peak in in 20201007_2
        logger.warning("an abnormal peak will be ignored due to pos_ratio=%.3g > %.3g", pos_ratio, EXTREME_POS_RATIO)
        j = rotated_argmin(-np.pi/4, y_copy, debug=False)
        y_copy[j:] = 0
        pt = np.argmax(y_copy)
        ignore_info = Struct(slice_=slice(j, None))

    y_max = y_copy[pt]

    half_min_width = min_width//2
    if check_bubble:
        bubble_check_ratio = np.average(y[pt-half_min_width:pt+half_min_width])/y_max
        print("bubble_check_ratio=", bubble_check_ratio)
        if bubble_check_ratio < 0.5:
            # for bubles in e.g. 20170226/Sugiyama
            f, t = fix_bubble(pt, y_copy)
            pt_ = np.argmax(y_copy)
            y_max = y_copy[pt]
            logger.warning("suspected bubble at %d:%d has been fixed to get a new peak top %d.", f, t+1, pt_)
            pt = pt_

    peak_tops = []
    peaks = []
    for k in range(10):
        try:
            left_width, right_width = guess_peak_width(x, y_copy, pt, min_width, debug=debug)
        except:
            log_exception(logger, "guess_peak_width failed to end looking for peaks: ")
            break

        print([k], "left_width, right_width=", left_width, right_width)
        if min(left_width, right_width) < min_width:
            break

        height_ratio = y_copy[pt]/min_ratio
        local_height_ratio = (y_copy[pt] - np.min(y_copy[[pt-left_width, pt+right_width]]))/y_max
        if local_height_ratio < min_ratio and height_ratio < min_ratio*2:
            break

        if debug:
            def debug_plot(title):
                print("left_width, right_width=", left_width, right_width)
                plt.push()
                fig, ax = plt.subplots()
                ax.set_title(title + "[%d]-th debug plot" % k)
                ax.plot(y)
                ax.plot(y_copy)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
                for w in [-left_width*2, -left_width, right_width, right_width*2]:
                    b = pt + w
                    ax.plot([b, b], [ymin, ymax], ':', color='yellow')
                fig.tight_layout()
                plt.show()
                plt.pop()
            debug_plot("before crush ")

        i = bisect_right(peak_tops, pt)
        peak_tops.insert(i, pt)
        print([k], "peak_tops=", peak_tops)
        peaks.insert(i, [pt-left_width, pt, pt+right_width])
        if num_peaks is not None and len(peaks) == num_peaks:
            break

        f_, t_ = get_covered_range(x, y_copy, peaks, i, pt, left_width, right_width, logger)
        y_copy[f_:t_] = 0
        pt_next = np.argmax(y_copy)

        if debug:
            print("pt_next=", pt_next)
            print("y_copy[pt_next]/y_max=", y_copy[pt_next]/y_max, min_ratio)
            debug_plot("after crush ")

        if len(np.where(y_copy[pt_next-half_min_width:pt_next+half_min_width] == 0)[0])/min_width > 0.3:
            break

        if y_copy[pt_next]/y_max < min_ratio:
            break

        pt = pt_next

    if return_ignore_info:
        return peaks, ignore_info
    else:
        return peaks

def get_covered_range(x, y, peaks, i, pt, left_width, right_width, logger):
    ysize = len(y)
    # covered_range cannot overlap the other peak ranges
    left_bound = 0 if i == 0 else peaks[i-1][-1]
    f_ = max(left_bound, pt - left_width*3)
    ex_slice = slice(f_, pt-left_width)
    r_value = linregress(x[ex_slice], y[ex_slice])[2]
    # print([i], "left extended r_value=", r_value)
    if r_value < 0.6:   # 0.569 for 20191006_proteins5 Xray
        # too long extension mus be fixed
        # as in 20191006_proteins5
        n = np.argmax(y[ex_slice])
        start = f_ + n
        new_f_ = start + np.argmin(y[start:pt-left_width])
        logger.info("[%d] covered_range left limit recomputed from %d to %d due to r_value=%.3g", i, f_, new_f_, r_value)
        f_ = new_f_

    right_bound = ysize if i == len(peaks)-1 else peaks[i+1][0]
    t_ = min(right_bound, pt + right_width*3)
    ex_slice = slice(pt+right_width, t_)
    r_value = linregress(x[ex_slice], y[ex_slice])[2]
    # print([i], "right extended r_value=", r_value)
    if r_value > -0.6:
        # too long extension mus be fixed
        n = np.argmax(y[ex_slice])
        start = pt + right_width
        new_t_ = start + np.argmin(y[start:start+n+1])
        logger.info("[%d] covered_range right limit recomputed from %d to %d due to r_value=%.3g", i, t_, new_t_, r_value)
        t_ = new_t_

    return f_, t_

class RobustPeaks:
    def __init__(self, x, y, debug=False):
        self.logger = logging.getLogger(__name__)
        self.x = x
        self.y = y
        self.max_y = np.max(y)
        self.peaks, self.ignore_info = recognize_peaks_roughly(x, y, logger=self.logger, return_ignore_info=True, debug=debug)

    def get_peaks(self):
        return self.peaks

    def get_limits_roughly(self, num_sigmas):
        epeaks = self.get_peaks()
        limit, pt, _ = epeaks[0]
        left_limit = self.get_left_sigma_limit(num_sigmas, limit, pt)

        _, pt, limit = epeaks[-1]
        right_limit = self.get_right_sigma_limit(num_sigmas, limit, pt)
        if self.ignore_info is not None:
            right_limit = min(right_limit, self.ignore_info.slice_.start)
        return left_limit, right_limit

    def get_left_sigma_limit(self, num_sigmas, limit, pt):
        y = self.y
        i = limit
        while i > 0 and y[i-1] < y[i]:
            i -= 1
        width = pt - i
        limit = pt - num_sigmas*(width//2)
        if limit > self.x[0] and self.y[limit]/self.max_y > MIN_HEIGHT_RATIO:
            # as in 20200630_11 Xray
            try:
                sy = smooth(self.y[0:limit])
                limit = np.where(sy < self.max_y * EXTEND_HEIGHT_RATIO)[0][-1]
            except:
                log_exception(self.logger, "extending left_sigma_limit failed: ")
                limit = self.x[0] 
        return max(self.x[0], limit)

    def get_right_sigma_limit(self, num_sigmas, limit, pt):
        y = self.y
        i = limit
        N = len(self.x) - 1
        while i < N and y[i] > y[i+1]:
            i += 1
        width = i - pt
        limit = pt + num_sigmas*(width//2)
        if limit < self.x[-1] and self.y[limit]/self.max_y > MIN_HEIGHT_RATIO:
            # not yet acknowledged
            self.logger.warning("right_sigma_limit might better be extended.")
        return min(self.x[-1], limit)

def demo(name, curve, debug=False):
    from time import time
    from molass_legacy.Elution.CurveUtils import simple_plot

    x = curve.x
    y = curve.y
    t0 = time()
    rp = RobustPeaks(x, y, debug=debug)
    peaks = rp.get_peaks()
    limits = rp.get_limits_roughly(10)
    print("it took %.3g seconds" % (time()-t0))

    plt.push()
    fig, ax = plt.subplots()
    ax.set_title("Demo Plot for " + name)
    simple_plot(ax, curve, legend=False)
    for k, p in enumerate(peaks):
        print([k], p)
        ax.plot(p, y[p], 'o')

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    for p in limits:
        ax.plot([p, p], [ymin, ymax], color='yellow')

    fig.tight_layout()
    plt.show()
    plt.pop()

    return limits
