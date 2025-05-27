"""
    PeakCurve.py

        temporary safer replacement of ElutionCurve for PeakMapper

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from molass_legacy.KekLib.SciPyCookbook import smooth
import molass_legacy.KekLib.DebugPlot as plt
from GeometryUtils import rotated_argminmax
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from OptimalSmoothing import OptimalSmoothing
from CurveFeatures import CurveFeatures

CORRECT_OK_LIMIT = 0.02     # > 0.015 for 20211209
NUM_CORRECTION = 2
END_WIDTH = 5
SAFE_SIZE_RATIO = 0.6       # < 0.65 for 20191118_3

def shift_range_pairs(j, range_pairs):
    return [ [ [ j + r for r in range_ ] for range_ in range_pair ] for range_pair in range_pairs ]

def get_peak_curve_info(curve, debug=False):
    from LPM import get_corrected   # moved due to ImportError: ... (most likely due to a circular import)

    x = curve.x
    y = curve.y
    max_y = curve.max_y

    start = 0
    for k in range(NUM_CORRECTION):
        cr_y = get_corrected(y)
        cm_y = np.cumsum(cr_y)  # note that this correction is for peak region recognition only
        if debug:
            with plt.Dp():
                fig,ax = plt.subplots()
                ax.set_title("get_peak_curve_info", fontsize=16)
                ax.plot(x, y, label="data")
                ax.plot(x, cr_y, ":", label="corrected")
                axt = ax.twinx()
                axt.grid(False)
                axt.plot(x, cm_y, color="cyan", label="cumulated")
                ax.legend(fontsize=16)
                axt.legend(loc="center right", fontsize=16)
                fig.tight_layout()
                plt.show()
        i1, i2 = rotated_argminmax(-np.pi/8, cm_y, debug=debug)
        half_width = (i2 - i1)*2//3
        center = (i1 + i2)//2
        j1 = max(0, center - half_width)
        j2 = min(len(x)-1, center+ half_width)
        slice_ = slice(j1, j2)
        x_ = x[slice_]
        y_ = cr_y[slice_]
        ratio1 = abs(np.average(y_[0:END_WIDTH])/max_y)
        ratio2 = abs(np.average(y_[-END_WIDTH:])/max_y)
        if debug:
            print([k], "ratio1=", ratio1, "ratio2=", ratio2)
        if ratio1 < CORRECT_OK_LIMIT and ratio2 < CORRECT_OK_LIMIT:
            break
        else:
            if k < NUM_CORRECTION - 1:
                start += j1
                x = x_
                y = y_

    prs = [int(round(p)) for p in curve.get_peak_region_sigma()]
    slice_ = slice(min(prs[0], start+j1), max(prs[1], start+j2))

    size_ratio = (slice_.stop - slice_.start)/len(curve.x)      # use curve.x not x for x may have been changed
    # print("------------------ size_ratio=", size_ratio)
    assert size_ratio < SAFE_SIZE_RATIO

    if debug:
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            ax1.plot(curve.x, curve.y, label="orignal data")
            ax1.plot(x, cr_y, label="corrected data")
            ax1.plot(x_, y_, ":", label="sliced data")
            ax2.plot(x_, y_, label="sliced data")
            for ax in ax1, ax2:
                ax.legend()
            fig.tight_layout()
            plt.show()

    if debug:
        with plt.Dp():
            fig,ax = plt.subplots()
            ax.set_title("get_peak_curve_info", fontsize=16)
            ax.plot(curve.x, curve.y, label="data")
            ax.plot(x, cr_y, ":", label="corrected")
            ymin,ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            for j in [slice_.start, slice_.stop]:
                ax.plot([j, j], [ymin,ymax], color="yellow")
            axt = ax.twinx()
            axt.grid(False)
            axt.plot(x, cm_y, color="cyan", label="cumulated")
            ax.legend(fontsize=16)
            axt.legend(loc="center right", fontsize=16)
            fig.tight_layout()
            plt.show()

    ecurve = ElutionCurve(curve.y[slice_])      # use no correction y
    ecurve.get_default_editor_ranges()

    if debug:
        from molass_legacy.Elution.CurveUtils import simple_plot
        fig, ax = plt.subplots()
        ax.set_title("get_peak_curve_info")
        simple_plot(ax, ecurve)
        plt.show()

    return ecurve, slice_

class PeakCurve(ElutionCurve):
    def __init__(self, curve, pkslice_info=None, debug=False):
        self.logger = logging.getLogger(__name__)

        if pkslice_info is None:
            pkcurve, slice_ = get_peak_curve_info(curve, debug=debug)
        else:
            pkcurve, slice_ = pkslice_info

        self.peak_slice = slice_
        self.x = x = curve.x
        self.y = y = curve.y
        self.possiblly_peakless = False
        self.emg_peaks = None
        self.paired_ranges = None

        if debug:
            from molass_legacy.Elution.CurveUtils import simple_plot
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("sliced curve")
                simple_plot(ax, pkcurve)
                fig.tight_layout()
                plt.show()

        j1 = slice_.start
        self.min_peak_width = pkcurve.min_peak_width
        self.peak_top_x = j1 + pkcurve.peak_top_x
        self.peak_info = [[j1 + j for j in rec] for rec in pkcurve.peak_info]
        self.boundaries = [j1 + b for b in pkcurve.boundaries]
        self.major_peak_info = [[j1 + j for j in rec] for rec in pkcurve.major_peak_info]
        self.editor_ranges = shift_range_pairs(j1, pkcurve.editor_ranges)

        self.max_x = np.argmax(y)
        self.max_y = y[self.max_x]
        self.min_y = np.percentile(y, 5)
        self.height = self.max_y - self.min_y

        self.smoothing = OptimalSmoothing(x, y, self.height, self.min_y)
        self.smoothing.compute_optimal_curves()
        self.y_for_spline = self.smoothing.spline_y
        self.feature_y = self.smoothing.feature_y
        self.spline = self.smoothing.spline
        self.sy = self.spline(x)

        self.d1 = self.smoothing.d1
        self.d1y = self.d1(x)
        self.d2 = self.smoothing.d2
        self.d2y = self.d2(x)
        self.d2_roots = self.smoothing.d2_roots

        self.peak_region_sigma = None

        try:
            self.feature = CurveFeatures(self)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            print(etb)

    def get_major_peak_info(self, ratio=None):
        return self.major_peak_info

    def get_peak_region_width(self):
        # note that this overrides ElutionCurve.get_peak_region_width()
        # or rather, ElutionCurve.get_peak_region_width() is a substitute for this
        # when a PeakCurve is not used
        return self.peak_slice.stop - self.peak_slice.start
