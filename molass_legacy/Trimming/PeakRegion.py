"""
    PeakRegion.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from bisect import bisect_right
from molass_legacy.KekLib.SciPyCookbook import smooth
from molass_legacy.DataStructure.LPM import get_corrected
from molass_legacy.Mapping.CurveSimilarity import CurveSimilarity

USE_ROTATION = True
if USE_ROTATION:
    from molass_legacy.KekLib.GeometryUtils import rotated_argminmax
    NORMAL_ANGLE = -np.pi/8
else:
    TAIL_RATIO = 0.15               # > 0.12 for Factin
    STRICT_RATIO = 0.004            # < 0.01 for 20190309_1, <= 0.004 for Sugiyama
IN_PEAK_DISMISS_RATIO = 0.3     # value to dismiss pH6, not to dismiss 20170301
SYMMETRY_ALLOW = 0.2            # > 0.12 for 20190529_1
SIGNIFICANT_RATIO_LIMIT = 0.05  # > 0.057 for 20161202, < 0.116 for SUB_TRN1, < 0.051 for Sugiyama
SPECIAL_CIRCUMVENT_LIMIT = 0.02 # > 0.0185 for 20170301
PTX_RATIO_LIMIT = 0.8
CORRECTION_THRESHOLD_RATIO = 0.3

class PeakRegion:
    def __init__(self, x_curve, a_curve, a_curve2, debug=False):
        self.logger = logging.getLogger(__name__)

        y = x_curve.y
        ratio = abs(y[0] - y[-1])/x_curve.max_y
        if ratio > CORRECTION_THRESHOLD_RATIO:
            # as in 20181203
            from molass_legacy.DataStructure.LPM import get_corrected
            from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
            x = x_curve.x
            y = get_corrected(y, x=x)
            x_curve = ElutionCurve(y, x=x)
            self.logger.info("x_curve has been temporarily replaced by a corrected curve due to the ratio %.3g", ratio)

        self.x_curve = x_curve
        self.a_curve = a_curve
        self.a_curve2 = a_curve2

        self.similarity = CurveSimilarity(a_curve, x_curve)
        self.compute_cumsum(x_curve, a_curve)
        self.set_default_ends(debug=debug)
        self.set_wider_ends()

    def compute_cumsum(self, x_curve, a_curve):
        xr_cy, xr_y = self.compute_cumsum_impl(x_curve)
        ptx = x_curve.get_primarypeak_i()
        ptx_ratio = xr_cy[ptx]/xr_cy[-1]
        print("ptx_ratio=", ptx_ratio)
        if ptx_ratio > PTX_RATIO_LIMIT:
            # as in 20200304_1
            self.logger.info("re-computing cumsum for peak region due to ptx_ratio=%.3g > %.3g", ptx_ratio, PTX_RATIO_LIMIT)
            xr_cy = self.compute_cumsum_from_uv(x_curve, a_curve)
        self.xr_y = xr_y
        self.xr_cy = xr_cy

    def compute_cumsum_impl(self, curve, debug=False):
        xr_y = get_corrected(smooth(curve.y))    # smoothing is required in low quality data as in pH7
        xr_cy = np.cumsum(xr_y)
        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            with plt.Dp():
                x = curve.x
                y = curve.y
                fig, ax = plt.subplots()
                ax.set_title("compute_cumsum_impl")
                ax.plot(x, y)
                ax.plot(x, xr_y)
                axt = ax.twinx()
                axt.plot(x, xr_cy)
                fig.tight_layout()
                plt.show()
        return xr_cy, xr_y

    def compute_cumsum_from_uv(self, x_curve, a_curve, debug=False):
        from scipy.interpolate import UnivariateSpline

        uv_cy, uv_y = self.compute_cumsum_impl(a_curve)
        scale = x_curve.max_y/a_curve.max_y
        uv_cy_spline = UnivariateSpline(a_curve.x, scale * uv_cy, s=0, ext=3)
        uv_x = x_curve.x * self.similarity.slope + self.similarity.intercept
        xr_cy = uv_cy_spline(uv_x)
        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.Elution.CurveUtils import simple_plot
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                simple_plot(ax1, a_curve, color="blue")
                simple_plot(ax2, x_curve, color="orange")
                ax1t = ax1.twinx()
                ax1t.grid(False)
                ax1t.plot(a_curve.x, uv_cy)
                ax2t = ax2.twinx()
                ax2t.grid(False)
                ax2t.plot(x_curve.x, xr_cy)
                fig.tight_layout()
                plt.show()
        return xr_cy

    def set_default_ends(self, debug=False, fig_file=None):
        if USE_ROTATION:
            xr_peak_ends, uv_peak_ends = self.get_peak_ends_with_rotation(NORMAL_ANGLE)
        else:
            xr_peak_ends, uv_peak_ends = self.get_peak_ends([TAIL_RATIO, 1 - TAIL_RATIO])
        self.set_peak_ends_with_traditional_infos(xr_peak_ends, uv_peak_ends, debug, fig_file)

    def set_wider_ends(self):
        if USE_ROTATION:
            pe = self.peak_ends
            half_width = (pe[1] - pe[0])*0.7
            center = (pe[1] + pe[0])/2
            uv_peak_ends = [ max(0, int(center - half_width)), min(len(self.a_curve.x)-1, int(center + half_width))]
        else:
            xr_peak_ends, uv_peak_ends = self.get_peak_ends([STRICT_RATIO, 1 - STRICT_RATIO])
        self.wider_ends = uv_peak_ends

    def get_peak_ends_impl(self, cy, proportions):
        peak_ends = []
        for r in proportions:
            j = bisect_right(cy, cy[-1]*r)
            peak_ends.append(j)
        return peak_ends

    def get_peak_ends(self, proportions, recursing=False):
        xr_cy = self.xr_cy
        xr_peak_ends = self.get_peak_ends_impl(xr_cy, proportions)
        xr_y = self.xr_y
        x_curve = self.x_curve

        if not USE_ROTATION:
            if xr_y[-1]/x_curve.max_y > TAIL_RATIO:
                # as in pH7
                xr_peak_ends[1] = len(xr_cy) - 1
                pass

        uv_peak_ends = self._guess_mapping(x_curve, xr_peak_ends, self.a_curve)

        if not recursing and not self.peak_ends_safety_check(self.a_curve, uv_peak_ends):
            self.logger.info("guessing peak region from molass_legacy.UV elution due to an unfavorable state.")
            # as in 20191118_3
            self.xr_cy = self.compute_cumsum_from_uv(x_curve, self.a_curve)
            xr_peak_ends, uv_peak_ends = self.get_peak_ends(proportions, recursing=True)

        return xr_peak_ends, uv_peak_ends

    def get_peak_ends_with_rotation(self, angle):
        xr_peak_ends = list(rotated_argminmax(angle, self.xr_cy, debug=False))
        uv_peak_ends = self._guess_mapping(self.x_curve, xr_peak_ends, self.a_curve)
        return xr_peak_ends, uv_peak_ends

    def _guess_mapping(self, x_curve, xr_peak_ends, a_curve):
        uv_peak_ends = [self.similarity.mapped_int_value(j) for j in xr_peak_ends]
        return uv_peak_ends

    def peak_ends_safety_check(self, ecurve, peak_ends, debug=False):
        # this is a risky judgement, any safer method is desired

        allow = 0       # setting safely for one can be harmful to the other as between 20171226 and 20191118_3
        ret = (peak_ends[0] - allow < ecurve.peak_info[0][1] and ecurve.peak_info[-1][1] < peak_ends[1] + allow)
                        # True for test_20191006_proteins5 and 20191118_3
        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.Elution.CurveUtils import simple_plot
            with plt.Dp():
                fig, ax = plt.subplots()
                simple_plot(ax, ecurve)
                ymin, ymax = ax.set_ylim()
                ax.set_ylim(ymin, ymax)
                for k, x in enumerate([peak_ends[0] - allow, ecurve.peak_info[0][1], ecurve.peak_info[-1][1], peak_ends[1] + allow]):
                    color = "yellow" if k in [0, 3] else "cyan"
                    ax.plot([x, x], [ymin, ymax], color=color)
                fig.tight_layout()
                plt.show()
        return ret

    def set_peak_ends_with_traditional_infos(self, xr_peak_ends, uv_peak_ends, debug, fig_file):
        x_curve = self.x_curve
        a_curve = self.a_curve
        a_curve2 = self.a_curve2

        first_peak, last_peak, uv_peak_ends_old, y = self._get_traditional_peak_ends(a_curve, curve2=a_curve2)
        xr_peak_ends_old = self._get_traditional_peak_ends(x_curve)[2]

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.Elution.CurveUtils import simple_plot
            from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

            if USE_ROTATION:
                lables = ["rotated min", "rotated max"]
            else:
                lables = ["%.3g" % p for p in [TAIL_RATIO, 1 - TAIL_RATIO]]

            xr_y = self.xr_y
            xr_cy = self.xr_cy
            in_folder = get_in_folder()
            plt.push()
            fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(18,5))
            fig.suptitle("Peak Region Determination from Integral Proportions for %s" % in_folder, fontsize=20)
            ax0.set_title("UV Peak Region", fontsize=16)
            ax1.set_title("Xray Peak Region", fontsize=16)
            ax2.set_title("LPM-corrected Xray Elution and Integral Proportions", fontsize=16)

            uv_slice = slice(*uv_peak_ends)
            xr_slice = slice(*xr_peak_ends)
            for ax, curve, color, slice_ in [(ax0, a_curve, "blue", uv_slice), (ax1, x_curve, "orange", xr_slice)]:
                simple_plot(ax, curve, color=color)
                x_ = curve.x
                y_ = curve.y
                rx = x_[slice_]
                zy = np.zeros(len(rx))
                ry = y_[slice_]
                ax.fill_between(rx, zy, ry, fc='pink', alpha=0.2)

            ax2.plot(x_curve.x, xr_y, ":", color="orange")
            axt = ax2.twinx()
            axt.grid(False)
            axt.plot(x_curve.x, xr_cy)

            ymin0, ymax0 = ax0.get_ylim()
            ax0.set_ylim(ymin0, ymax0)
            ymin1, ymax1 = ax1.get_ylim()
            ax1.set_ylim(ymin1, ymax1)
            ymin2, ymax2 = ax2.get_ylim()
            ax2.set_ylim(ymin2, ymax2)

            for i, p in enumerate(uv_peak_ends_old + uv_peak_ends):
                color = "yellow" if i < 2 else "cyan"
                ax0.plot([p, p], [ymin0, ymax0], color=color)

            for i, p in enumerate(xr_peak_ends_old + xr_peak_ends):
                color = "yellow" if i < 2 else "cyan"
                ax1.plot([p, p], [ymin1, ymax1], color=color)
                ax2.plot([p, p], [ymin2, ymax2], color=color)

            xmint, xmaxt = axt.get_xlim()
            ymint, ymaxt = axt.get_ylim()
            for i, p in enumerate(xr_peak_ends):
                px = p
                py = xr_cy[p]
                dx = (xmaxt - xmint)*0.1*(-1 + 2*i)
                dy = (ymaxt - ymint)*0.1*(1 - 2*i)
                axt.annotate(lables[i], xy=(px, py), xytext=(px+dx, py+dy), ha='center', arrowprops=dict(arrowstyle="->", color='k'))

            fig.tight_layout()
            if fig_file is None:
                plt.show()
            else:
                from time import sleep
                plt.show(block=False)
                fig = plt.gcf()
                fig.savefig(fig_file)
                sleep(1)
            plt.pop()

        self.y = y
        self.first_peak = first_peak
        self.last_peak = last_peak

        excess_width_ratio = (xr_peak_ends_old[0] - xr_peak_ends_old[0])/(xr_peak_ends[1] - xr_peak_ends_old[1])
        symmetry_deviation = abs(excess_width_ratio - 1)
        print("excess_width_ratio=%.3g" % excess_width_ratio)
        print("symmetry_deviation=%.3g" % symmetry_deviation)
        if symmetry_deviation < SYMMETRY_ALLOW:
            # trick to discard anomaly in 20190529_1
            self.peak_ends = uv_peak_ends
            self.logger.info("peak_ends have been set from integral proportions due to symmetry_deviation=%.3g < %.3g", symmetry_deviation, SYMMETRY_ALLOW)
        else:
            self.peak_ends = uv_peak_ends

    def _get_traditional_peak_ends(self, curve, curve2=None):
        first_peak = curve.peak_info[0]
        if curve2 is None:
            y = None
        else:
            x = curve2.x
            y = curve2.y
            pos_ratio = first_peak[1]/len(x)
            # print("pos_ratio=", pos_ratio)
            if pos_ratio < 0.05:    # pos_ratio = 0.023 for Kosugi8, 0.127 for 20160227
                # as in Kosugi8
                j = first_peak[2]
                y_ = y.copy()
                y_[0:j] = y[j]
                y = y_
                first_peak = curve.peak_info[1]
                self.logger.info("the first peak with pos_ratio=%.3g has been replaced with the second peak.", pos_ratio)

        w = IN_PEAK_DISMISS_RATIO
        end_L = first_peak[0]*(1-w) + first_peak[1]*w
        last_peak = curve.peak_info[-1]
        end_R = last_peak[2]
        peak_ends = [end_L, end_R]
        return first_peak, last_peak, peak_ends, y

    def get_first_peak(self):
        return self.first_peak, self.y

    def get_ends(self):
        return self.peak_ends

    def get_size(self):
        return self.peak_ends[1] - self.peak_ends[0]

    def get_wider_ends(self):
        return self.wider_ends

    def get_slice(self):
        return slice(self.peak_ends[0], self.peak_ends[1]+1)

    def get_outside(self):
        ret = np.ones(len(self.y), dtype=bool)
        ret[self.get_slice()] = False
        return ret

    def is_in_the_region(self, x):
        return self.peak_ends[0] <= x and x <= self.peak_ends[1]

    def is_special(self, x, return_ratio=False):
        sig_ratio = self.get_significance_ratio(x)
        print("sig_ratio=", sig_ratio)
        ret_bool = self.is_in_the_wider_region(x) and sig_ratio > SIGNIFICANT_RATIO_LIMIT
        if ret_bool:
            ret, diff_ratio = self.can_be_treated_as_usual(x)
            if ret:
                self.logger.info("this case is circumvented as not special due to diff_ratio=%.3g", diff_ratio)
                ret_bool = False

        if return_ratio:
            return ret_bool, sig_ratio
        else:
            return ret_bool

    def is_in_the_wider_region(self, x):
        return self.wider_ends[0] <= x and x <= self.wider_ends[1]

    def get_significance_ratio(self, x):
        return self.a_curve.spline(x)/self.a_curve.max_y

    def get_distance_ratio(self, x):
        return (max(0, self.peak_ends[0] - x) + max(0, x - self.peak_ends[1]))/(self.peak_ends[1] - self.peak_ends[0])

    def can_be_treated_as_usual(self, x):
        if x < self.a_curve.primary_peak_x:
            y1 = self.a_curve.spline(x)
            y2 = self.a_curve.y[-1]
            diff_ratio = abs(y2 - y1)/self.a_curve.max_y
            ret_bool = diff_ratio < SPECIAL_CIRCUMVENT_LIMIT
        else:
            diff_ratio = None
            ret_bool = False
        return ret_bool, diff_ratio
