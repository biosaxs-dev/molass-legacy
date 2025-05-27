"""
    CurveFeatures.py

        Optimization of mapping between UV-absorbance and Xray-scattering

    Copyright (c) 2019-2025, SAXS Team, KEK-PF
"""
import numpy as np
import logging
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline

PEAK_TOP_WEIGHT     = 1.0
MAJOR_POINT_WEIGHT  = 0.5      # MAJOR_POINT_WEIGHT==OTHER_POINT_WEIGHT for 20190305_2
OTHER_POINT_WEIGHT  = 0.5
FOOT_POINT_WEIGHT   = 0.8

POSITIVE_DERIV_ADOPT_RATIO  = 0.1
NEGATIVE_DERIV_ADOPT_RATIO  = 0.1
VALUE_ADOPT_RATIO           = 0.03
ADD_FOOT_POINTS     = False
FOOT_LEVEL_RATIO    = 0.1
FOOT_ADD_WIDTH      = 10

# TODO: remove duplication of this code
def rotate( th, wx, wy ):
    k = -1 if th > 0 else 0
    cx  = wx[k]
    cy  = wy[k]
    wx_ = wx - cx
    wy_ = wy - cy
    c   = np.cos( th )
    s   = np.sin( th )
    return cx + ( wx_*c - wy_* s ), cy + ( wx_*s + wy_* c )

class CurveFeatures:
    def __init__(self, curve, data_label=""):
        self.logger = logging.getLogger(__name__)
        self.curve = curve

        x = curve.x

        d2y = curve.d2y
        d2_roots = curve.d2_roots
        max_d1 = np.max(curve.d1y)
        min_d1 = np.min(curve.d1y)
        roots = []
        for r in d2_roots:
            ratio = curve.spline(r)/curve.max_y
            if ratio < VALUE_ADOPT_RATIO:
               continue

            d1y_r = curve.d1(r)
            if d1y_r > 0:
                if d1y_r/max_d1 > POSITIVE_DERIV_ADOPT_RATIO:
                    roots.append(r)
            else:
                if abs(d1y_r/min_d1) > NEGATIVE_DERIV_ADOPT_RATIO:
                    roots.append(r)

        if ADD_FOOT_POINTS:
            if len(curve.peak_info) == 1 and curve.has_few_points():
                foot_level = FOOT_LEVEL_RATIO * curve.max_y
                shifted_spline = LSQUnivariateSpline(x, curve.y_for_spline-foot_level, curve.knots[1:-1], ext=3)
                foots = shifted_spline.roots()
                left, top, right = curve.peak_info[0]
                for f in foots:
                    if left - FOOT_ADD_WIDTH < f and f < right + FOOT_ADD_WIDTH:
                        roots.append(f)

        if False:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy._MOLASS.SerialSettings import get_setting
            in_folder = get_setting('in_folder')
            fig = plt.figure(figsize=(8,6))
            ax = fig.gca()
            d = 0.01 if in_folder.find('OA_Ald') >= 0 else 0.02
            yticks = np.arange(0, 3, 0.1) if data_label.find('UV') >= 0 else np.arange(0, 0.3, d)
            ax.yaxis.set_ticks(yticks)
            ax.tick_params(labelsize = 24)
            peak_top = False
            if peak_top:
                point_type = 'Peak Top Points'
            else:
                axt = ax.twinx()
                # axt.tick_params(labelsize = 24)
                axt.set_axis_off()

                point_type = 'Inflection Points'
            ax.set_title('%s %s' % (data_label, point_type) , fontsize=40)
            # ax.plot(x, curve.y, 'o', markersize=3)
            ax.plot(x, curve.y, linewidth=3)
            if peak_top:
                for info in curve.peak_info:
                    top_x = info[1]
                    ax.plot(top_x, curve.y[top_x], 'o', color='red', markersize=10)
            else:
                axt.plot(x, curve.d1y, ':', linewidth=3)
                for r in roots:
                    axt.plot(r, curve.d1(r), 'o', color='red', markersize=10)
                    ax.plot(r, curve.spline(r), 'o', color='yellow', markersize=10)
            fig.tight_layout()
            plt.show()

        if False:
            if len(curve.peak_info) == 1 and len(roots) == 2:
                if curve.half_x is not None and len(curve.half_x) == 2:
                    roots = list(curve.half_x)
                    self.logger.info("inflection points have been replaced with half width points " + str(roots))

        self.roots = roots
        self.weights = np.ones(len(roots))*OTHER_POINT_WEIGHT
        if len(curve.peak_info) == 1:
            left, topx, right = curve.peak_info[0]
            width = right - left
            for k, r in enumerate(roots):
                dist_ratio = abs(r - topx)/width
                # print([k], "dist_ratio=", dist_ratio)
                if dist_ratio > 0.5:
                    self.weights[k] = MAJOR_POINT_WEIGHT

    def get_points(self, adopted_peak_info=None, add_feet=False):
        # add_feet=True made 20181203 worse. why?

        if adopted_peak_info is None:
            adopted_peak_info = self.curve.peak_info
        peak_rec_list = [[ info[1], 1.0 ] for info in adopted_peak_info]
        root_rec_list = [[r, w] for r, w in zip(self.roots, self.weights)]
        if add_feet:
            foot_rec_list = [[f, FOOT_POINT_WEIGHT] for f in self.get_foot_points(adopted_peak_info)]
        else:
            foot_rec_list = []
        records = peak_rec_list + root_rec_list + foot_rec_list
        records_ = sorted(records)
        # print('records_=', records_)
        points = np.array([rec[0] for rec in records_])
        weights = np.array([rec[1] for rec in records_])
        return points, weights

    def get_foot_points(self, adopted_peak_info=None):
        num_peaks = len(adopted_peak_info)

        size = len(self.curve.x)
        start = 0
        prev_top_i = None
        foot_points = []

        for k, prec in enumerate(adopted_peak_info):
            top_i = prec[1]

            left_start = start
            left_stop = top_i

            try:
                left_foot = self.find_foot(-1, left_start, left_stop)
                foot_points.append(left_foot)
            except AssertionError:
                pass

            right_start = top_i
            right_stop = size if k == num_peaks - 1 else self.get_valley_bottom(top_i, adopted_peak_info[k+1][1])

            try:
                right_foot = self.find_foot(+1, right_start, right_stop)
                foot_points.append(right_foot)
            except AssertionError:
                pass

            prev_top_i = top_i
            start = right_stop

        return foot_points

    def get_valley_bottom(self, top_i, next_opt_i):
        n = np.argmin(self.curve.y[top_i:next_opt_i])
        return top_i + n

    def find_foot(self, sign, start, stop):

        curve = self.curve

        slice_ = slice(start, stop)

        x = curve.x[slice_]
        y = curve.y[slice_]

        x1, x2 = x[[0, -1]]
        y1, y2 = y[[0, -1]]

        xscale = 1/abs(x2 - x1)
        yscale = 1/abs(y2 - y1)

        x_ = x * xscale
        y_ = y * yscale

        angle = sign* np.pi/4

        rx, ry = rotate(angle, x_, y_)

        pnn = np.argpartition(ry, 5)
        n = int(np.average(pnn[0:5]) + 0.5)

        if False:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.Elution.CurveUtils import simple_plot
            plt.push()
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
            ax1, ax2 = axes

            fig.suptitle("find_foot ")
            simple_plot(ax1, curve)
            x_ = curve.x[start:stop]
            y_ = curve.y[start:stop]
            ax2.plot(rx, ry)
            ax2.plot(rx[n], ry[n], 'o', color='yellow')
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            plt.show()
            plt.pop()

        return start + n
