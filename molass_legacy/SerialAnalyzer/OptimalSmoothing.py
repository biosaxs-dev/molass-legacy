"""

    OptimalSmoothing.py

    Copyright (c) 2019-2025, SAXS Team, KEK-PF

"""
import numpy as np
from matplotlib.gridspec import GridSpec
from molass_legacy.KekLib.SciPyCookbook import smooth
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.stats import linregress
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Elution.CurveUtils import MAJOR_PEAK_RATIO
from molass_legacy.KekLib.ExceptionTracebacker import log_exception

SMOOTHING_WINDOW_LEN    = 10
MODEST_NOISE_LEVEL_LIM  = 0.01
NUM_POINTS_PER_KNOT     = 10
MIN_NUM_KNOTS           = 50    # > 40 for pH6
MAX_NUM_KNOTS           = 100
NUM_KONTS_LOW_QUALITY   = 20
MIN_NEIGHBOR_DIFF_RATIO = 0.02
USE_REFINED_BEFORE_CHK  = True
MICROFLUIDIC_START_POS  = 0.5
CORRECT_FOR_FEATURE     = False
CORRECT_LIMIT_RATIO     = 0.05
NEGATIVE_D2_LIMIT_RATIO = 0.01      # 
REFINE_DELTA_X_RATIO    = 0.05
DELTA_X_LIMIT_RATIO     = 0.025     # < 0.026  for SUB_TRN1, < 0.0285 for 20180206
DELTA_Y_LIMIT_RATIO     = 0.01      # < 0.0011 for SUB_TRN1 > 0.005 for 20190607_1, > 0.01 for 20170304
SMALL_DIFF = 1e-6

"""
SUB_TRN1 ratios

original
    [4] 80.87661531533368 (0.01135730305678646, 0.06339442040371927) True

analysis copy
    [4] 72.0857775324413 (3.510140665787276e-05, 0.006980129693787952) False
    [6] 81.22860618043339 (0.0011093295770500207, 0.0265100851340219) True

"""

WINDOW_LEN_PEAK_RECOG_MIN = 20
WINDOW_LEN_PEAK_RECOG_MAX = 40
MAX_DENOM_DELTA_R_RATIO = 500

class OptimalSmoothing:
    def __init__(self, x, y, height=None, bottom=None, min_y_ratio=None, orig_top_x=None):
        self.logger = None
        self.x  = x
        self.y  = y
        self.orig_top_x = orig_top_x
        if height is None:
            height = np.max(y) - np.min(y)
        self.height = height

        if bottom is None:
            bottom = np.min(y)

        self.bottom = bottom
        self.delta_x = DELTA_X_LIMIT_RATIO * len(self.x)

        if min_y_ratio is None:
            min_y_ratio = DELTA_Y_LIMIT_RATIO

        self.min_y_ratio = min_y_ratio
        self.delta_y =  min_y_ratio * height

        min_num_knots = len(x)//2 if len(x) < 100 else MIN_NUM_KNOTS
        num_knots = min( MAX_NUM_KNOTS, max( min_num_knots, len(x)//NUM_POINTS_PER_KNOT ) )
        # num_knots = NUM_KONTS_LOW_QUALITY
        self.knots = np.linspace(x[0], x[-1], num_knots+2)

        if False:
            spline = UnivariateSpline(x, y, s=0)
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("OptimalSmoothing.__init__ knots debug")
                ax.plot(x, y)
                ax.plot(self.knots, spline(self.knots), "o")
                fig.tight_layout()
                plt.show()

    def get_logger(self):
        if self.logger is None:
            import logging
            self.logger = logging.getLogger(__name__)

    def plot_variations(self):
        import molass_legacy.KekLib.DebugPlot as plt
        from DataUtils import get_in_folder
        x = self.x
        y = self.y
        in_folder = get_in_folder()

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle("Variations of smoothing window for " + in_folder, fontsize=20)

        gs = GridSpec(3, 4)

        wlen_table = np.array([
            [5,  10, 15, 20],
            [25, 30, 35, 40],
            [50, 70, 100, 200],
            ])

        for i in range(wlen_table.shape[0]):
            for j in range(wlen_table.shape[1]):
                ax = fig.add_subplot(gs[i,j])
                axt = ax.twinx()
                window_len = wlen_table[i, j]

                if window_len > len(y):
                    continue

                ax.set_title("window_len=%d" % window_len)
                ax.plot(x, y)
                sm_y = smooth(y, window_len=window_len)
                ax.plot(x, sm_y)
                spline = UnivariateSpline(x, sm_y, s=0, ext=3)
                d1 = spline.derivative(1)
                d1_y = d1(x)
                axt.plot(x, d1_y, ':', color='green')
                d1_spline = UnivariateSpline(x, d1_y, s=0, ext=3)
                d1_roots = d1_spline.roots()
                d2 = spline.derivative(2)

                peak_x = d1_roots[ np.logical_and( spline(d1_roots) - self.bottom > self.height*0.05, d2(d1_roots) < 0 ) ]
                ax.plot(peak_x, spline(peak_x), 'o', color='red', label='d1_roots')

                feature_spline = LSQUnivariateSpline( x, sm_y, self.knots[1:-1], ext=3 )
                # f_d2 = feature_spline.derivative(2)
                f_d2 = d2
                d2_y = f_d2(x)
                d2_spline = UnivariateSpline(x, d2_y, s=0, ext=3)
                d2_roots = d2_spline.roots()
                d1_max = np.max( np.abs(d1_y) )
                feature_x = d2_roots[ np.logical_and( spline(d2_roots) > self.height*0.03, np.abs(d1(d2_roots))/d1_max > 0.1) ]

                ax.plot(feature_x, spline(feature_x), 'o', color='yellow', label='d2_roots')
                ax.legend()

        fig.tight_layout()
        fig.subplots_adjust( top=0.92 )
        plt.show()

    def compute_optimal_curves(self, microfluidic=False, debug=False):
        x = self.x
        y = self.y

        approximate_peak_num_points = len(y)//8
        num_points = approximate_peak_num_points//4 + 2
        window_len = min(SMOOTHING_WINDOW_LEN, num_points)

        y_for_interpolate = smooth(y, window_len=window_len)
        spline = UnivariateSpline(x, y_for_interpolate, s=0, ext=3)

        noise_level = np.std(y - y_for_interpolate) / self.height

        self.noise_level = noise_level
        self.spline_y = y_for_interpolate
        self.spline = spline

        window_len = min(WINDOW_LEN_PEAK_RECOG_MAX, max(WINDOW_LEN_PEAK_RECOG_MIN, approximate_peak_num_points//2))
        # print('noise_level=', noise_level, 'window_len=', window_len)
        average_left_y = np.average(y[0:approximate_peak_num_points])
        average_right_y = np.average(y[-approximate_peak_num_points:])
        end_diff_ratio = abs(average_left_y - average_right_y)/self.height

        if CORRECT_FOR_FEATURE:
            if end_diff_ratio > CORRECT_LIMIT_RATIO:
                to_be_smoothed_y = self.baseline_correct_simply(x, y)
            else:
                to_be_smoothed_y = y
        else:
            to_be_smoothed_y = y

        y_for_peak_recog = smooth(to_be_smoothed_y, window_len=window_len)
        try:
            spline_peak_recog = LSQUnivariateSpline( x, y_for_peak_recog, self.knots[1:-1], ext=3 )
        except Exception as exc:
            self.get_logger()
            log_exception(self.logger, "LSQUnivariateSpline failed, knots=%s : " % str(self.knots) )
            if False:
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.set_title("compute_optimal_curves LSQUnivariateSpline debug")
                    ax.plot(x, y_for_peak_recog)
                    ax.plot(self.knots[1:-1], spline(self.knots[1:-1]), "o")
                    fig.tight_layout()
                    plt.show()
            raise exc
        # spline_peak_recog = UnivariateSpline(x, y_for_peak_recog, s=0, ext=3)

        self.feature_y = y_for_peak_recog
        self.spline_peak_recog = spline_peak_recog

        d1 = spline_peak_recog.derivative(1)
        d1y = d1(x)
        d1_spline = UnivariateSpline(x, d1y, s=0, ext=3)
        self.d1 = d1_spline
        self.d1y = d1y
        self.d1_roots = d1_spline.roots()

        d2 = spline_peak_recog.derivative(2)
        d2y = d2(x)
        d2_spline = UnivariateSpline(x, d2y, s=0, ext=3)
        self.d2 = d2_spline
        self.d2y = d2y
        self.d2_roots = d2_spline.roots()

        if microfluidic:
            start_pos = int(MICROFLUIDIC_START_POS * x[-1])
            n = np.argmax(y_for_interpolate[start_pos:])
            self.top_x = np.array([start_pos + n])
            return

        # print('orig_top_x=', self.orig_top_x)
        # debug = self.orig_top_x is not None
        try:
            top_cadidates = self.get_good_top_candidates(debug=debug)
        except:
            import inspect
            for frm in inspect.stack()[1:]:
                print("---- : %s %s (%d)" % (frm.filename, frm.function, frm.lineno))
            self.get_logger()
            log_exception(self.logger, "get_good_top_candidates failed: ", n=10)
            top_cadidates = []
        if len(top_cadidates) == 0:
            top_cadidates = self.get_original_top_x()
        # print('top_cadidates=', top_cadidates)
        # assert len(top_cadidates) > 0

        sd1 = spline.derivative(1)
        sd1y = sd1(x)
        sd1_spline = UnivariateSpline(x, sd1y, s=0, ext=3)
        sd1_roots = sd1_spline.roots()
        refine_delta_x = REFINE_DELTA_X_RATIO * len(self.x)

        def get_refined_top_points_impl(tobe_refined):
            pk = 5
            refined = []
            last_refined = None
            for p in tobe_refined:
                dist = np.abs(sd1_roots - p)
                neighbor_roots_ = sd1_roots[dist < refine_delta_x]
                pk = min(5, len(neighbor_roots_)-1)
                kk = np.argpartition(np.abs(neighbor_roots_ - p), pk)[0:pk+1]
                if len(kk) > 0:
                    n = np.argmax( self.spline(neighbor_roots_[kk]) )
                    refined_x = neighbor_roots_[kk[n]]
                else:
                    # is this right?
                    refined_x = p
                # print([n], refined_x)
                if last_refined is None or abs(refined_x - last_refined) > SMALL_DIFF:
                    # this check is required for 20180617,
                    # where [4] 158.38825643054835 occurred twice consecutively
                    refined.append(refined_x)
                    last_refined = refined_x

                # if abs(p - refined_x) > 5:
                if debug:
                    import molass_legacy.KekLib.DebugPlot as plt
                    print([p], tobe_refined)
                    x = self.x
                    y = self.y
                    sy = self.spline_y
                    with plt.Dp():
                        fig, ax = plt.subplots()
                        ax.set_title("get_refined_top_points debug")
                        ax.plot(x, y, label='data')
                        ax.plot(x, sy, label='spline')
                        ax.plot(neighbor_roots_, self.spline(neighbor_roots_), 'o', label='neighbor_roots_' )
                        kk_x = neighbor_roots_[kk]
                        ax.plot(kk_x, self.spline(kk_x), 'o', label='neighbor_roots_[kk]' )
                        ax.plot(p, self.spline(p), 'o', color='yellow', label='p' )
                        ax.plot(refined_x, self.spline(refined_x), 'o', color='red', label='refined_x' )
                        ax.legend()
                        plt.show()

            return np.array(refined)

        def get_refined_top_points(tobe_refined):
            try:
                return get_refined_top_points_impl(tobe_refined)
            except:
                # ValueError: attempt to get argmax of an empty sequence at
                # n = np.argmax( self.spline(neighbor_roots_[kk]) )
                log_exception(self.logger, "get_refined_top_points failed: ")
                return tobe_refined

        if USE_REFINED_BEFORE_CHK:
            # necessary for 20180329_microfluidic
            refined_cadidates = get_refined_top_points(top_cadidates)
            top_x = refined_cadidates

        else:
            top_x = top_cadidates

        if len(top_x) == 0:
            try:
                # as in 20181203
                n = np.argmax(spline(self.d1_roots))
                top_x = np.array([self.d1_roots[n]])
                self.get_logger()
                self.logger.warning("top_x has been made as [self.d1_roots[%d]] in an emergency measure.", n)
            except:
                pass

        if debug or len(top_x) == 0:
            import inspect
            for frm in inspect.stack()[1:]:
                print("---- : %s %s (%d)" % (frm.filename, frm.function, frm.lineno))

            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.Tools.SpotDebugger import SpotDebugger
            debugger = SpotDebugger(None, (self,))
            debugger.show()

            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
                fig.suptitle("No peak debug in " + __name__)
                ax1.plot(x, y)
                axt = ax2.twinx()
                ax2.plot(x, y, label='data')
                ax2.plot(x, self.spline_y, label='spline_y')
                ax2.plot(x, self.feature_y, label='feature_y')
                axt.plot(x, d1y, ':', label='d1y')
                axt.plot(x, d2y, ':', label='d2y')
                ax2.plot(self.d1_roots, spline(self.d1_roots), 'o', color='yellow', markersize=10, label='d1_roots')
                ax2.plot(sd1_roots, spline(sd1_roots), 'o', color='pink', label='sd1_roots')
                ax2.plot(top_cadidates, spline(top_cadidates), 'o', label='top_cadidates')
                ax2.plot(refined_cadidates, spline(refined_cadidates), 'o', label='refined_cadidates')
                # ax2.legend(loc='upper left')
                ax2.legend(loc='center right')
                axt.legend(loc='upper right')
                fig.tight_layout()
                plt.show()

        if len(top_x) == 0:
            raise RuntimeError("No peak")

        refined_x = get_refined_top_points(top_x)

        if debug:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("compute_optimal_curves debug")
                ax.plot(x, self.spline(x))
                ax.plot(self.d1_roots, self.spline(self.d1_roots), 'o', color='yellow', markersize=15, alpha=0.5, label='d1_roots')
                ax.plot(top_cadidates, self.spline(top_cadidates), 'o', color='cyan', label='top_cadidates')
                if len(top_x) > 0:
                    ax.plot(top_x, self.spline(top_x), 'o', color='pink', label='top_x')
                    ax.plot(refined_x, self.spline(refined_x), 'o', color='red', label='refined_x')
                ax.legend()
                plt.show()

        self.top_x = np.array(refined_x)

    def baseline_correct_simply(self, x, y):
        k30 = int(len(y)*0.3)
        ppp = np.argpartition(y, k30)
        p30 = ppp[0:k30]
        x_ = x[p30]
        y_ = y[p30]
        slope, intercept, r_value, p_value, std_err = linregress( x_, y_ )
        corrected_y = y - (slope*x + intercept)
        if False:
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(x, y)
            ax.plot(x, corrected_y)
            ax.plot(x, slope*x + intercept, ':', color='red')
            plt.show()
        return corrected_y

    def get_original_top_x(self):
        ret_candidates = []
        if self.orig_top_x is None:
            pass
        else:
            refine_delta_x = REFINE_DELTA_X_RATIO * len(self.x)
            c_roots = self.d1_roots
            ret_candidates = []
            for cr in c_roots:
                dist = np.min(np.abs(self.orig_top_x - cr))
                if dist < refine_delta_x:
                    ret_candidates.append(cr)

            self.get_logger()
            self.logger.warning("top candidates %s have been retrieved from the original." % (str(ret_candidates)))
            if False:
                print('self.orig_top_x=', self.orig_top_x, 'refine_delta_x=', refine_delta_x)
                print('c_roots=', c_roots)
                print('ret_candidates=', ret_candidates)

        return np.array(ret_candidates)

    def get_good_top_candidates(self, debug=False):
        tall_enough = self.spline(self.d1_roots) - self.bottom > self.height*MAJOR_PEAK_RATIO
        max_d2 = np.max(np.abs(self.d2(self.d1_roots)))
        d2_is_well_negative = self.d2(self.d1_roots) < -max_d2 * NEGATIVE_D2_LIMIT_RATIO
        if False:
            print('d1_roots=', self.d1_roots)
            print('max_d2=', max_d2, max_d2 * NEGATIVE_D2_LIMIT_RATIO)
            print('d2(self.d1_roots=', d2(self.d1_roots))

        if debug:
            print('-------------- get_good_top_candidates begin')

        d1_roots = self.d1_roots
        num_roots = len(d1_roots)
        if num_roots < 2:
            return d1_roots

        default_delta_x = self.delta_x

        index = np.arange(num_roots)
        cadidates = np.logical_and( tall_enough, d2_is_well_negative )

        ret_candidates = []

        for k, root_x in enumerate(d1_roots):
            if not cadidates[k]:
                continue

            prev_r = d1_roots[k-1 if k > 0 else k+1]
            next_r = d1_roots[k+1 if k < num_roots - 1 else k-1]
            delta_r = min( abs(root_x - prev_r),  abs(next_r - root_x) )
            delta_r_ratio = delta_r/min(MAX_DENOM_DELTA_R_RATIO, len(self.x))

            prev_x = root_x - delta_r
            next_x = root_x + delta_r

            prev_y = self.spline_peak_recog(prev_x)
            next_y = self.spline_peak_recog(next_x)
            root_y = self.spline_peak_recog(root_x)

            delta_y = min(root_y - prev_y, root_y - next_y)
            delta_y_ratio = delta_y/self.height

            adopt = False
            if delta_y_ratio > self.min_y_ratio and delta_r_ratio > DELTA_X_LIMIT_RATIO:
                ret_candidates.append(root_x)
                adopt = True

            if debug:
                print([k], root_x, (delta_y_ratio, delta_r_ratio), adopt)

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            print('-------------- get_good_top_candidates end')
            x = self.x
            y = self.y
            spline_pr = self.spline_peak_recog

            plt.push()
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(x, y)
            ax.plot(x, self.spline_peak_recog(x))
            ax.plot(self.d1_roots, spline_pr(self.d1_roots), 'o', label='d1_roots')
            ax.plot(ret_candidates, spline_pr(ret_candidates), 'o', label='ret_candidates')
            ax.legend()
            fig.tight_layout()
            plt.show()
            plt.pop()

        return np.array(ret_candidates)

    def get_halfmax_points_of_a_peak(self, k):
        half_h = (self.top_x[k] - self.bottom)/2
        # ...
