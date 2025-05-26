"""
    ElutionCurve.py

        recognition of peaks

    Copyright (c) 2018-2025, SAXS Team, KEK-PF
"""
import copy
import logging
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Elution.CurveUtils import proof_plot, simple_plot
from .CurveFeatures import CurveFeatures
from .OptimalSmoothing import OptimalSmoothing
from molass_legacy.DataStructure.PeakInfo import PeakInfo
from molass_legacy.DataStructure.AnalysisRangeInfo import PairedRange
from molass_legacy.Irregular.Preprocess import correct_flat_negative_baseline

DEBUG   = False

TOP_X_ALLOWANCE = 5
NUM_POINTS_PER_KNOT     = 10
MIN_NUM_KNOTS           = 50    # > 40 for pH6
MAX_NUM_KNOTS           = 100
NUM_KNOTS_MID_QUALITY   = 40
QRATIO_LIMIT            = 0.4   # > 0.273 for pH6, > 0.32 for 20170426
DETERMINE_KNOTS_ONCE    = False # must be False for Factin
NUM_KONTS_LOW_QUALITY   = 20    # < 25 for BL-10C/Ald
PEAK_INFO_FIND_RATIO    = 0.2
ANALYSIS_RANGE_RATIO    = 0.5
BETTER_RANGE_RATIO      = 0.1
SUFFICIENT_PEAK_H_RATIO = 0.3
MININUM_PEAK_WIDTH      = 10
MININUM_PEAK_W_RATIO    = 0.05
NEIGHBOR_IGNORE_RATIO   = 0.7
LOW_PEAK_CHECK_RATIO    = 0.7   # > 0.3 for 20170304_UV, > 0.59 for Factin
LOW_PEAK_CHECK_LIMIT    = 0.2   #
LOW_PEAK_RAISE_RATIO    = 1.02  # to allow the first low peak in Sugiyama
BETER_SPLINE_LIM_PNTS   = 20
BETER_SPLINE_LIM_WIDTH  = 0.5
MODEST_NOISE_LEVEL_LIM  = 0.01
FOOT_LEVEL_RATIO        = 0.1
FOOT_ADD_WIDTH          = 200
USE_HALF_MAXIMUM_POINTS = False
MAJOR_PEAK_RATIO        = 0.1
SEDIMENTATION_LIMIT = 0.03
END_DIFF_WIDTH = 50
END_DIFF_MIN_HEIGHT = SEDIMENTATION_LIMIT/2
RELIABLE_SIGMA_RATIO = 2
RELIABLE_ALPHA = 0.03
USE_RELIABLE_ALPHA = True
MIN_HALF_WIDTH = 5

class ElutionCurve:
    def __init__( self, y,
                    x=None,
                    max_y=None,
                    j0=0,
                    microfluidic=None,
                    unify_close_peaks=True,     # need be True for Factin
                    low_quality=False,
                    min_y_ratio=None,
                    debug_plot=False,
                    debug_smooth=False,
                    add_features=True,
                    cover_curve=None,
                    orig_top_x=None,
                    delay_emg_peaks=False,      # True for residual curves in ElutionDecomposer
                    possiblly_peakless=False ):

        self.logger = logging.getLogger( __name__ )
        if x is None:
            x = np.arange( len(y) )
        self.x  = x
        self.y  = correct_flat_negative_baseline(y)
        self.y_orig = y     # for backward compatibility
        m = np.argmax(y)
        self.max_x = x[m]   # task: rename to a better name
        self.max_y = y[m]
        self.min_peak_width = len(self.x) * MININUM_PEAK_W_RATIO
        self.min_y = np.percentile(y, 5)
        self.height = self.max_y - self.min_y
        self.j0 = j0
        self.mean = None
        self.variance = None
        self.peak_slice = slice(0, len(x))      # comptibility with PeakCurve
        self.peak_region_sigma = None
        self.end_slices = None

        self.low_quality = low_quality
        self.debug_plot = debug_plot

        if microfluidic is None:
            microfluidic = get_setting('use_mtd_conc')
        self.microfluidic = microfluidic

        self.smoothing = OptimalSmoothing(x, y, self.height, self.min_y, min_y_ratio=min_y_ratio, orig_top_x=orig_top_x)
        self.smoothing.compute_optimal_curves(microfluidic, debug_smooth)

        self.y_for_spline = self.smoothing.spline_y
        self.feature_y = self.smoothing.feature_y
        self.spline = self.smoothing.spline
        self.sy = self.spline(x)

        self.d1 = self.smoothing.d1
        self.d1y = self.d1(x)
        self.d2 = self.smoothing.d2
        self.d2y = self.d2(x)
        self.d2_roots = self.smoothing.d2_roots

        self.peak_top_x = self.get_significant_peaks( low_quality=low_quality, unify_close_peaks=unify_close_peaks )
        if len(self.peak_top_x ) == 0:
            raise RuntimeError("No peak_top_x")

        self.set_peak_dependent_info()
        self.emg_peaks = None
        # assure consistency between self.emg_peaks and peak_info
        self.possiblly_peakless = possiblly_peakless
        if delay_emg_peaks:
            self.set_primary_info()
        else:
            if possiblly_peakless:
                self.emg_peaks = []
                self.set_primary_info()
            else:
                self.get_emg_peaks(logger=self.logger)

        if USE_HALF_MAXIMUM_POINTS:
            if len(self.peak_info) == 1:
                half_spline = LSQUnivariateSpline( x, self.y_for_spline - self.max_y/2, self.knots[1:-1], ext=3 )
                self.half_x = half_spline.roots()
            else:
                self.half_x = None

        if add_features:
            try:
                self.feature = CurveFeatures(self)
            except:
                from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
                etb = ExceptionTracebacker()
                print(etb)

        self.editor_ranges = None
        self.paired_ranges = None

        if debug_plot:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.KekLib.DebugPlot import get_parent
            fig = plt.figure( figsize=(16, 8) )
            proof_plot(self, get_parent(), fig)
            plt.show()

    def get_xy(self):
        # for forward compatibility
        return self.x, self.y

    def set_peak_dependent_info(self, cover_curve=None):
        # print('initial peak_top_x=', self.peak_top_x)
        work_boundaries = self.get_peak_boundaries()
        # print('work_boundaries=', work_boundaries)

        ranges, ptxes, valid_boundaries = self.get_ranges_by_ratio( PEAK_INFO_FIND_RATIO, boundaries=work_boundaries, return_ptxes=True)
        covered = False
        if len(ranges) == 0:
            if cover_curve is None:
                raise RuntimeError("No ranges")
            else:
                # as with Factin at wavelength=400
                # use the given cover curve to cover peak related info
                ranges = cover_curve.peak_info
                ptxes  = cover_curve.peak_top_x
                valid_boundaries = cover_curve.boundaries
                covered = True
                self.logger.warning("created an elution curve using a cover curve.")

        self.peak_top_x = ptxes
        # print('final peak_top_x=', self.peak_top_x)
        # print('valid_boundaries=', valid_boundaries)
        self.boundaries = valid_boundaries
        # self.boundaries = self.get_range_boudaries(ranges, valid_boundaries)

        # ------- backward compatibility attributes
        if not covered:
            # buggy for BSA_201
            # task: re-consider this
            self.modify_ranges_to_avoid_too_narrow_valley( ranges )
        # print('final ranges=', ranges)

        self.peak_info  = ranges
        if len(self.peak_info) == 0:
            from molass_legacy.KekLib.DebugPlot import get_parent
            fig = plt.figure( figsize=(16, 8) )
            proof_plot(self, get_parent(), fig)
            plt.show()
        assert len(self.peak_info) > 0
        self.set_major_infos()

    def get_emg_peaks(self, **kwargs):
        if self.emg_peaks is None:
            if False:
                from importlib import reload
                import DataStructure.EmgPeak
                reload(DataStructure.EmgPeak)
            from molass_legacy.DataStructure.EmgPeak import get_peaks
            self.emg_peaks = get_peaks(self, **kwargs)

        if len(self.emg_peaks) == 0 or self.possiblly_peakless:
            # case for UV elution at a longer wave length
            # len(self.emg_peaks) == 0 occurs for pH6
            self.set_primary_info()
            return self.emg_peaks

        # print('self.emg_peaks=', self.emg_peaks)
        # print('self.peak_info=', self.peak_info)

        # the following treatment is required for cases such as 20170309
        if len(self.emg_peaks) < len(self.peak_info):
            new_peak_top_x = []
            new_peak_info = []
            for k, peak in enumerate(self.peak_info):
                mid_x = peak[1]
                found = False
                for epeak in self.emg_peaks:
                    if abs(mid_x - epeak.top_x) < TOP_X_ALLOWANCE:
                        found = True
                        new_peak_top_x.append(self.peak_top_x[k])
                        new_peak_info.append(peak)
                        break
                if not found:
                    self.logger.info("peak %s without any corresponding emg_peak has been ignored.", str(peak))
            self.peak_top_x = np.array(new_peak_top_x)
            self.peak_info = new_peak_info
            self.boundaries = self.get_peak_boundaries()
        elif len(self.emg_peaks) > len(self.peak_info):
            new_emg_peaks = []
            for epeak in self.emg_peaks:
                top_x = epeak.top_x
                found = False
                for peak in self.peak_info:
                    if abs(top_x - peak[1]) < TOP_X_ALLOWANCE:
                        found = True
                        new_emg_peaks.append(epeak)
                        break
                if not found:
                    self.logger.info("emg_peak %s without any corresponding peak has been ignored.", str(epeak))
            self.emg_peaks = new_emg_peaks

        assert len(self.emg_peaks) == len(self.peak_info)

        self.set_primary_info()
        return self.emg_peaks

    def set_primary_info(self):
        primary_peak_no = None
        primary_peak    = None
        primary_peak_y  = None
        for i, info in enumerate(self.peak_info):
            px  = info[1]
            py  = self.spline( px )
            if primary_peak_y is None or py > primary_peak_y:
                primary_peak_no = i
                primary_peak    = px
                primary_peak_y  = py

        self.primary_peak_no    = primary_peak_no
        self.primary_peak_x     = primary_peak
        self.primary_peak_i     = None if primary_peak is None else int(primary_peak + 0.5)
        self.primary_peak_y     = primary_peak_y

        num_points = self.get_primary_peak_num_points()

        if False:

            if num_points is not None and num_points < BETER_SPLINE_LIM_PNTS:
                w = np.min(np.abs( self.knots - self.peak_top_x ))
                if w > BETER_SPLINE_LIM_WIDTH:
                    try:
                        top_x = self.peak_top_x[self.primary_peak_no]
                        lower_knots = self.knots[self.knots < top_x]
                        lower_knots[-1] = (lower_knots[-2] + top_x)/2
                        upper_knots = self.knots[self.knots > top_x]
                        upper_knots[0]  = (top_x + upper_knots[1])/2
                        self.knots = np.hstack( [lower_knots, [top_x], upper_knots] )
                        self.spline = LSQUnivariateSpline( x, self.y_for_spline, self.knots[1:-1], ext=3 )   # ext=3 is required for pH6
                        self.logger.info( "spline has been re-computed for this elution curve due to the small number(%d) of peak points" % num_points )
                    except:
                        self.logger.info("failed to get a better spline")

        # self.rich_peak_info = self.make_rich_peak_info(ranges)

    def get_primarypeak_x(self):
        return self.max_x

    def get_primarypeak_i(self):
        return int(round(self.max_x - self.x[0]))

    def get_primarypeak_i(self):
        return self.primary_peak_i

    def get_height_ratio(self, pos):
        return (self.spline(pos) - self.min_y) / self.height

    def get_major_valley_bottoms(self, major_peak_tops):
        bottoms = np.array(self.boundaries)
        ret_bottoms = []
        for k, top in enumerate(major_peak_tops):
            if k+1 < len(major_peak_tops):
                next_top = major_peak_tops[k+1]
                b = bottoms[np.logical_and(bottoms > top, bottoms < next_top)]
                # len(b) == 0 occured for 20181203 in FullOptInit
                if len(b) > 0:
                    ret_bottoms.append(b[0])

        return np.array(ret_bottoms)

    def set_major_infos(self):
        peak_rec_list = []
        peak_top_list = []
        for k, rec in enumerate(self.peak_info):
            if self.get_height_ratio(rec[1]) > MAJOR_PEAK_RATIO:
                peak_rec_list.append(rec)
                peak_top_list.append(self.peak_top_x[k])
        self.major_peak_info = peak_rec_list
        self.major_peak_top_x = np.array(peak_top_list)
        self.major_boundaries = self.get_major_valley_bottoms(self.major_peak_top_x)

    def get_major_peak_info(self, ratio=None, add_j0=False):
        if ratio is None:
            if add_j0:
                ret_info = [ [self.j0+i for i in rec] for rec in self.major_peak_info ]
            else:
                ret_info = self.major_peak_info
            return ret_info

        assert not add_j0

        peak_rec_list = []
        for k, rec in enumerate(self.peak_info):
            if self.get_height_ratio(rec[1]) > ratio:
                peak_rec_list.append(rec)

        return peak_rec_list

    def get_primary_peak_num_points(self):
        try:
            lower, center, upper = self.peak_info[self.primary_peak_no]
            num_points = upper - lower + 1
        except:
            num_points = None
        self.primary_peak_num_points = num_points
        return num_points

    def get_primary_peak_info(self):
        return self.peak_info[self.primary_peak_no]

    def has_few_points(self):
        return self.primary_peak_num_points < BETER_SPLINE_LIM_PNTS

    def determine_knots( self, i, x, low_quality=False ):
        if low_quality:
            num_nkots = NUM_KONTS_LOW_QUALITY
        else:
            if i == 0:
                num_nkots = min( MAX_NUM_KNOTS, max( MIN_NUM_KNOTS, len(x)//NUM_POINTS_PER_KNOT ) )
            else:
                num_nkots = NUM_KNOTS_MID_QUALITY
        return np.linspace( 0, len(x), num_nkots + 2 )

    def is_ignorable_peak(self, y):
        return y < self.max_y * PEAK_INFO_FIND_RATIO

    def get_significant_peaks( self, low_quality=False, unify_close_peaks=False ):
        self.roots = roots_ = self.smoothing.top_x
        return self.roots

    def low_peak_stands_out_clearly( self, root, peak_y, min_root_diff ):
        ratio = self.get_low_peak_stand_out_ratio( root, peak_y, min_root_diff )
        if ratio < LOW_PEAK_CHECK_LIMIT:
            return True

        ratio_w = self.get_low_peak_stand_out_ratio( root, peak_y, min_root_diff * 2 )
        improved_ratio = ratio_w/ratio
        # print( 'improved_ratio=', improved_ratio )

        return improved_ratio < 0.7

    def get_low_peak_stand_out_ratio( self, root, peak_y, min_root_diff ):
        peak_y_ = peak_y * LOW_PEAK_RAISE_RATIO     # to allow low peaks as in Sugiyama

        start   = int( max(0, root - min_root_diff ) + 0.5 )
        stop    = int( min(len(self.x), root + min_root_diff ) + 0.5 )
        slice_  = slice( start, stop )
        y_  = self.y[slice_]
        ratio   = len( np.where( y_ > peak_y_ )[0] ) / (stop - start)
        if False:
            print( 'exceed ratio=', ratio )
            x_  = self.x[slice_]
            plt.plot( x_, y_ )
            plt.plot( x_, self.spline( x_ ) )
            ppp = np.where( y_ > peak_y_ )[0]
            plt.plot( x_[ppp], y_[ppp], 'o', color='yellow' )
            plt.plot( root, peak_y, 'o', color='red' )
            plt.show()
        return ratio

    def get_peak_boundaries(self, peak_top_x=None, debug=False):

        if peak_top_x is None:
            peak_top_x = self.peak_top_x

        last_ptx = None
        ret_boundaries = []
        if len(peak_top_x) == 1:
            # temporary fix for PgpTak in V2
            return ret_boundaries

        print('get_peak_boundaries: peak_top_x=', peak_top_x)
        for ptx in peak_top_x:
            if last_ptx is not None:
                start   = int(last_ptx)
                stop    = int(ptx)
                slice_ = slice( start, stop )
                x_  = self.x[slice_]
                y_  = self.spline(x_)
                n   = np.argmin( y_ )
                try:
                    kth = min(6, int( len(y_) * 0.1 ))
                    y_kth = np.argpartition( y_, kth )
                    n_  = int( np.average( y_kth[0:kth] ) + 0.5 )
                    y_lim = y_[n]*1.3
                    if y_lim > 0 and y_[n_] < y_lim:
                        # to avoid cases like 20161006_quality
                        n   = n_
                    print('try success: kth=', kth, 'y_kth[0:kth]=', y_kth[0:kth], 'n_=', n_)
                except:
                    # nothing to do?
                    pass

                ret_boundaries.append( start + n )

            last_ptx = ptx

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            x = self.x
            y = self.y
            with plt.Dp():
                fig = plt.figure()
                ax = fig.gca()
                ax.set_title("get_peak_boundaries debug")
                ax.plot(x, y, label='data')
                ax.plot(x, self.spline(x), label='spline')
                ax.plot(peak_top_x, self.spline(peak_top_x), 'o', color='red', label='peak top')
                ax.plot(ret_boundaries, self.spline(ret_boundaries), 'o', color='green', label='boundaries')
                ax.legend()
                fig.tight_layout()
                plt.show()

        if len(peak_top_x) > 0 and len(ret_boundaries) != len(peak_top_x) - 1:
            print('ret_boundaries=', ret_boundaries)
            print('peak_top_x=', peak_top_x)
            assert len(ret_boundaries) == len(peak_top_x) - 1

        return ret_boundaries

    def get_lower_from_y(self, start, pti, yr, debug=False):
        try:
            # y_ = self.sy[start:pti] if self.low_quality else self.y[start:pti]
            y_ = self.sy[start:pti]
            lower0  = start + np.where( y_ < yr )[0][-1]
        except:
            # in case there exists no such points
            lower0  = start

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            print("pti=", pti)
            x = self.x
            y = self.y
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("get_lower_from_y")
                ax.plot(x, y)
                ax.plot(x, self.sy, ":")
                xmin, xmax = ax.get_xlim()
                ax.set_xlim(xmin, xmax)
                ax.plot([xmin, xmax], [yr, yr], color="yellow")
                for j in [start, pti]:
                    px = x[j]
                    ax.plot(px, self.spline(px), "o")
                x_ = x[lower0]
                ax.plot(x_, self.spline(x_), "o", color="red")
                fig.tight_layout()
                plt.show()

        return lower0

    def get_lower_from_d1y( self, start, pti ):
        try:
            lower1  = start + np.where( self.d1y[start:pti] < 0 )[0][-1]
        except:
            # in case there exists no such points
            lower1  = start
        return lower1

    def get_upper_from_y(self, stop, pti, yr, debug=False):
        try:
            # y_ = self.sy[pti:stop] if self.low_quality else self.y[pti:stop]
            y_ = self.sy[pti:stop]
            upper0  = pti + np.where( y_ < yr )[0][0]
        except:
            # in case there exists no such points
            upper0  = stop - 1
        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            x = self.x
            y = self.y
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("get_upper_from_y")
                ax.plot(x, y)
                ax.plot(x, self.sy, ":")
                xmin, xmax = ax.get_xlim()
                ax.set_xlim(xmin, xmax)
                ax.plot([xmin, xmax], [yr, yr], color="yellow")
                for p in [stop-1, pti]:
                    px = x[p]
                    ax.plot(px, self.spline(px), "o")
                x_ = x[upper0]
                ax.plot(x_, self.spline(x_), "o", color="red")
                fig.tight_layout()
                plt.show()
        return upper0

    def get_upper_from_d1y( self, stop, pti ):
        try:
            # let slice start at pti+1 to avoid taking pti as upper1
            upper1  = pti + np.where( self.d1y[pti+1:stop] > 0 )[0][0]
        except:
            # in case there exists no such points
            upper1  = stop - 1
        return upper1

    def get_default_editor_ranges(self):
        if self.editor_ranges is None:
            ranges = []
            for prec, epeak in zip(self.peak_info, self.emg_peaks):
                top = prec[1]
                left, right = epeak.get_model_x_from_ratio(ANALYSIS_RANGE_RATIO)
                left_ = max(0, int(round(left)))
                right_ = min(len(self.x)-1, int(round(right)))
                if left_ < top and top < right_ and right_ - left_> MIN_HALF_WIDTH*2:
                    range_ = [[left_, top], [top, right_]]
                else:
                    left_c = top - MIN_HALF_WIDTH
                    right_c = top + MIN_HALF_WIDTH
                    self.logger.warning("illegal (or too narrow) range values (%d, %d, %d) corrected to (%d, %d) as a temporary fix.", left_, top, right_, left_c, right_c)
                    range_  = [[left_c, right_c]]
                ranges.append(range_)
            self.editor_ranges = ranges
        return self.editor_ranges

    def get_default_paired_ranges(self):
        if self.paired_ranges is None:
            paired_ranges = []
            for k, erec in enumerate(self.get_default_editor_ranges()):
                left, top = erec[0]
                pinfo = PeakInfo(k, top)
                paired_ranges.append(PairedRange(pinfo, *erec))     # erec = [ [left, top], [top, right] ]
            self.paired_ranges = paired_ranges
        return self.paired_ranges

    def get_ranges_by_ratio( self, ratio, boundaries=None, return_ptxes = False, debug=False ):
        if boundaries is None:
            boundaries = self.boundaries
            regular = True
        else:
            regular = False

        # boundaries_ should be integers. why are non-integers coming in?
        boundaries_ = np.array(boundaries)

        # debug = get_setting("local_debug")

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            x = self.x
            y = self.y
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("get_ranges_by_ratio begin")
                ax.plot(x, y)
                for b in boundaries_:
                    ax.plot(b, self.spline(b), "o", color="green")
                fig.tight_layout()
                plt.show()

        ranges = []
        ptxes = []
        valid_boundaries = []
        start = 0

        for k, ptx in enumerate(self.peak_top_x):
            pti = int(ptx - self.x[0] + 0.5)

            if k < len(boundaries_):
                if regular:
                    stop = int(boundaries_[k])
                else:
                    try:
                        stop = int(boundaries_[boundaries_ > ptx][0])
                    except:
                        stop = len(self.y)
            else:
                stop = len(self.y)

            yr  = self.spline(ptx) * ratio
            lower   = self.get_lower_from_y( start, pti, yr, debug=debug )
            upper   = self.get_upper_from_y( stop, pti, yr, debug=debug )
            width   = upper - lower

            if debug:
                print("peak_top_x=", self.peak_top_x)
                print("boundaries_=", boundaries_)
                print("[lower, pti, upper]=",  [lower, pti, upper] )
                x = self.x
                y = self.y
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.set_title("[%d] range" % k)
                    ax.plot(x, y)
                    for p in [lower, upper]:
                        px = x[p]
                        ax.plot(px, self.spline(px), "o")
                    fig.tight_layout()
                    plt.show()

            if debug and (lower >= pti or pti >= upper):
                print('regular=', regular)
                print('boundaries=', boundaries)
                print('start=', start, 'stop=', stop)
                print( 'width=', width, 'min_peak_width=', self.min_peak_width )
                fig = plt.figure()
                ax = fig.gca()
                ax.cla()
                ax.set_title('[%d] ptx=%g' % (k, ptx))
                ax.plot( self.x, self.y )
                top_x = self.peak_top_x
                ax.plot(top_x, self.spline(top_x), 'o', color='orange' )

                b = boundaries
                if b is not None and len(b) > 0:
                    ax.plot(self.x[b], self.y[b], 'o', color='green' )

                colors = ['yellow', 'red', 'cyan']
                for j, x_ in enumerate([lower, ptx, upper]):
                    c = colors[j]
                    ax.plot( x_, self.spline(x_), 'o', color=c )
                plt.show()

            height_ratio = self.spline(ptx)/self.max_y
            def major_height_ratio_ok():
                top_x = (lower + upper)//2
                dw = width//3
                ratio = np.average(self.y[top_x-dw:top_x+dw])/self.max_y
                self.logger.info("adopted a peak at %d with average_ratio=%.3g", top_x, ratio)
                return ratio > 0.1

            if height_ratio > SUFFICIENT_PEAK_H_RATIO or width >= self.min_peak_width or major_height_ratio_ok():
                if (    lower >= 0 and lower <= pti
                    and pti <= upper and upper < len(self.x)
                    ):
                    if len(ranges) > 0:
                        last_upper = ranges[-1][2]
                        if lower - last_upper < 2:
                            # as in 20191118_3
                            last_range = copy.deepcopy(ranges[-1])
                            ranges[-1][2] = last_upper - 1
                            self.logger.warning("ranges modified as a temporary fix from %s to %s, %s to %s",
                                                str(last_range), str(ranges[-1]), str([lower, pti, upper]), str([lower+1, pti, upper]))
                            lower += 1
                    ranges.append( [lower, pti, upper] )
                    ptxes.append( ptx )
                    if len(ranges) > 1:
                        valid_boundaries.append(start)  # i.e., start=last stop
                else:
                    self.logger.warning("ignored irregular range " + str([lower, pti, upper]) + " in get_ranges_by_ratio")
            else:
                if width < 0:
                    self.logger.warning("negative peak width %d = %d - %d" % (width, upper, lower))
                else:
                    self.logger.warning("too small peak width %d = %d - %d" % (width, upper, lower))

                # if len(valid_boundaries) > 0:
                #    valid_boundaries.pop()

            start   = stop

        if debug:
            print("---- ret ranges=", ranges)
            print("---- ret ptxes=", ptxes)
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("get_ranges_by_ratio")
                ax.plot(self.x, self.y)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
                for range_ in ranges:
                    for k, j in enumerate(range_):
                        x_ = self.x[j]
                        ax.plot([x_, x_], [ymin, ymax], ":", color="yellow")
                        ax.plot(x_, self.spline(x_), "o", color="C%d" % (k+1))
                fig.tight_layout()
                plt.show()

        if len(ranges) == 0:
            assert len(valid_boundaries) == 0
            ranges, ptxes = self.make_just_one_peak_info(ratio)

        assert len(ranges) == len(ptxes)

        if len(ranges) != len(valid_boundaries) + 1:
            print('ranges=', ranges, 'valid_boundaries=', valid_boundaries)
            fig = plt.figure()
            ax = fig.gca()
            simple_plot(ax, self, "get_ranges_by_ratio end", boundaries=valid_boundaries, spline=True)
            plt.show()

        assert len(ranges) == len(valid_boundaries) + 1

        if return_ptxes:
            return ranges, np.array(ptxes), valid_boundaries
        else:
            return ranges

    def make_just_one_peak_info(self, ratio):
        # a_curve2 for 20190221_1_ver2

        n = np.argmax(self.spline(self.peak_top_x))
        ptx = self.peak_top_x[n]
        pti = int(ptx - self.x[0] + 0.5)
        yr  = self.spline(ptx) * ratio
        lower   = self.get_lower_from_y( 0, pti, yr )
        upper   = self.get_upper_from_y( len(self.x), pti, yr )

        peak_rec = [lower, pti, upper]
        self.logger.warning("made just one peak %g, %s to avoid creating a no-peak curve instance." % (ptx, str(peak_rec)))
        return [peak_rec], [ptx]

    def modify_ranges_to_avoid_too_narrow_valley(self, ranges, debug=False):
        min_width = min(MININUM_PEAK_WIDTH, int(len(self.x)*MININUM_PEAK_W_RATIO/2))

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            print("min_width=", min_width)
            print("ranges before=", ranges )
            print("self.boundaries=", self.boundaries )
            ranges_save = copy.deepcopy(ranges)
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("modify_ranges_to_avoid_too_narrow_valley debug")
                ax.plot(self.x, self.y)
                for rec in ranges:
                    for k, j in enumerate(rec):
                        ax.plot(j, self.spline(j), "o", color="C%d" % (k+1))
                ax.plot(self.boundaries, self.spline(self.boundaries), "o", color="green")
                fig.tight_layout()
                plt.show()

        if len(ranges) == len(self.boundaries) + 1:
            boundaries = self.boundaries
        else:
            boundaries = self.get_range_boudaries(ranges, self.boundaries)

        num_changes = 0
        for k, b in enumerate(boundaries):
            n = b - ranges[k][2]
            if n < min_width:
                ranges[k][2] -= ( min_width - n )
                num_changes += 1
            n = ranges[k+1][0] - b
            if n < min_width:
                ranges[k+1][0] += ( min_width - n )
                num_changes += 1

        if num_changes > 0 and debug:
            self.logger.warning('%d in points %s were changed to avoid too narrow valleys resulting in %s.', num_changes, str(ranges_save), str(ranges))
            # too many to log

    def get_range_boudaries(self, ranges, boundaries):
        """
        TODO: investigate this case in pH7
        """
        print( 'ranges=', ranges )
        print( 'boundaries=', boundaries )

        ret_boundaries = []
        start = 0
        for k, range_ in enumerate(ranges):
            lower1, top1, upper1 = range_
            if k < len(ranges) - 1:
                next_range = ranges[k+1]
                lower2, top2, upper2 = next_range
                left_lim = (top1 + upper1)/2
                right_lim = (lower2 + top2)/2
                for b in boundaries[start:]:
                    if left_lim < b and b < right_lim:
                        if upper1 > b:
                            range_[2] = b
                        if b > lower2:
                            next_range[0] = b
                        ret_boundaries.append(b)
                        start = k
                        break

        if len(ranges) > 0 and len(ret_boundaries) != len(ranges) - 1:
            print(ranges)
            print(boundaries)
            print(ret_boundaries)
            assert len(ret_boundaries) != len(ranges) - 1

        print( 'ret_boundaries=', ret_boundaries )

        return ret_boundaries

    def remove_peaks( self, to_remove ):
        if len(to_remove) == 0:
            return

        # required for 20190302-analysis_copy
        new_top_x = []
        for k, x in enumerate(self.peak_top_x):
            if k not in to_remove:
                new_top_x.append(x)

        self.peak_top_x = np.array(new_top_x)
        self.set_peak_dependent_info()
        self.logger.warning("peaks %s have been removed.", str(to_remove))

    def add_peaks( self, to_add ):
        print( 'to_add=', to_add )
        if len(to_add) == 0:
            return
        # TODO
        assert False

    def make_rich_peak_info(self, ranges):
        """
        experimental
        """
        rich_peak_info = copy.deepcopy(ranges)

        x = self.x
        y = self.smoothed_y

        foot_level = FOOT_LEVEL_RATIO * self.max_y
        shifted_spline = LSQUnivariateSpline(x, y-foot_level, self.knots[1:-1], ext=3)
        candidate_foots = shifted_spline.roots()
        # print('candidate_foots=', candidate_foots)

        start_peak = 0
        foots = [[]]*len(ranges)
        for f in candidate_foots:
            for k, info in enumerate(ranges[start_peak:], start=start_peak):
                left, _, right = info
                if left - FOOT_ADD_WIDTH < f and f < right + FOOT_ADD_WIDTH:
                    foots[k].append(f)
                    start_peak = k
                    break

        print('foots=', foots)
        foots_ = np.array(foots).flatten()

        if False:
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(x, y, label='data')
            ax.plot(x, self.spline(x), label='spline')
            # ax.plot(x, shifted_spline(x), label='shifted spline')
            ax.plot(candidate_foots, self.spline(candidate_foots), 'o', color='pink', label='candidate_foots')
            ax.plot(foots_, self.spline(foots_), 'o', color='red', label='foots')
            ax.legend()
            plt.show()

        return rich_peak_info

    def locally_tall_enough(self, pos):
        delta_x = self.smoothing.delta_x
        delta_y = self.smoothing.delta_y

        py = self.spline(pos)
        left_h  = py - self.spline(pos-delta_x)
        right_h = py - self.spline(pos+delta_x)
        local_height = (left_h + right_h)/2
        return local_height > delta_y

    def locally_tall_enough_evenif_rotated(self, pos, delta_x=None):
        # TODO: check after rotation

        if delta_x is None:
            delta_x = self.smoothing.delta_x
        delta_y = self.smoothing.delta_y

        py = self.spline(pos)
        left_h  = py - self.spline(pos-delta_x)
        right_h = py - self.spline(pos+delta_x)
        local_height = (left_h + right_h)/2
        return local_height > delta_y

    def compute_valley_bottom_score(self, k):
        b = self.boundaries[k]
        bottom_y = max(0, self.spline(b))
        top_y_left  = self.peak_info[k][1]
        top_y_right = self.peak_info[k+1][1]

        left_ratio = bottom_y/top_y_left
        right_ratio = bottom_y/top_y_right

        ratio = min(1, (left_ratio + right_ratio)/2 )
        return 1 - ratio

    def compute_moments(self):
        if self.mean is None:
            x = self.x
            y = np.zeros(len(x))

            # self.y can be negative as in 20200123_1
            positive_y = self.y > 0
            y[positive_y] = self.y[positive_y]

            sum_y = np.sum(y)
            mean  = np.sum(x*y)/sum_y
            variance = np.sum(x**2*y)/sum_y - mean**2
            self.mean = mean
            self.variance = variance
        return self.mean, self.variance

    def get_end_y(self, slice_):
        y_ = self.y[slice_]
        h = abs(y_[-1] - y_[0])
        ok = False
        if h/self.max_y < END_DIFF_MIN_HEIGHT:
            ok = True
        else:
            std = np.std(y_)
            if std/self.max_y < END_DIFF_MIN_HEIGHT:
                ok = True
        if ok:
            self.end_slices.append(slice_)
            return np.average(y_)

        # in this case, slice_ must be narrowed as obsesrved in 20181204
        if slice_.start == 0:
            return self.get_end_y(slice(0, slice_.stop//2))
        else:
            return self.get_end_y(slice(slice_.start//2, None))

    def compute_sedimentation_rate(self, debug=False):
        emg_peaks = self.get_emg_peaks()
        self.end_slices = []
        start_y = self.get_end_y(slice(0, END_DIFF_WIDTH))
        end_y = self.get_end_y(slice(-END_DIFF_WIDTH, None))
        sedrate = (end_y - start_y)/(self.max_y - start_y)

        slice1 = self.end_slices[0]
        width1 = slice1.stop

        slice2 = self.end_slices[1]
        width2 = abs(slice2.start)
        start2 = len(self.x) + slice2.start

        num_emg_peaks = len(emg_peaks)

        if num_emg_peaks > 0:
            # to reduce the calulation
            indeces = set([0, num_emg_peaks-1])

            lim_x_pairs = []
            for i in indeces:
                if USE_RELIABLE_ALPHA:
                    pair = emg_peaks[i].get_model_x_from_ratio(RELIABLE_ALPHA)
                else:
                    pair = emg_peaks[i].get_sigma_points(RELIABLE_SIGMA_RATIO)
                lim_x_pairs.append(pair)
            sigma_x1 = lim_x_pairs[0][0]
            sigma_x2 = lim_x_pairs[-1][1]
            reliable = (width1 >= END_DIFF_WIDTH//8 and slice1.stop < sigma_x1) and (width2 >= END_DIFF_WIDTH//8 and sigma_x2 < start2)
            if debug:
                import molass_legacy.KekLib.DebugPlot as plt
                print(self.end_slices, sigma_x1, sigma_x2, start2)
                print(width1, width2)

                x = self.x
                y = self.y

                plt.push()
                fig, ax = plt.subplots()
                ax.set_title("compute_sedimentation_rate debug")
                ax.plot(x, y, label='data')
                for slice_ in [slice(0,END_DIFF_WIDTH), slice(-END_DIFF_WIDTH, None)]:
                    ax.plot(x[slice_], y[slice_], 'o', color='yellow')

                for peak in emg_peaks:
                    print(self.j0, peak, peak.get_sigma_points(RELIABLE_SIGMA_RATIO))

                for k in set([0, len(emg_peaks)-1]):
                    peak = emg_peaks[k]
                    ax.plot(x, peak.get_model_y(x), label='model')

                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
                for sx in [sigma_x1, sigma_x2]:
                    ax.plot([sx, sx], [ymin, ymax], ':', color='gray')

                ax.legend()
                fig.tight_layout()
                plt.show()
                plt.pop()
        else:
            # temporary fix for pH6 to avoid a bug with the revised EmgPeak
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(self.logger, "compute_sedimentation_rate failure: ", n=10)
            reliable = False

        self.logger.info("computed end_slices as %s, sedimentation rate as %.3g which is%s reliable", str(self.end_slices), sedrate, "" if reliable else " not")
        return sedrate, reliable

    def get_end_slices(self):
        if self.end_slices is None:
            # this case is not desirable.
            # improve so that self.compute_sedimentation_rate be called only once.
            self.compute_sedimentation_rate()
        return self.end_slices

    def peak_no_from_range(self, start, stop):
        max_width = None
        ret_no = None
        for k, rec in enumerate(self.peak_info):
            lsp, _, rsp = rec
            width = max(0, min(rsp, stop) - max(lsp, start))
            if width > 0:
                if max_width is None or width > max_width:
                    ret_no = k
                    max_width = width
        return ret_no

    def get_end_points(self, debug=False):
        from molass_legacy.KekLib.GeometryUtils import rotate
        x = self.x
        y = self.y
        pinfo1 = self.peak_info[0]
        position_ratio = pinfo1[1]/len(x)
        if position_ratio < 0.05:
            pinfo1 = self.peak_info[1]
        pinfo2 = self.peak_info[-1]
        ret_points = []
        for k, info in enumerate([pinfo1, pinfo2]):
            m = info[1]
            if k == 0:
                slice_ = slice(0, m)
                angle = -np.pi/4        # 
                ty = y[m]
            else:
                slice_ = slice(m, None)
                angle = np.pi/4
                ty = y[m]
            x_ = self.x[slice_]
            wx = (x_ - x_[0]) / len(x_)
            h_ = y[m]
            y_ = y[slice_]
            wy = y_ / h_
            rx, ry = rotate(angle, wx, wy)
            n = np.argmin(ry)
            if n > 0 and n < len(ry) - 1:
                n_ = int(x_[0]) + n
                px = x[m]
                py = y[m]
                fx = x[n_]
                fy = y[n_]
                gx = px + (fx - px)*ty/(py - fy)
            else:
                if k == 0:
                    n = 0
                    fx = x_[n]
                    fy = y_[n]
                    gx = fx
                else:
                    # as in 20161119/Kosugi8
                    n = -1
                    fx = x_[n]
                    fy = y_[n]
                    gx = fx
            ret_points.append(gx)

            if debug:
                import molass_legacy.KekLib.DebugPlot as plt
                plt.push()
                fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(21,6))
                ax0.plot(x_, y_)
                ax0.plot(fx, fy, 'o', color='red')
                ax0.plot(gx, py - ty, 'o', color='yellow')
                ax1.plot(wx, wy)
                ax1.plot(wx[n], wy[n], 'o', color='red')
                ax2.plot(rx, ry)
                ax2.plot(rx[n], ry[n], 'o', color='red')
                fig.tight_layout()
                plt.show()
                plt.pop()

        return ret_points

    def get_peak_position(self, start, stop):
        j = (start + stop)/2
        min_pos = None
        min_val = None
        for rec in self.peak_info:
            pos = rec[1]
            val = abs(pos - j)
            if min_val is None or val < min_val:
                min_val = val
                min_pos = pos
        self.logger.info("pos=%d for range(%d, %d)", min_pos, start, stop)
        return min_pos

    def get_model_param_list(self):
        return [epeak.get_params() for epeak in self.get_emg_peaks()]

    def get_peak_region_sigma(self):
        if self.peak_region_sigma is None:
            emg_peaks = self.get_emg_peaks()
            h1, m1, s1, t1 = emg_peaks[0].get_params()
            h2, m2, s2, t2 = emg_peaks[-1].get_params()
            peak_region = (max(0, m1-3*s1), min((len(self.x)), m2+3*s2))
            self.peak_region_sigma = peak_region
        return self.peak_region_sigma

    def get_peak_region(self, sigma_scale=3):
        emg_peaks = self.get_emg_peaks()
        h1, m1, s1, t1 = emg_peaks[0].get_params()
        h2, m2, s2, t2 = emg_peaks[-1].get_params()
        peak_region = (max(0, m1-sigma_scale*s1), min((len(self.x)), m2+sigma_scale*s2))
        return peak_region

    def get_peak_region_width(self):
        peak_region = self.get_peak_region_sigma()
        return peak_region[1] - peak_region[0]
    
    def is_roughly_centered(self):
        x = self.x
        xc = (x[0] + x[-1])/2
        return abs(xc - self.max_x)/(x[-1] - x[0]) < 0.3