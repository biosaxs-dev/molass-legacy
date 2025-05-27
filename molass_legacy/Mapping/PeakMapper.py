"""

    PeakMapper.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF

"""
import copy
from bisect import bisect_right
import numpy            as np
import logging
from scipy.stats        import pearsonr, linregress
from scipy.interpolate  import UnivariateSpline
from scipy.optimize     import minimize
from itertools          import combinations
import OurStatsModels   as sm
from importlib import reload
from .SimplestMapper import SimplestMapper
from .PeakCurve import PeakCurve, get_peak_curve_info
from molass_legacy.SerialAnalyzer.ElutionCurve       import ElutionCurve
from molass_legacy.Elution.CurveUtils  import simple_plot, find_rotated_extreme_arg
from molass_legacy.KekLib.SciPyCookbook      import smooth
from molass_legacy.KekLib.BasicUtils         import Struct
from molass_legacy.KekLib.ExceptionTracebacker import log_exception, warnlog_exception
import molass_legacy.KekLib.DebugPlot        as plt
from molass_legacy._MOLASS.SerialSettings     import get_setting

COMPARE_ADDING_PEAKS_2WAYS  = True
NUM_PEAKS_SCORING_WEIGHT    = 0.01
USE_SCORED_BOUNDARY         = False
ROUGH_MIN_Y_RATIO_LIST      = [0.02, 0.1]
ADEQUACY_CORRELATION        = 0.997     # 0.996 for monoE, 0.99997 for SUB_TRN1
RISKY_CORRELATION_LIMIT     = 0.95      # > 0.88 for 20190221_2
NUM_ROOTS_TO_AVOID_NOISE    = 5         # to void low quality peaks as in pH7, > 3 for SUB_TRN1
FEASIBLE_DIFF_NUM           = 2
REDUCE_WHEN_INFEASIBLE      = True
ADOPTABLE_SCORE_RATIO       = 1.05      # > 1.01 for 20170304
RELIABLE_PEAK_SCORE         = 0.9
NUM_FRAMES_BOUNADRY         = 400       # < 461 for 20210628_2
MINIMUM_HEIGHT_RATIO        = 0.04      # > 0.03 for 20200121_2
ACCEPTABLE_SIMILARITY       = 0.97      # > 0.93 for 20210323_1, 0.96 for 20200121_2
DEEM_SINGLE_PEAK_RATIO      = 0.1
ACCEPTABLE_ENDS_RATIO_DEV   = 0.5

def debug_plot( a_curve, x_curve, where_str, params=None):
    from DataUtils import get_in_folder

    plt.push()
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle( 'PeakMapper debug plot for ' + get_in_folder() + ' (%s)' % where_str, fontsize=24 )
    ax1 = fig.add_subplot( 121 )
    ax2 = fig.add_subplot( 122 )
    ax1.set_title( "UV Curve", fontsize=20 )
    ax2.set_title( "Xray Curve", fontsize=20 )

    ax1.plot( a_curve.y )
    ax1.plot( a_curve.spline(a_curve.x), color='lime' )
    ax2.plot( x_curve.y, color='orange' )
    ax2.plot( x_curve.spline(x_curve.x), color='lime' )

    def plot_peaks( ax, curve ):
        for info in curve.peak_info:
            peak = info[1]
            ax.plot( peak, curve.spline( peak ), 'o', color='red', markersize=10 )

        for b in curve.boundaries:
            ax.plot( b, curve.spline( b ), 'o', color='yellow', markersize=8  )

    plot_peaks( ax1, a_curve )
    plot_peaks( ax2, x_curve )

    if params is not None:
        A, B = params
        A_, B_ = 1/A, -B/A
        for x1 in a_curve.peak_top_x:
            x = A_*x1 + B_
            y = x_curve.spline(x)
            ax2.plot(x, y, 'o', color='cyan')

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()
    plt.pop()

def simple_plot_pair_curves(title, a_curve, x_curve):
    from DataUtils import get_in_folder
    in_folder = get_in_folder()

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        fig.suptitle(title + " with %s" % in_folder, fontsize=20)
        simple_plot(ax1, a_curve, color="blue")
        simple_plot(ax2, x_curve, color="orange")
        fig.tight_layout()
        plt.show()

def mapping_proof(title, a_curve, x_curve, mapping):
    x = x_curve.x
    y = x_curve.y
    A, B = mapping
    uv_x = x*A + B

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        fig.suptitle(title)

        simple_plot(ax1, a_curve, legend=False)
        simple_plot(ax2, x_curve)

        ax1.plot(uv_x, y, label="mapped")
        ax1.legend()

        fig.tight_layout()
        plt.show()

def make_unique_peak_info( old_list ):
    temp_list = sorted( old_list )

    new_list = []
    last_info = None
    for info in temp_list:
        if last_info is None or info[0] != last_info[0]:
            new_list.append( info )

        last_info = info

    return new_list

def make_unique_boundaries( old_list ):
    return sorted( list( set( old_list ) ) )

class ProxyCurve(ElutionCurve):
    def __init__( self, curve, peak_top_x, peak_info, boundaries, add_peak_info, add_boundaries, correct=False ):
        self.__dict__   = curve.__dict__.copy()
        self.logger     = logging.getLogger(__name__)
        self.peak_top_x = np.array(peak_top_x)
        self.peak_info  = peak_info
        self.boundaries = boundaries
        self.peak_slice = slice(0, len(curve.x))    # comptibility with PeakCurve
        self.emg_peaks = None
        if correct:
            self.correct_valley_bottm(debug=True)

    def correct_valley_bottm(self, debug=False):

        if debug:
            saved_boundaries = copy.deepcopy(self.boundaries)

        for k, b in enumerate(self.boundaries):
            start = max(0, b - 20)
            stop  = min(len(self.x), b + 20)
            y_ = self.spline(self.x[start:stop])
            ppp = np.argpartition(y_, 5)
            n = start + int(np.average(ppp[0:5+1]) + 0.5)
            if n != b:
                self.logger.info("corrected %dth valley bottm from %d to %d." % (k, b, n))

        if debug:
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(saved_boundaries, self.spline(saved_boundaries), 'o', color='gray', alpha=0.2, markersize=10, label='saved_boundaries')
            simple_plot(ax, self, "correct_valley_bottm debug")
            fig.tight_layout()
            plt.show()

    def get_peak_region_width(self):
        # note that ElutionCurve does not currently have this method
        return self.peak_slice.stop - self.peak_slice.start

class MajorCurve(ElutionCurve):
    def __init__(self, curve):
        self.__dict__   = curve.__dict__.copy()
        self.logger     = logging.getLogger(__name__)
        self.peak_top_x = curve.major_peak_top_x
        self.peak_info  = curve.major_peak_info
        self.boundaries = curve.major_boundaries

class LessPeakCurve(ElutionCurve):
    def __init__(self, curve, num_less):
        self.__dict__   = curve.__dict__.copy()
        self.logger     = logging.getLogger(__name__)
        peak_height_ratios = [ curve.get_height_ratio(top) for top in curve.peak_top_x ]
        less_index = np.argpartition(peak_height_ratios, num_less)[:num_less]
        print('peak_height_ratios=', peak_height_ratios)
        print('less_index=', less_index)
        new_peak_top_x = []
        for k, top in enumerate(curve.peak_top_x):
            if k in less_index:
                continue
            new_peak_top_x.append(top)
        self.peak_top_x = np.array(new_peak_top_x)
        self.set_peak_dependent_info()

class MappedInfo(list):
    def __init__(self, info_list, curves=None):
        list.__init__(self, info_list)
        self._curves = curves
        best_params = info_list[0]
        if best_params is None:
            # this can happen in the preliminary recongnition as in 20171226
            pass
        else:
            self._A, self._B = best_params

    def xr_to_uv(self, xr_p):
        return self._A * xr_p + self._B

    def uv_to_xr(self, uv_p):
        return (uv_p - self._B)/self._A

def slice_length(s):
    return s.stop - s.start

class PeakMapper:
    def __init__(self, a_curve, x_curve, simply=False, pre_recog=None, debug=False, debug_detail=False ):
        self.logger = logging.getLogger(__name__)

        self.a_curve = a_curve
        self.x_curve = x_curve
        self.peak_mapping_only = get_setting('peak_mapping_only')
        self.feature_mapped = False
        self.reliable = True
        self.debug  = debug
        self.debug_detail = debug_detail

        fallback = False
        if not simply:
            try:
                sliced_pm_info = self.get_sliced_peak_mapper(a_curve, x_curve, debug=debug)
                self.pre_recog  = pre_recog
                self.make_it_compatible_from_sliced(sliced_pm_info, debug=debug)
            except Exception as exc:
                warnlog_exception(self.logger, "sliced peak mapping failed: ")
                simply = True
                fallback = True
                self.logger.warning("resorting to the simple treatment (i.e., without slicing)  due to the previous error.")

        if simply:
            if debug:
                simple_plot_pair_curves("PeakCurves (0)", self.a_curve, self.x_curve)

            try:
                self.pre_recog = None
                self.do_mapping(debug=debug)
                if fallback:
                    A, B =self.get_imroved_mapping(fallback=fallback)
                    self.mapped_info[0] = (A, B)
                    self.best_info[1] = (A, B)
            except Exception as exc:
                # as in 20201007_2
                from .ReducedCurve import make_reduced_curve
                self.logger.warning("resorting to curve reducing due to exception: %s", str(exc))
                curves = []
                for curve in self.a_curve, self.x_curve:
                    curves.append(make_reduced_curve(curve))
                self.do_mapping(reduced_curves=curves, debug=debug)

    def preliminary_check(self, a_curve, x_curve):
        a_frames = len(a_curve.x)
        x_frames = len(x_curve.x)
        frames_ratio = a_frames/x_frames
        almost_same_scale = abs(frames_ratio - 1) < 0.1

        peaktop_ratio = (a_curve.primary_peak_x/a_frames)/(x_curve.primary_peak_x/x_frames)
        proportinal_peaktop = abs(peaktop_ratio - 1) < 0.1

        a_peaks = len(a_curve.peak_info)
        x_peaks = len(x_curve.peak_info)
        same_few_num_peaks = a_peaks == x_peaks and x_peaks <= 2

        return Struct(  almost_same_scale=almost_same_scale,
                        frames_ratio=frames_ratio,
                        proportinal_peaktop=proportinal_peaktop,
                        a_peaks = a_peaks,
                        x_peaks = x_peaks,
                        same_few_num_peaks=same_few_num_peaks,
                        )

    def check_peak_existence(self, a_curve, x_curve, title, plot=True):
        for curve in a_curve, x_curve:
            if len(curve.peak_info) == 0:
                if plot:
                    with plt.Dp():
                        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                        fig.suptitle(title)
                        simple_plot(ax1, a_curve, color="blue")
                        simple_plot(ax2, x_curve, color="orange")
                        fig.tight_layout()
                        plt.show()
                assert False

    def get_sliced_peak_mapper(self, a_curve, x_curve, debug=False):
        self.check_peak_existence(a_curve, x_curve, "get_sliced_peak_mapper entry")

        check_info = self.preliminary_check(a_curve, x_curve)
        if debug:
            print("check_info=", check_info)
            simple_plot_pair_curves("get_sliced_peak_mapper (1)", a_curve, x_curve)

        peak_curve_ok = False
        if len(x_curve.x) >= NUM_FRAMES_BOUNADRY:
            try:
                a_curve_, a_slice_ = get_peak_curve_info(a_curve, debug=debug)
                x_curve_, x_slice_ = get_peak_curve_info(x_curve, debug=debug)
                try:
                    a_curve_, a_slice_, x_curve_, x_slice_ = self.improve_slices_if_possible(check_info, a_curve_, a_slice_, x_curve_, x_slice_)
                except:
                    warnlog_exception(self.logger, "improve_slices_if_possible failed: ")
                peak_curve_ok = True
            except:
                # as in 20201127_5
                warnlog_exception(self.logger, "get_sliced_peak_mapper(1)")
        elif len(a_curve.x) >= NUM_FRAMES_BOUNADRY*2:
            # as in 20201009_1
            try:
                a_curve_, a_slice_ = get_peak_curve_info(a_curve, debug=debug)
                x_curve_, x_slice_ = x_curve, None
                peak_curve_ok = True
            except:
                warnlog_exception(self.logger, "get_sliced_peak_mapper (2)")

        if not peak_curve_ok:
            # as in 20191114_3 where len(a_curve.x) >= 1000 while len(x_curve.x) < 1000
            # avoiding the inconvenience in get_peak_curve_info, which should better be investigated anyway
            a_curve_, a_slice_ = a_curve, None
            x_curve_, x_slice_ = x_curve, None

        if debug:
            simple_plot_pair_curves("get_sliced_peak_mapper (2)", a_curve_, x_curve_)

        self.check_peak_existence(a_curve_, x_curve_, "get_sliced_peak_mapper before recursive construction")

        pm = PeakMapper(a_curve_, x_curve_, simply=True, debug=debug)
        return Struct(pm=pm, a_slice_=a_slice_, x_slice_=x_slice_)

    def improve_slices_if_possible(self, check_info, a_curve_, a_slice_, x_curve_, x_slice_, debug=False):
        sliced_check_info = self.preliminary_check(a_curve_, x_curve_)
        # print("------------------- check_info=", check_info)
        # print("------------------- sliced_check_info=", sliced_check_info)
        if sliced_check_info.a_peaks > 2 and sliced_check_info.x_peaks >= 2:
            # as in 20220716/FER_OA_302
            return a_curve_, a_slice_, x_curve_, x_slice_

        if check_info.almost_same_scale:
            pass
        else:
            if check_info.proportinal_peaktop:
                pass
            else:
                ok = True
                if check_info.same_few_num_peaks:
                    if not sliced_check_info.same_few_num_peaks:
                        # as in 20200121_3
                        ok = False
                if ok:
                    return a_curve_, a_slice_, x_curve_, x_slice_
                else:
                    pass

        ratio = slice_length(a_slice_)/(check_info.frames_ratio * slice_length(x_slice_))
        if abs(ratio - 1) < 0.1:
            return a_curve_, a_slice_, x_curve_, x_slice_

        """
        try to improve the slice in cases like 20200121_2 or 20200630_11
        """

        if not check_info.same_few_num_peaks:
            # like 20200630_11
            # note that unmatching peaks will be removed
            num_ratio = len(a_curve_.peak_info)/len(x_curve_.peak_info)
            # assert (ratio - 1)*(num_ratio - 1) > 0      # i.e. of the same sign
            curve_name = "x_curve" if num_ratio < 1 else "a_curve"
            self.logger.info("unmatching peaks will probably be removed from %s", curve_name)

        if ratio < 1:
            curve1 = self.a_curve
            curve1_ = a_curve_
            slice1_ = a_slice_
            curve2 = self.x_curve
            curve2_ = x_curve_
            slice2_ = x_slice_
            self.logger.info("improve_slices_if_possible: improving x_slice_=%s from a_slice_=%s for ratio=%.3g", str(x_slice_), str(a_slice_), ratio)
        else:
            curve1 = self.x_curve
            curve1_ = x_curve_
            slice1_ = x_slice_
            curve2 = self.a_curve
            curve2_ = a_curve_
            slice2_ = a_slice_
            self.logger.info("improve_slices_if_possible: improving a_slice_=%s from x_slice_=%s for ratio=%.3g", str(a_slice_), str(x_slice_), ratio)

        sm = SimplestMapper(curve1, curve2)
        slice_ = slice(sm.map_1to2(slice1_.start), sm.map_1to2(slice1_.stop))
        ecurve = ElutionCurve(curve2.y[slice_])
        ecurve.get_default_editor_ranges()

        if debug:
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("improve_slices_if_possible result")
                simple_plot(ax1, curve1)
                simple_plot(ax2, curve2, legend=False)
                ax2.plot(slice_.start+ecurve.x, ecurve.y, ":", label="with improved slice")
                ax2.legend()
                fig.tight_layout()
                plt.show()

        if ratio < 1:
            return curve1_, slice1_, ecurve, slice_
        else:
            return ecurve, slice_, curve1_, slice1_

    def check_minimum_mapping_quality(self, a_curve, x_curve, where_text):
        x = [0]
        y = [0]
        for info1, info2 in zip(a_curve.peak_info, x_curve.peak_info):
            x.append(info2[1])
            y.append(info1[1])
        x.append(len(x_curve.x))
        y.append(len(a_curve.x))

        correlation = pearsonr(x, y)[0]
        print(where_text, "------------------ correlation=", correlation)

    def do_mapping(self, reduced_curves=None, debug=False):

        if reduced_curves is None:
            a_curve = self.a_curve
            x_curve = self.x_curve
        else:
            a_curve, x_curve = reduced_curves

        if debug:
            simple_plot_pair_curves("before mapping", a_curve, x_curve)

        if self.debug:
            from CallStack import CallStack
            cstack = CallStack()
            print('PeakMapper call stack=', cstack)
            debug_plot( a_curve, x_curve, "before" )

        A, B = self.get_simplest_params(a_curve, x_curve)

        if debug:
            import Mapping.PeakMappingSolver
            from importlib import reload
            reload(Mapping.PeakMappingSolver)
        from .PeakMappingSolver import PeakMappingSolver
        pms = PeakMappingSolver(a_curve, x_curve, (A, B), debug=debug)
        uv_mapper, xr_mapper = pms.get_opt_mappers()
        try:
            better_a_curve = self.make_better_curve(a_curve, x_curve, xr_mapper, (A, B), "better_a_curve", debug=debug)
            better_x_curve = self.make_better_curve(x_curve, a_curve, uv_mapper, (1/A, -B/A), "better_x_curve", debug=debug)
            curves = [better_a_curve, better_x_curve]
        except Exception as exc:
            warnlog_exception(self.logger, "making thinner curves due to ")
            thinner_a_curve = self.make_thinner_curve(a_curve, uv_mapper)
            thinner_x_curve = self.make_thinner_curve(x_curve, xr_mapper)
            curves = [thinner_a_curve, thinner_x_curve]

        if self.debug:
            debug_plot(*curves, "after")

        self.check_peak_existence(*curves, "do_mapping result", plot=False)
        self.check_minimum_mapping_quality(*curves, "do_mapping result")

        self.mapped_curves = curves

        if debug:
            simple_plot_pair_curves("after mapping", *curves)

    def make_it_compatible_from_sliced(self, sliced_pm_info, debug=False):
        pm = sliced_pm_info.pm
        a_slice_ = sliced_pm_info.a_slice_
        a_start = 0 if a_slice_ is None else a_slice_.start
        x_slice_ = sliced_pm_info.x_slice_
        x_start = 0 if x_slice_ is None else x_slice_.start
        mapped_curves = pm.get_mapped_curves()
        # A, B = pm.best_info[1]
        A, B = pm.get_imroved_mapping()

        B_ = a_start + A*(-x_start) + B

        if debug:
            simple_plot_pair_curves("make_it_compatible_from_sliced", *mapped_curves)

        # None's will not be used
        self.best_info = [pm.best_info[0], (A,B_), None, None, None]
        self.mapped_info = MappedInfo([(A, B_), [], None], [self.a_curve, self.x_curve])
        curves = []
        for curve, mp_curve, slice_ in [(self.a_curve, mapped_curves[0], a_slice_),
                                        (self.x_curve, mapped_curves[1], x_slice_) ]:
            if slice_ is None:
                pkcurve = mp_curve
            else:
                pkcurve = PeakCurve(curve, pkslice_info=(mp_curve, slice_))
                pkcurve.get_emg_peaks()

                """
                try:
                    pkcurve = PeakCurve(curve, pkslice_info=(mp_curve, slice_))
                    pkcurve.get_emg_peaks()
                except:
                    warnlog_exception(self.logger, "make_it_compatible_from_sliced: ")
                    pkcurve = curve
                """

            curves.append(pkcurve)

        self.check_minimum_mapping_quality(*curves, "make_it_compatible_from_sliced result")
        self.mapped_curves = curves

        if debug:
            simple_plot_pair_curves("after make_it_compatible_from_sliced", *curves)

    def get_simplest_params(self, curve1, curve2):
        max_score, max_params, max_indeces, max_correl, max_simil, max_info = self.get_best_mapping_params(curve1, curve2, peak_mapping_only=False)
        B, A = max_params

        if len(curve1.peak_info) < 3 or len(curve2.peak_info) < 3:
            a, b = self.compare_primary_peak_mapping_params(curve1, curve2, (A, B))
        else:
            # avoid wornsening as with 202300705 (BSA)
            a, b = A, B

        # for backward compatibility begin
        self.best_info = [max_score, (a, b), max_indeces, max_correl, max_simil]
        self.mapped_info = MappedInfo([(a, b), [max_correl, max_simil], max_info], [self.a_curve, self.x_curve])
        # for backward compatibility end

        return a, b

    def get_mapped_curves( self ):
        return self.mapped_curves

    def find_best_mapping( self, curve1, curve2, check_adequacy=True, features=True):

        curve1_p, curve2_p, best_info_p, mapped_info_p = self.find_best_mapping_impl(curve1, curve2, False, check_adequacy, debug=False)

        if features and not self.peak_mapping_only:
            curve1_f, curve2_f, best_info_f, mapped_info_f = self.find_best_mapping_impl(curve1_p, curve2_p, True, check_adequacy, mapped_info=mapped_info_p, debug=False)

            if best_info_f[0] > best_info_p[0]:
                best_info = best_info_f
                mapped_info = mapped_info_f
                ret_curves = [curve1_f, curve2_f]
                self.feature_mapped = True
                self.logger.info("features matching has been successful: %g > %g" % (best_info_f[0],  best_info_p[0]))
            else:
                best_info = best_info_p
                mapped_info = mapped_info_p
                ret_curves = [curve1_p, curve2_p]

                if self.pre_recog is not None:
                    mapped_info = self.pre_recog.get_restricted_mapped_info()
                    curve1_f, curve2_f, best_info_f, mapped_info_f = self.find_best_mapping_impl(curve1_p, curve2_p, True, check_adequacy, mapped_info=mapped_info, debug=False)
                    if best_info_f[0] > best_info_p[0]:
                        best_info = best_info_f
                        mapped_info = mapped_info_f
                        ret_curves = [curve1_f, curve2_f]
                        self.feature_mapped = True

                if self.feature_mapped:
                    self.logger.info("features matching has been successful using pre-recog: %g > %g" % (best_info_f[0],  best_info_p[0]))
                else:
                    self.logger.warning("features matching has been worse: %g <= %g" % (best_info_f[0], best_info_p[0]))
        else:
            best_info = best_info_p
            mapped_info = mapped_info_p
            ret_curves = [curve1_p, curve2_p]

        self.mapped_info = mapped_info
        self.best_info = best_info

        return ret_curves

    def evaluate_mapping_impl( self, size1, size2, curve1, peaks1, index1, curve2, peaks2, index2, use_ends=True, debug=False):
        if use_ends and (len(peaks1) < 2 or len(peaks2) < 2):
            y   = [ 0, size1-1 ]
            x   = [ 0, size2-1 ]
            w   = [ 1, 1 ]
        else:
            y   = []
            x   = []
            w   = []

        for p1, p2 in zip( peaks1[index1], peaks2[index2] ):
            y.append( p1 )
            x.append( p2 )
            w.append( 10 )

        if not use_ends:
            for rec1, rec2 in zip(curve1.peak_info, curve2.peak_info):
                for i in [0, 2]:
                    y.append( rec1[i] )
                    x.append( rec2[i] )
                    w.append( 5 )

        X   = sm.add_constant(x)
        mod = sm.WLS( y, X, weights=w )
        res = mod.fit()

        if False:
            B, A = res.params
            mp_x = curve2.x*A + B
            mp_y = curve2.spline(mp_x)/curve2.max_y*curve1.max_y
            rmsd = np.sqrt(np.average((curve1.spline(mp_x) - mp_y)**2))

        if debug:
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("evaluate_mapping_impl")
                simple_plot(ax1, curve1, color="blue")
                ax1.plot(mp_x, mp_y, ":")
                simple_plot(ax2, curve2, color="orange")
                fig.tight_layout()
                plt.show()

        correl, simil = self.compute_mapped_curve_simimarity(res.params, x, y, w, [index1, index2], curve1, curve2, debug=False )

        return correl, simil, res.params, [x, y, w]

    def evaluate_mapping( self, size1, size2, curve1, peaks1, index1, curve2, peaks2, index2):

        first_result = self.evaluate_mapping_impl(size1, size2, curve1, peaks1, index1, curve2, peaks2, index2)
        correl, simil, params, info = first_result
        first_score = correl + simil
        # print('first_score=', first_score, (correl, simil) )
        if len(curve1.peak_info) == 1 and len(curve2.peak_info) == 1 and simil < 0.9:
            try:
                second_result = self.evaluate_mapping_impl(size1, size2, curve1, peaks1, index1, curve2, peaks2, index2, use_ends=False)
                correl, simil, params, info = second_result
                second_score = correl + simil
                # print('second_score=', second_score, (correl, simil) )
            except:
                second_score = 0
            score_ratio = second_score/first_score
            print('score_ratio=', score_ratio)
            if score_ratio > ADOPTABLE_SCORE_RATIO:
                # there should be enough superiority to adopt the non-use_ends result
                self.logger.info("the 2nd result scored %.3g has been adopted over the 1st result scored %.3g." % (second_score, first_score) )
                ret_result = second_result
            else:
                ret_result = first_result
        else:
            ret_result = first_result

        return ret_result

    def evaluate_mapping_features( self, curve1, points1, weights1, index1, curve2, points2, weights2, index2):
        y = []
        x = []
        w = []

        assert_later = False

        inappropriate = False
        for p1, w1, p2, w2 in zip( points1[index1], weights1[index1], points2[index2], weights2[index2] ):
            y.append( p1 )
            x.append( p2 )
            if assert_later:
                if abs(w1 - w2) > 1e-6:
                    inappropriate = True
            else:
                # debug_plot(curve1, curve2, "evaluate_mapping_features")
                assert abs(w1 - w2) < 1e-6
            w.append( max(w1, w2) )

        X   = sm.add_constant(x)
        mod = sm.WLS( y, X, weights=w )
        res = mod.fit()

        correl, simil = self.compute_mapped_curve_simimarity(res.params, x, y, w, [index1, index2], curve1, curve2, features=True, debug=False )

        if assert_later:
            if inappropriate:
                assert False
                # pass

        return correl, simil, res.params, [x, y, w]

    def compute_mapped_curve_simimarity(self, params, x, y, w, indeces, curve1, curve2, features=False, nodebug=False, debug=False):
        intercept, slope = params

        correlation = pearsonr(x, y)[0]

        j_ = np.array([0, len(curve1.x)])
        i_ = 1/slope*j_ - intercept/slope
        i_start = int( min(0, i_[0]) )
        i_stop = int( max(len(curve2.x), i_[-1]))

        i = np.arange(i_start, i_stop)
        j = slope * i + intercept

        source_y = curve2.spline(i)
        scale = curve2.height/curve1.height
        mapped_y_ = curve1.spline(j) * scale

        def obj_func(params):
            d = np.sum(np.abs(mapped_y_ + params[0] - source_y))
            return d

        # allow any vertical shift and minimize the gap area
        result = minimize(obj_func, np.zeros(1))

        diff_area = obj_func(result.x)
        mapped_y = mapped_y_ + result.x[0]

        whole_area = curve2.height * len(source_y)
        similarity = abs(1 -  diff_area/whole_area)

        if debug:
        # if True:
            from matplotlib.patches import Polygon
            from DataUtils import get_in_folder

            in_folder = get_in_folder()
            markersize = 10

            plt.push()
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
            ax1, ax2, ax3 = axes
            ax1t = ax1.twinx()

            fig.suptitle("Mapping Explained for %s: indeces=%s" % (in_folder, str(indeces)), fontsize=20)
            ax1.set_title("Data Curves", fontsize=16)
            ax2.set_title("Elution No Mapping", fontsize=16)
            ax3.set_title("Mapped Curves", fontsize=16) 

            c_colors = ['blue', 'orange']
            p_colors = ['cyan', 'orange']

            ax1.plot(j, curve1.spline(j), color=c_colors[0])
            ax1t.plot(i, source_y, color=c_colors[1])

            ax2.plot(i, j, color='green')
            xmin2, xmax2 = ax2.get_xlim()
            ymin2, ymax2 = ax2.get_ylim()

            spanx = xmax2 - xmin2
            spany = ymax2 - ymin2
            xmin_ = -spanx*0.1
            ymin_ = -spany*0.1

            ax2.plot(x, [ymin_]*len(x), 'o', markersize=markersize, color=p_colors[1])
            ax2.plot([xmin_]*len(y), y, 'o', markersize=markersize, color=p_colors[0])

            for px, py, pw in zip(x, y, w):
                ax1.plot(py, curve1.spline(py), 'o', markersize=markersize, color=p_colors[0])
                ax1t.plot(px, curve2.spline(px), 'o', markersize=markersize, color=p_colors[1])
                ax2.plot(px, py, 'o', markersize=markersize, color='lime')
                py_ = slope * px + intercept
                ax3.plot(px, curve1.spline(py_) * scale, 'o', markersize=markersize, color=p_colors[0])
                ax3.plot(px, curve2.spline(px), 'o', markersize=markersize, color=p_colors[1])
                shrinky = spany*0.02/(py - ymin_)
                ax2.annotate("", xy=(px, py), xytext=(px, ymin_), alpha=0.5,
                            arrowprops=dict(arrowstyle='->', ls='dashed', lw=1.5, color=c_colors[1], shrinkA=10, shrinkB=10),
                            )
                shrinkx = spanx*0.02/(px - xmin_)
                ax2.annotate("", xy=(xmin_, py), xytext=(px, py), alpha=0.5,
                            arrowprops=dict(arrowstyle='->', ls='dashed', lw=1.5, color=c_colors[0], shrinkA=10, shrinkB=10),
                            )

                if pw > 0.9:
                    ax1.plot(py, curve1.spline(py), 'o', markersize=5, color='red')
                    ax1t.plot(px, curve2.spline(px), 'o', markersize=5, color='red')
                    ax3.plot(px, curve1.spline(py_) * scale, 'o', markersize=5, color='red')
                    ax3.plot(px, curve2.spline(px), 'o', markersize=5, color='red')

            ax3.plot(i, source_y, color=c_colors[1])
            ax3.plot(i, mapped_y, color=c_colors[0])
            ax3.plot(i, mapped_y_, ':', color=c_colors[0])

            poly_points = list(zip( i, source_y )) + list( reversed( list( zip( i, mapped_y ) ) ) )
            diff_poly   = Polygon( poly_points, alpha=0.1 )
            ax3.add_patch(diff_poly)

            xmin, xmax = ax2.get_xlim()
            ymin, ymax = ax2.get_ylim()
            tx = (xmin + xmax)/2
            ty = (ymin + ymax)/2
            ax2.text(tx, ty, "Linearity=%.3g" % correlation, ha='center', va='center', fontsize=50, alpha=0.1)

            xmin, xmax = ax3.get_xlim()
            ymin, ymax = ax3.get_ylim()
            tx = (xmin + xmax)/2
            ty = (ymin + ymax)/2
            ax3.text(tx, ty, "Similarity=%.3g" % similarity, ha='center', va='center', fontsize=50, alpha=0.1)

            fig.tight_layout()
            fig.subplots_adjust(top=0.85)
            plt.show()
            plt.pop()

        if False and not nodebug and self.debug_detail or debug:
            from scipy.interpolate import UnivariateSpline
            from matplotlib.patches import Polygon

            colors = ['blue', 'orange']

            size1   = len(curve1.y)
            size2   = len(curve2.y)

            if features:
                points1, weights1 = curve1.feature.get_points()
                points2, weights2 = curve2.feature.get_points()
                peaks1  = points1
                peaks2  = points2
                points_label = 'points'
                weight_boundary = 0.75
            else:
                peaks1  = np.array( [ info[1] for info in curve1.peak_info ] )
                peaks2  = np.array( [ info[1] for info in curve2.peak_info ] )
                points_label = 'peaks'
                weight_boundary = 5

            unmapped_index = []
            for k in range(len(peaks1)):
                if k not in indeces:
                    unmapped_index.append(k)

            mapped_spline = UnivariateSpline(i, mapped_y, s=0, ext=3)
            mapped_peaks = 1/slope*peaks1 - intercept/slope

            fig = plt.figure(figsize=(14,7))
            fig.suptitle("Debug: ac_vector: indeces=" + str(indeces))

            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.set_title("Index Mapping") 
            ax2.set_title("Mapped Curves") 

            x_  = [ 0, size2-1 ]
            y_  = [ slope*z + intercept  for z in x_ ]
            ax1.plot( x_, y_, color='green' )

            for px, py, pw in zip(x, y, w):
                # color = 'red' if pw > weight_boundary else 'black'
                color = 'red'
                msize = 10 if pw > weight_boundary else 5
                ax1.plot(px, py, 'o', color=color, markersize=msize)

            ax2.plot(i, source_y, color=colors[1], label='source')
            ax2.plot(i, mapped_y, color=colors[0], label='mapped')
            ax2.plot(i, mapped_y_, ':', color=colors[0], label='mapped')

            if len(unmapped_index) > 0:
                mapped_peaks_ = mapped_peaks[unmapped_index]
                ax2.plot(mapped_peaks_, mapped_spline(mapped_peaks_), 'o', color='yellow', markersize=10, label='unmapped ' + points_label)

            """
            TO BE FIXED:
            there was an error occured in 20170511
            IndexError: index 1 is out of bounds for axis 0 with size 1
            """
            print('mapped_peaks=', mapped_peaks, 'indeces=', indeces)
            mapped_peaks_ = mapped_peaks[indeces[0]]
            ax2.plot(mapped_peaks_, mapped_spline(mapped_peaks_), 'o', color='red', markersize=10, label='mapped ' + points_label)

            ax2.plot(peaks2, curve2.spline(peaks2), 'o', color='pink', markersize=10, label='source ' + points_label)

            poly_points = list(zip( i, source_y )) + list( reversed( list( zip( i, mapped_y ) ) ) )
            diff_poly   = Polygon( poly_points, alpha=0.2 )
            ax2.add_patch(diff_poly)

            xmin, xmax = ax1.get_xlim()
            ymin, ymax = ax1.get_ylim()
            tx = (xmin + xmax)/2
            ty = (ymin + ymax)/2
            ax1.text(tx, ty, "Linearity=%.3g" % correlation, ha='center', va='center', fontsize=60, alpha=0.2)

            xmin, xmax = ax2.get_xlim()
            ymin, ymax = ax2.get_ylim()
            tx = (xmin + xmax)/2
            ty = (ymin + ymax)/2
            ax2.text(tx, ty, "Similarity=%.3g" % similarity, ha='center', va='center', fontsize=60, alpha=0.2)

            ax2.legend()
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            plt.show()

        return correlation, similarity

    def get_best_mapping_params(self, curve1, curve2, peak_mapping_only=False):
        if peak_mapping_only:
            peaks1  = np.array( [ info[1] for info in curve1.get_major_peak_info(ratio=0.3) ] )     # ratio=0.1 is too small for 20190221_2
            peaks2  = np.array( [ info[1] for info in curve2.get_major_peak_info(ratio=0.3) ] )     # 
        else:
            peaks1  = np.array( [ info[1] for info in curve1.peak_info ] )
            peaks2  = np.array( [ info[1] for info in curve2.peak_info ] )

        size1   = len(curve1.y)
        size2   = len(curve2.y)
        num_peaks1 = len(peaks1)
        num_peaks2 = len(peaks2)

        greater_num = max(num_peaks1, num_peaks2)
        less_num = min(num_peaks1, num_peaks2)

        greater_index = list(range(greater_num))
        full_index1 = list(range(num_peaks1))
        full_index2 = list(range(num_peaks2))

        max_score   = None
        max_params  = None
        max_indeces = None
        max_correl  = None
        max_simil   = None
        max_info    = None

        for c in combinations(greater_index, less_num):
            index = list(c)
            if num_peaks1 >= num_peaks2:
                index1 = index
                index2 = full_index2
            else:
                index1 = full_index1
                index2 = index
            assert len(index1) == len(index2)
            correl, simil, params, info = self.evaluate_mapping( size1, size2, curve1, peaks1, index1, curve2, peaks2, index2 )
            score = correl + simil
            if max_score is None or score > max_score:
                max_score   = score
                max_params  = params
                max_indeces = (index1, index2)
                max_correl  = correl
                max_simil   = simil
                max_info    = info

        return max_score, max_params, max_indeces, max_correl, max_simil, max_info

    def get_best_mapping_features_params( self, curve1, curve2, mapped_info ):

        major_peak_info1 = curve1.get_major_peak_info()
        major_peak_info2 = curve2.get_major_peak_info()

        adopted_peak_info1 = []
        adopted_peak_info2 = []
        best_params = mapped_info[0]
        A, B = best_params
        A_, B_ = 1/A, -B/A

        size2 = len(curve2.x)
        for rec1 in major_peak_info1:
            top1 = rec1[1]
            mapped_top1 = top1*A_ + B_
            for rec2 in major_peak_info2:
                top2 = rec2[1]
                if abs(mapped_top1 - top2)/size2 < 0.05:
                    adopted_peak_info1.append(rec1)
                    adopted_peak_info2.append(rec2)
                    break

        points1, weights1 = curve1.feature.get_points(adopted_peak_info1)
        points2, weights2 = curve2.feature.get_points(adopted_peak_info2)

        num_points1 = len(points1)
        num_points2 = len(points2)

        diff_num = abs(num_points1 - num_points2)

        if REDUCE_WHEN_INFEASIBLE and diff_num > FEASIBLE_DIFF_NUM:
            """
            not yet usable for 20170426
            """
            print('num_points1=', num_points1, 'num_points2=', num_points2)

            A_, B_ = 1/A, -B/A
            self.logger.warning("reducing points: %d vs. %d." % (num_points1, num_points2))

            def reduce_num_points(scale, shift, more_points, more_weights, less_points, less_weights):
                np1 = len(more_points)
                np2 = len(less_points)
                min_distances = [np.min( np.abs( scale*p+shift - less_points) ) for p in more_points]
                k = np1 - (np2 + FEASIBLE_DIFF_NUM)
                ppp = np.argpartition(min_distances, k)
                adopted_index = ppp[k:]
                print('adopted_index=', adopted_index)
                ret_points = more_points[adopted_index]
                ret_weights = more_weights[adopted_index]
                return ret_points, ret_weights

            if num_points1 > num_points2:
                points1, weights1 = reduce_num_points(A_, B_, points1, weights1, points2, weights2)
                num_points1 = len(points1)
            else:
                points2, weights2 = reduce_num_points(A, B, points2, weights2, points1, weights1)
                num_points2 = len(points2)

            diff_num = num_points1 - num_points2
            self.logger.warning("reduced points: %d vs. %d." % (num_points1, num_points2))

            if False:
                names = ['UV', 'Xray']
                colors = ['blue', 'orange']

                plt.push()
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,7))
                ax1, ax2 = axes
                simple_plot(ax1, curve1, names[0], color=colors[0], legend=False)
                simple_plot(ax2, curve2, names[1], color=colors[1], legend=False)
                ax1.plot(points1, curve1.spline(points1), 'o', color='yellow')
                ax2.plot(points2, curve2.spline(points2), 'o', color='yellow')

                mapped_points = A_*points1+B_
                ax2.plot(mapped_points, curve2.spline(mapped_points), 'o', color='cyan')
                fig.tight_layout()
                plt.show()
                plt.pop()

        assert diff_num <= FEASIBLE_DIFF_NUM

        greater_num = max(num_points1, num_points2)
        less_num = min(num_points1, num_points2)

        greater_index = list(range(greater_num))
        full_index1 = list(range(num_points1))
        full_index2 = list(range(num_points2))

        max_score   = None
        max_params  = None
        max_indeces = None
        max_correl  = None
        max_simil   = None
        max_info    = None

        for c in combinations(greater_index, less_num):
            index = list(c)
            if num_points1 >= num_points2:
                index1 = index
                index2 = full_index2
            else:
                index1 = full_index1
                index2 = index
            try:
                correl, simil, params, info = self.evaluate_mapping_features(curve1, points1, weights1, index1, curve2, points2, weights2, index2)
            except AssertionError:
                # invalid combination
                if False:
                    log_exception(self.logger, "evaluate_mapping_features failure: ")
                continue

            score = correl + simil
            if max_score is None or score > max_score:
                max_score   = score
                max_params  = params
                max_indeces = (index1, index2)
                max_correl  = correl
                max_simil   = simil
                max_info    = info

        pre_recog_covered = False
        if max_score is None:
            if self.pre_recog is None:
                assert False
            else:
                best_params, max_score_pair, max_info = self.pre_recog.get_restricted_mapped_info()
                a, b = best_params
                max_params = np.array([b, a])
                x, y, w = max_info
                correl, simil = self.compute_mapped_curve_simimarity(max_params, x, y, w, None, curve1, curve2, features=True, nodebug=True)
                max_score = correl + simil
                # max_score = 2
                max_corel = correl
                max_simil = simil
                pre_recog_covered = True
                self.logger.warning("to cover the failure, feature matching params are replaced with the pre-recognized params.")

        return max_score, max_params, max_indeces, max_correl, max_simil, max_info

    def find_best_mapping_impl(self, curve1, curve2, features, check_adequacy=True, mapped_info=None, debug=False):

        num_peaks_1 = len(curve1.peak_info)
        num_peaks_2 = len(curve2.peak_info)
        assert num_peaks_1 == len(curve1.boundaries) + 1
        assert num_peaks_2 == len(curve2.boundaries) + 1

        debug = False
        # if debug and self.debug:
        if debug:
            def best_debug_plot(title, curve1, curve2, feature_info=None, orig_curves=None):
                from DataUtils import get_in_folder
                reversed_=False
                plt.push()
                fig, axes = plt.subplots( figsize=(21, 8), nrows=1, ncols=2 )
                fig.suptitle(title + ' for ' + get_in_folder(), fontsize=30)
                ax1, ax2 = axes
                if reversed_:
                    c1, c2 = curve2, curve1
                else:
                    c1, c2 = curve1, curve2

                if orig_curves is None:
                    c1_orig = c1
                    c2_orig = c2
                else:
                    if reversed_:
                        c2_orig, c1_orig = orig_curves
                    else:
                        c1_orig, c2_orig = orig_curves

                points1, weights1 = c1_orig.feature.get_points()
                points2, weights2 = c2_orig.feature.get_points()
                ax1.plot(points1, c1_orig.spline(points1), 'o', color='yellow', markersize=20, label='original feature points')
                ax2.plot(points2, c2_orig.spline(points2), 'o', color='yellow', markersize=20, label='original feature points')

                if orig_curves is not None and feature_info is not None:
                    points1, weights1 = c1.feature.get_points()
                    points2, weights2 = c2.feature.get_points()
                    alt_index, alt_params = feature_info
                    if len(points1) < len(points2):
                        index1 = list(range(len(points1)))
                        index2 = alt_index
                    else:
                        index1 = alt_index
                        index2 = list(range(len(points2)))
                    points1_ = points1[index1]
                    points2_ = points2[index2]
                    ax1.plot(points1_, c1.spline(points1_), 'o', color='cyan', markersize=15, label='mapped feature points')
                    ax2.plot(points2_, c2.spline(points2_), 'o', color='cyan', markersize=15, label='mapped feature points')

                    # mapped_points = points_[alt_index]
                    # ax_.plot(mapped_points, c_.spline(mapped_points), color='cyan', markersize=10, label='mapped points')

                simple_plot(ax1, c1, "UV", color='blue', legend=False)
                simple_plot(ax2, c2, "Xray", color='orange', legend=False)

                ax1.legend(fontsize=16)
                ax2.legend(fontsize=16)
                fig.tight_layout()
                fig.subplots_adjust(top=0.85)
                plt.show()
                plt.pop()
            best_debug_plot("find_best_mapping begin", curve1, curve2)

        feature_info = None

        if features:
            try:
                max_score, max_params, max_indeces, max_correl, max_simil, max_info = self.get_best_mapping_features_params(curve1, curve2, mapped_info=mapped_info)
                feature_info = [ max_indeces, max_params ]
                B, A = max_params
                best_params = np.array([A, B])
                self.logger.info("get_best_mapping_features_params has been successful: A=%g, B=%g" % (A, B))
            except:
                self.logger.warning(" " + etb.last_lines())
                log_exception(self.logger, "get_best_mapping_features_params failed: ")
                max_score, best_params, max_indeces, max_correl, max_simil, max_info = 0, None, None, None, None, None
            curve1_, curve2_ = curve1, curve2
        else:
            index1 = list(range(num_peaks_1))
            max_score, max_params, max_indeces, max_correl, max_simil, max_info = self.get_best_mapping_params(curve1, curve2)
            # curve1_, curve2_, best_params = self.reconstruct_curves(curve1, curve2, max_params, max_indeces, max_correl, max_simil, check_adequacy)
            curve1_, curve2_, best_params = self.reconstruct_curves_simple(curve1, curve2, max_params, max_indeces,)

        best_info = [max_score, best_params, max_indeces, max_correl, max_simil]
        mapped_info = MappedInfo([best_params, [max_correl, max_simil], max_info], [self.a_curve, self.x_curve])

        # if self.debug and debug:
        if debug:
            best_debug_plot("find_best_mapping result", curve1_, curve2_, feature_info, orig_curves=[curve1, curve2])

        return curve1_, curve2_, best_info, mapped_info

    def reconstruct_curves_simple(self, curve1, curve2, max_params, max_indeces):
        B, A = max_params
        A_  = 1/A
        B_  = -B/A

        new_peak_top_x1 = []
        new_peak_info1 = []
        new_boundaries1 = []
        new_peak_top_x2 = []
        new_peak_info2 = []
        new_boundaries2 = []

        index1, index2 = max_indeces
        prev_pair = None
        for n, pair in enumerate(zip(index1, index2)):
            m, k = pair

            top_x1 = curve1.peak_top_x[m]
            peak_rec1 = curve1.peak_info[m]
            new_peak_top_x1.append(top_x1)
            new_peak_info1.append(peak_rec1)

            top_x2 = curve2.peak_top_x[k]
            peak_rec2 = curve2.peak_info[k]
            new_peak_top_x2.append(top_x2)
            new_peak_info2.append(peak_rec2)

            if n > 0:
                boundary1, boundary2 = self.find_valley_bottom_pair(curve1, curve2, prev_pair, m, k, A_, B_)
                new_boundaries1.append(boundary1)
                new_boundaries2.append(boundary2)

            prev_pair = pair

        assert len(new_peak_info1) == len(new_boundaries1) + 1
        assert len(new_peak_info2) == len(new_boundaries2) + 1
        assert len(new_peak_info1) == len(new_peak_info2)

        curve1_ =  ProxyCurve( curve1, new_peak_top_x1, new_peak_info1, new_boundaries1, [], [] )
        curve2_ =  ProxyCurve( curve2, new_peak_top_x2, new_peak_info2, new_boundaries2, [], [] )

        best_params = np.array([A, B])
        return curve1_, curve2_, best_params

    def find_valley_bottom_pair(self, curve1, curve2, prev_pair, m, k, A_, B_):
        prev_m, prev_k = prev_pair
        bc1_list = curve1.boundaries[prev_m:m]
        bc2_list = curve2.boundaries[prev_k:k]

        # select the best matching bottom pair
        min_dist = None
        min_j = None
        min_i = None
        for j, bc1 in enumerate(bc1_list):
            for i, bc2 in enumerate(bc2_list):
                dist = abs(bc1*A_ + B_ - bc2 )
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_j = j
                    min_i = i

        return bc1_list[min_j], bc2_list[min_i]

    def add_corresponding_peaks(self, curve1, curve2, A_, B_, debug_text=""):
        debug = True
        if debug:
            from matplotlib.patches import Rectangle
            from DataUtils import get_in_folder
            print('A_, B_=', A_, B_)
            plt.push()
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
            ax1, ax2 = axes
            fig.suptitle("add_corresponding_peaks entry: " + debug_text, fontsize=20)
            simple_plot(ax1, curve1, "curve1", color="blue")
            simple_plot(ax2, curve2, "curve2", color="orange")

            ymin, ymax = ax2.get_ylim()
            ax2.set_ylim(ymin, ymax)
            allow = len(curve2.x) * 0.05
            for top_x in curve2.peak_top_x:
                f = top_x - allow
                t = top_x + allow
                p = Rectangle(
                        (f, ymin),  # (x,y)
                        t - f,   # width
                        ymax - ymin,    # height
                        facecolor   = 'yellow',
                        alpha       = 0.3,
                    )
                ax2.add_patch(p)

            xmin, xmax = ax2.get_xlim()
            ax2t = ax2.twinx()
            ax2t.grid(False)
            ax2t.set_xlim(xmin, xmax)
            mapped_x = A_ * curve1.x  + B_
            ax2t.plot(mapped_x, curve1.y, ':', color='blue', label='mapped data')
            for top_x in curve1.peak_top_x:
                mx = A_ * top_x + B_
                my = curve1.spline(top_x)
                ax2t.plot(mx, my, 'o', color='yellow', label='mapped peak tops')

            ax2t.legend(bbox_to_anchor=(1, 0.9), loc='upper right')
            fig.tight_layout()
            plt.show()
            plt.pop()

        new_peak_top_x2 = []
        new_peak_info2 = []
        new_boundaries2 = []

        size2 = len(curve2.x)
        peaks2 = []
        start = 0
        stop = None
        last_peak_rec2 = None
        for m, top_x1 in enumerate(curve1.peak_top_x):
            mapped_top_x = A_*top_x1 + B_

            bottom_added = False
            if m > 0 and len(new_peak_top_x2) > 0:
                try:
                    bottom_x1 = curve1.boundaries[m-1]
                    mapped_bottom_x = A_*bottom_x1 + B_
                    bottom_x2 = self.find_counter_part_bottom(curve2, mapped_bottom_x, debug)
                    new_boundaries2.append(bottom_x2)
                    bottom_added = True
                except AssertionError:
                    pass

            try:
                mapped_peak_rec = [int(A_*i + B_ + 0.5) for i in curve1.peak_info[m]]
                top_x2, peak_rec2 = self.find_counter_part_peak(curve2, mapped_top_x, mapped_peak_rec, debug)
                new_peak_top_x2.append(top_x2)
                if last_peak_rec2 is not None:
                    if last_peak_rec2[2] >= peak_rec2[0]:
                        b = new_boundaries2[-1]
                        r = peak_rec2[0]
                        peak_rec2[0] = int((last_peak_rec2[2] + b)/2)
                        self.logger.warning("the left end modified from %d to %d in the %d-th range in add_corresponding_peaks", m, r, peak_rec2[0])
                new_peak_info2.append(peak_rec2)
                last_peak_rec2 = peak_rec2
            except AssertionError:
                warnlog_exception(self.logger, "add_corresponding_peaks: try peak")
                if bottom_added:
                    new_boundaries2.pop(-1)

            if debug:
                print("--------------------", [m], "new_peak_info2=", new_peak_info2)

        # assert len(new_peak_top_x2) == len(new_boundaries2) + 1

        if len(new_peak_top_x2) == len(new_boundaries2) + 1:
            curve2_ =  ProxyCurve( curve2, new_peak_top_x2, new_peak_info2, new_boundaries2, [], [] )
        else:
            curve2_ = curve2

        if debug:
            print('len(new_peak_top_x2)=', len(new_peak_top_x2))
            print('len(new_boundaries2)=', len(new_boundaries2))
            plt.push()
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
            ax1, ax2 = axes
            fig.suptitle("add_corresponding_peaks exit: " + debug_text)
            simple_plot(ax1, curve1, "curve1")
            simple_plot(ax2, curve2_, "curve2_")
            fig.tight_layout()
            plt.show()
            plt.pop()

        return curve2_

    def find_the_nearest_index(self, curve, vecx, px):
        """
        must choose the nearest since diff_ratio < 0.05 can detect multiple candidates
        """
        pr_width = curve.get_peak_region_width()

        min_ratio = None
        min_k = None
        for k, x_ in enumerate(vecx):
            diff_ratio = abs(px - x_)/pr_width
            if diff_ratio < 0.05:
                if min_ratio is None or diff_ratio < min_ratio:
                    min_k = k
                    min_ratio = diff_ratio

    def find_counter_part_peak(self, curve, ref_peak_x, ref_peak_rec, debug=False):
        k = self.find_the_nearest_index(curve, curve.peak_top_x, ref_peak_x)
        if k is not None:
            if debug:
                print('find_counter_part_peak: found', k)
            return curve.peak_top_x[k], curve.peak_info[k]

        delta_x = len(curve.x)*0.05
        start = int(ref_peak_x - delta_x)
        stop  = int(ref_peak_x + delta_x)

        n = find_rotated_extreme_arg(curve, start, stop, +1)

        # avoid minor peaks such as the point near 650 in 20170304 UV
        # default delta_x is too small for the 1st minor peak in 20160227
        assert curve.locally_tall_enough_evenif_rotated(n, delta_x=delta_x)

        if debug:
            print('-------------------- find_counter_part_peak: found rotated', n)

        return n, [ref_peak_rec[0], n, ref_peak_rec[2]]

    def find_counter_part_bottom(self, curve, ref_bottom_x, debug=False):
        k = self.find_the_nearest_index(curve, curve.boundaries, ref_bottom_x)
        if k is not None:
            return curve.boundaries[k]

        delta_x = len(curve.x)*0.05
        start = int(ref_bottom_x - delta_x)
        stop  = int(ref_bottom_x + delta_x)

        n = find_rotated_extreme_arg(curve, start, stop, -1)

        if debug:
            print('-------------------- find_counter_part_bottom: found rotated', n)

        return n

    def show_mapping_proof(self):
        a_curve_, x_curve_  = self.get_mapped_curves()
        mapping_proof("mapping proof", a_curve_, x_curve_, self.best_info[1])

    def make_better_curve(self, curve1, curve2, mapper, mapping, debug_text, debug=False):
        nomatch = mapper[mapper < 0]
        if len(nomatch) == 0:
            return curve1

        A, B = mapping

        new_peak_top_x = []
        new_peak_info = []
        new_boundaries = []

        max_y = curve1.max_y
        ptx1 = curve1.peak_top_x
        pif1 = curve1.peak_info
        bdy1 = curve1.boundaries
        ptx_ = curve2.peak_top_x*A + B
        bry_ = np.array(curve2.boundaries)*A + B

        if debug:
            print("mapper=", mapper)
            if debug_text.find("a_curve") >= 0:
                curve1_id = "UV"
                curve2_id = "XR"
            else:
                curve1_id = "XR"
                curve2_id = "UV"
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("%s entry" % debug_text)
                ax1.set_title(curve1_id + " Elution")
                ax2.set_title(curve2_id + " Elution")
                simple_plot(ax1, curve1, legend=False)

                pty_ = curve1.spline(ptx_)
                ax1.plot(ptx_, pty_, "o", color="cyan", markersize=20, alpha=0.2, label="mapped peak")
                ax1.legend()

                simple_plot(ax2, curve2)
                fig.tight_layout()
                plt.show()

        ins = 0
        last_peak_info = None
        def adjusted_peak_info(temp_info):
            if last_peak_info is None or last_peak_info[2] < temp_info[0]:
                return temp_info

            ret_info = [last_peak_info[2]+1, *temp_info[1:]]
            assert ret_info[0] < ret_info[1]

            return ret_info

        try:
            for k, j in enumerate(mapper):
                if j >= 0:
                    continue
                x_ = ptx_[k]
                i = bisect_right(ptx1, x_)
                print([k], "i=", i)
                n = 0
                for p in range(ins, i):
                    new_peak_top_x.append(ptx1[p])
                    peak_info = adjusted_peak_info(pif1[p])
                    new_peak_info.append(peak_info)
                    last_peak_info = peak_info
                    if p > 0:
                        new_boundaries.append(bdy1[p-1])
                    n += 1
                ins += n
                top_x = ptx_[i]

                height_ratio = curve1.spline(top_x)/max_y
                # print([k], "-------------------- height_ratio=", height_ratio)
                if height_ratio < MINIMUM_HEIGHT_RATIO:
                    self.logger.info("peak with height_ratio=%.3g < %.3g will be discarded.", height_ratio, MINIMUM_HEIGHT_RATIO)
                    # eventually resorting to make_thinner_curve
                    assert False

                new_peak_top_x.append(top_x)
                peak_info = adjusted_peak_info([int(round(p*A + B)) for p in curve2.peak_info[i]])
                new_peak_info.append(peak_info)
                last_peak_info = last_peak_info
                if len(bry_) > 0:
                    new_boundaries.append(bry_[max(0, i-1)])    # task: prove max(0, i-1) is correct
        except AssertionError as exc:
            if debug:
                mx_ = curve2.x*A + B
                scale = curve1.max_y/curve2.max_y
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.set_title("adjusted_peak_info AssertionError")
                    simple_plot(ax, curve1, legend=False)
                    ax.plot(mx_, scale*curve2.y, ":", label="mapped")
                    ax.plot(ptx_, scale*curve2.spline(curve2.peak_top_x), "o", markersize=20, color="cyan", alpha=0.2)
                    ax.legend()
                    fig.tight_layout()
                    plt.show()
            raise exc

        for p in range(ins, len(ptx1)):
            new_peak_top_x.append(ptx1[p])
            new_peak_info.append(pif1[p])
            if p > 0:
                new_boundaries.append(bdy1[p-1])

        curve1_ =  ProxyCurve( curve1, new_peak_top_x, new_peak_info, new_boundaries, [], [] )

        if debug:
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("%s result" % debug_text)
                simple_plot(ax1, curve1_, legend=False)

                pty_ = curve1_.spline(ptx_)
                ax1.plot(ptx_, pty_, "o", color="cyan", markersize=20, alpha=0.2, label="mapped peak")
                ax1.legend()

                simple_plot(ax2, curve2)
                fig.tight_layout()
                plt.show()

        return curve1_

    def make_thinner_curve(self, curve, mapper):
        nomatch = mapper[mapper < 0]
        if len(nomatch) == 0:
            return curve

        new_peak_top_x = []
        new_peak_info = []
        new_boundaries = []

        ptx = curve.peak_top_x
        pif = curve.peak_info
        bdy = curve.boundaries

        n = 0
        for k, j in enumerate(mapper):
            if j < 0:
                # primary_peak must not be lost
                if k == curve.primary_peak_no:
                    raise AssertionError("primary peak lost")
                continue

            new_peak_top_x.append(ptx[k])
            new_peak_info.append(pif[k])
            if n > 0 and k > 0:
                new_boundaries.append(bdy[k-1])
            n += 1

        curve_ =  ProxyCurve( curve, new_peak_top_x, new_peak_info, new_boundaries, [], [] )
        return curve_

    def can_be_deemed_single_peak(self, curves):
        if len(curves[0].peak_info) == 1:
            return True

        num_peaks = []
        for curve in curves:
            n = 0
            max_y = curve.max_y
            for info in curve.peak_info:
                ratio = curve.spline(info[1])/max_y
                if ratio > DEEM_SINGLE_PEAK_RATIO:
                    n += 1
            num_peaks.append(n)
        return np.min(num_peaks) == 1

    def get_imroved_mapping(self, fallback=False, debug=False):
        from LPM import get_corrected   # moved due to ImportError: ... (most likely due to a circular import)

        if debug:
            from DebugUtils import show_call_stack
            show_call_stack("----: ", indented_only=True)            

        curves = self.mapped_curves
        A, B = self.best_info[1]

        if not self.can_be_deemed_single_peak(curves):
            return A, B

        max_simil = self.best_info[4]
        if len(curves[0].peak_info) == 1:
            uv_px = curves[0].peak_info[0][1]
            mp_px = A * curves[1].peak_info[0][1] + B
            dx = uv_px - mp_px
            self.logger.info("an adjustment %.3g of peak top deviation will be applied.", dx)
        else:
            dx = 0

        B += dx
        if max_simil < ACCEPTABLE_SIMILARITY:
            in_fallback = " in fallback" if fallback else ""
            self.logger.info("trying to improve to mapping due to poor similarity %.3g < %.3g%s.", max_simil, ACCEPTABLE_SIMILARITY, in_fallback)
        else:
            return A, B

        x1 = curves[0].x
        y1 = get_corrected(curves[0].y)
        i1 = curves[0].primary_peak_i
        x2 = curves[1].x
        y2 = get_corrected(curves[1].y)
        i2 = curves[1].primary_peak_i

        spline = UnivariateSpline(x1, y1, s=0, ext=3)
        scale = y1[i1]/y2[i2]

        def objective(p):
            a, b = p
            x = a*x2 + b
            return np.sum((spline(x) - scale*y2)**2)

        ret = minimize(objective, (A, B), method="Nelder-Mead")
        a, b = ret.x

        mapped_ends = [ex*a + b for ex in x2[[0, -1]]]
        ratio = (mapped_ends[1] - mapped_ends[0])/(x1[-1] - x1[0])
        if abs(ratio - 1) > ACCEPTABLE_ENDS_RATIO_DEV:
            # as in 20200123_4
            self.logger.info("mapping (%.3g, %.3g) is discarded due to ratio=%.3g", a, b, ratio)
            a, b = A, B

        if debug:
            print("fallback=", fallback)
            spline2 = UnivariateSpline(x2, y2, s=0, ext=3)
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("get_imroved_mapping")
                ax1.set_title("Uncertain Mapping")
                ax1.plot(x1, y1)
                for info in curves[0].peak_info:
                    px = info[1]
                    ax1.plot(px, spline(px), "o", color="red")
                ax1.plot(A*x2+B, scale*y2, ":")
                for info in curves[1].peak_info:
                    px = info[1]
                    ux = A*px + B
                    ax1.plot(ux, scale*spline2(px), "o", color="cyan", markersize=20, alpha=0.3)

                def draw_slice(ax, slice_):
                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim(ymin, ymax)
                    for j in [slice_.start, slice_.stop]:
                        ax.plot([j, j], [ymin, ymax], ":", color="yellow")

                draw_slice(ax1, curves[0].peak_slice)

                ax2.set_title("Improved Mapping")
                ax2.plot(x1, y1)
                for info in curves[0].peak_info:
                    px = info[1]
                    ax2.plot(px, spline(px), "o", color="red")
                ax2.plot(a*x2+b, scale*y2, ":")
                for info in curves[1].peak_info:
                    px = info[1]
                    ux = a*px + b
                    ax2.plot(ux, scale*spline2(px), "o", color="cyan", markersize=20, alpha=0.3)

                draw_slice(ax2, curves[0].peak_slice)

                fig.tight_layout()
                plt.show()

        return a, b

    def compare_primary_peak_mapping_params(self, curve1, curve2, mapping, debug=False):
        px1 = curve1.primary_peak_x
        px2 = curve2.primary_peak_x
        scale = curve1.spline(px1)/curve2.spline(px2)
        uv_x = curve1.x
        x2 = curve2.x

        def objective(p, add_no_mapping=False, debug=False):
            a, b, w = p
            if w <= 1e-3:
                return np.inf

            x1 = a*x2 + b
            compare_interval = np.logical_and(px1 - w <= x1, x1 <= px1 + w)
            x1_ = x1[compare_interval]
            x2_ = x2[compare_interval]
            dev = np.sum((curve1.spline(x1_) - scale*curve2.spline(x2_))**2)

            if add_no_mapping:
                # add penalty from region with no mapping as in OA_Ald with different primary peaks
                if x1[0] < uv_x[0]:
                    extend_L = np.arange(x1[0], uv_x[0], 1)   # uv_x[1] - uv_x[0] == 1
                else:
                    extend_L = []
                if uv_x[-1] < x1[-1]:
                    extend_R = np.arange(uv_x[-1]+1, x1[-1]+1, 1)
                else:
                    extend_R = []
                ex_uv_x = np.concatenate([extend_L, uv_x, extend_R])
                penalty = np.sum(curve1.spline(ex_uv_x[ex_uv_x < x1[0]])**2) + np.sum(curve1.spline(ex_uv_x[ex_uv_x > x1[-1]])**2)
            else:
                penalty = 0

            if debug:
                print("dev/w=", dev/w)
                print("penalty=", penalty)
                with plt.Dp():
                    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                    fig.suptitle("objective debug")

                    simple_plot(ax1, curve1, legend=False)
                    simple_plot(ax2, curve2, legend=False)

                    y1_ = curve1.spline(x1_)
                    y2_ = curve2.spline(x2_)
                    ax1.plot(x1_, y1_, ":", label="compare_interval")
                    ax2.plot(x2_, y2_, ":", label="compare_interval")
                    ax1.fill_between(x1_, y1_, scale*y2_, color='pink', alpha=0.2)

                    for ax in ax1, ax2:
                        ax.legend()

                    fig.tight_layout()
                    plt.show()

            return dev/w + penalty

        a_init = len(curve1.x)/len(curve2.x)
        b_init = px1 - a_init*px2
        pinfo = curve1.peak_info[curve1.primary_peak_no]
        w_init = (pinfo[2] - pinfo[0])/2
        ret = minimize(objective, (a_init, b_init, w_init))
        a, b, w = ret.x
        if debug:
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("compare_primary_peak_mapping_params")
                simple_plot(ax, curve1)
                x1 = a*x2 + b
                ax.plot(x1, scale*curve2.y, ":")
                fig.tight_layout()
                plt.show()

        W = len(curve1.x)
        primary_score = objective((a, b, W), add_no_mapping=True)
        A, B = mapping
        compare_score = objective((A, B, W), add_no_mapping=True)
        if primary_score >= compare_score:
            self.logger.info("adopt normal peak mapping with score comparison %.3g <= %.3g", compare_score, primary_score)
            a, b = A, B
        else:
            self.logger.info("adopt primary peak mapping with score comparison %.3g < %.3g", primary_score, compare_score)

        return a, b
