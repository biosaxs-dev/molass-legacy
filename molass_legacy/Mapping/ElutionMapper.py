"""

    ElutionMapper.py

        Optimization of mapping between UV-absorbance and Xray-scattering

    Copyright (c) 2018-2024, SAXS Team, KEK-PF

"""
import copy
import numpy                as np
from scipy.interpolate      import UnivariateSpline
from scipy import optimize
from scipy.spatial import distance
import logging
from molass_legacy.KekLib.BasicUtils import Struct
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve, PEAK_INFO_FIND_RATIO
from molass_legacy.Elution.CurveUtils      import get_probably_corresponding_index
from molass_legacy._MOLASS.SerialSettings         import ( get_setting, reset_setting, set_setting,
                                        UV_BASE_NO, UV_BASE_CONST,
                                        UV_BASE_STANDARD, UV_BASE_SHIFTED,
                                        XRAY_BASE_NO, XRAY_BASE_CONST,
                                        INTEGRAL_BASELINE
                                        )
from .MappingParams import set_mapper_opt_params, get_mapper_opt_params, MappedInfo
from molass_legacy.SerialAnalyzer.ScatteringBaseCorrector import compute_baseline_using_LPM_impl
from molass_legacy.Baseline.ScatteringBaseline import ScatteringBaseline
from molass_legacy.KekLib.CanvasDialog import CanvasDialog
from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker, log_exception
# from molass_legacy.SerialAnalyzer.AnimObjects import AnimMapper
from .PeakMapper import PeakMapper
from .SingleComponent       import SingleComponent, PEAK_EVAL_RANGE_RATIO
from molass_legacy.UV.XrayProportional import make_proportional_mapping_info_impl
import molass_legacy.KekLib.DebugPlot as plt

# EQUALIZE_OUTSIDE_OF_FC  = True
ZEROS_TO_OUTSIDE_OF_FC  = True
MAPPED_RATIO_ALLOW      = 0.5       # > 0.21 for 20180225, > 0.35 for 20160628
ALLOWANCE_PERCENT       = 10        # > 5 for 20161119/Kosugi3a
SCALE_MIN_RATIO         = 0.7
SCALE_MAX_RATIO         = 1.4
AJUSTMENT_ITERATION     = 2
XRAY_INTERCEPT_PERCENT  = 5
USE_TOTAL_STD_DIFF      = True
USE_WEIGHTED_DIFF       = True
STD_DIFF_SCALE          = 10        # this value is for the case where USE_TOTAL_STD_DIFF == True
VALLEY_DIFF_WEIGHT      = 0.5
MAPPING_RANGE_RATIO     = PEAK_INFO_FIND_RATIO
MAPPING_RANGE_DELTA     = 0.03      # adjustment for cases such as 20161202
USE_SAVED_INFO_TO_GET_RANGES    = False
REFINE_MAPPING_RANGES   = True
USE_SIMPLE_SCALE_FOR_VALLEY = False
SMALL_VALUE_LIMIT       = 1e-5
USE_SAME_RATIO_BOUNDARY = True
ACCEPTABLE_STD_DIFF     = 1.0       # 0.63 for Kosugi3a
REQUIRE_HELPER_STD_DIFF = 1.0
VERY_SMALL_RATIO        = 1e-5
USE_AREA_SCALING        = True
PENALTY_KEEP_POS_VALUE  = 1e-5
PENALTY_KEEP_POS_RATIO  = 0.03
USE_SIMPLE_MAPPING      = True

TO_BE_REMOVED   = False

"""
refactoring memo

    optimize
        optimize_mapping

        adjust_baselines
            adjust_uv_baseline_only
                minimize
            adjust_xray_baseline_only
                minimize
            adjust_both_baselines
                minimize
"""

class ElutionMapper:
    def __init__(self, serial_data, input_sd, pre_recog, callbacks=None):
        self.logger = logging.getLogger( __name__ )

        assert serial_data.orig_info is None

        self.orig_sd = serial_data
        self.input_sd = input_sd
        self.pre_recog = pre_recog
        self.initialize(self.orig_sd)

        self.num_iterations = get_setting( 'correction_iteration' )
        self.use_xray_conc  = get_setting('use_xray_conc')
        self.use_mtd_conc  = get_setting('use_mtd_conc')
        self.enable_lrf_baseline = get_setting('enable_lrf_baseline')
        self.baseline_corrected_copy = None
        self.sd_update_callback = None
        self.optimize_callback = None
        self.x_curve_update_callback = None
        if callbacks is not None:
            self.sd_update_callback = callbacks[0]
            self.optimize_callback = callbacks[1]
            self.x_curve_update_callback = callbacks[2]

        self.logger.info('mapper has been created with conc_type %d.' % self.get_conc_type() )

        if False:
            fig = plt.figure(figsize=(12,6))
            fig.suptitle("ElutionMapper.__init__")
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.plot(self.a_vector)
            ax1.plot(self.a_spline(self.a_x))
            ax2.plot(self.x_vector, color='orange')
            fig.tight_layout()
            plt.show()

    def set_sd_update_callback(self, callback):
        self.sd_update_callback = callback

    def set_optimize_callback(self, callback):
        self.optimize_callback = callback

    def set_x_curve_update_callback(self, callback):
        self.x_curve_update_callback = callback

    def initialize(self, sd):
        self.serial_data = sd.get_copy()
        self.top_x_pair = sd.get_original_peak_top_x_pair()
        self.orig_info  = sd.orig_info
        self.absorbance = sd.absorbance
        if self.absorbance is None:
            self.logger.error("serial_data.absorbance is None. state the recipe here.")
            raise RuntimeError("Unprepared Serial Data")

        self.sd_xray_curve = sd.xray_curve
        self.a_vector   = self.absorbance.a_vector
        self.a_x        = np.arange( len(self.a_vector) )
        self.a_spline   = UnivariateSpline( self.a_x, self.a_vector, s=0, ext=3 )
        self.x_vector   = sd.ivector
        self.x_x        = np.arange( len(self.x_vector) )
        self.x_spline   = UnivariateSpline( self.x_x, self.x_vector, s=0, ext=3 )
        self.pre_recog  = sd.pre_recog
        self.feature_mapped = False

        for info_name in ['decomp_editor_info', 'range_editor_info']:
            info_value = get_setting(info_name)
            if info_value is not None:
                reset_setting(info_name)
                self.logger.info("cleared memorized %s", info_name)

    def get_conc_type( self ):
        # TODO: unify
        if self.use_xray_conc:
            type_ = 1
        elif self.use_mtd_conc:
            type_ = 2
        else:
            type_ = 0
        return type_

    def set_opt_params(self, mapping_params):
        self.opt_params = mapping_params
        btype = mapping_params['uv_baseline_type']
        self.absorbance.compute_base_curve(self.pre_recog, btype)
        self.optimize(opt_params=self.opt_params)
        set_mapper_opt_params(self.opt_params)

    def optimize( self, opt_params=None, curve_make_only=False, sync_options=None, helper_info=None, apply_patch=False ):
        self.initialize(self.orig_sd)   # possibly required to ensure consistency in multiple optimization; making sure

        self.uniformly_scaled_vector = None
        self.sci_list = None
        self.cd_degrees = None
        self.A_simple = None

        if self.optimize_callback is not None:
            self.optimize_callback()

        if opt_params is None:
            opt_params = get_mapper_opt_params()

        self.logger.info( 'mapping optimization with parameters: ' + str(opt_params) )

        conc_type = self.get_conc_type()
        if conc_type == 1:
            make_proportional_mapping_info_impl(self, opt_params)
            return
        elif conc_type == 2:
            import Microfluidics
            from MicrofluidicMapping    import make_microfluidic_mapping_info_impl
            make_microfluidic_mapping_info_impl(self, opt_params)
            return
        else:
            pass

        self.opt_params = opt_params
        self.sync_options = sync_options
        self.make_elution_curves( opt_params, helper_info )

        self.x_curve_y_adjusted = self.x_curve.y    # for convenience for ElutinMapperCanvas.draw_mapped

        self.ensure_peak_consistency()

        if curve_make_only:
            return

        self.compute_initial_params()

        self.in_uv_adjustment_mode = False
        self.in_xray_adjustment_mode = False

        try:
            self.optimize_mapping() 
        except:
            log_exception(self.logger, "optimize_mapping failure")
            self.resort_to_simpler_solution()

        self.adjust_baselines( opt_params )

        self.set_peak_eval_ranges()
        self.compute_std_diff()

        if False:
            A_avg   = np.average( [ result[0] for result in self.opt_results ] )
            B_avg   = np.average( [ result[1] for result in self.opt_results ] )
            self.check_mapping_normality( A_avg, B_avg, std_diff=self.std_diff )

        self.scomp      = SingleComponent(self)

        self.debug_log_results()

    def debug_log_results(self):
        self.logger.info("optimize: map_params=%s, scale_params=%s", str(self.map_params), str(self.scale_params))

    def resort_to_simpler_solution(self):
        pass

    def make_elution_curves(self, opt_params, helper_info, debug=False):
        a_curve_temp = self.make_a_curve( opt_params, helper_info )
        x_curve_temp = self.make_x_curve( opt_params )

        self.peak_mapper = pm = PeakMapper( a_curve_temp, x_curve_temp, pre_recog=self.pre_recog, debug=False )
        mapped_curves   = pm.get_mapped_curves()
        self.a_curve    = mapped_curves[0]
        self.x_curve    = mapped_curves[1]
        if self.x_curve_update_callback is not None:
            self.x_curve_update_callback(self.x_curve)
        self.x_curve.get_emg_peaks(logger=self.logger)

        if debug:
            # note that mapped_curves hve been corrected

            def plot_curve(ax, curve):
                ax.plot(curve.x, curve.y)

            curves = self.orig_sd.get_elution_curves()
            with plt.Dp():
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
                fig.suptitle("make_elution_curves debug")
                plot_curve(axes[0,0], curves[0])
                plot_curve(axes[0,1], curves[1])
                plot_curve(axes[1,0], self.mapped_curves[0])
                plot_curve(axes[1,1], self.mapped_curves[1])
                fig.tight_layout()
                plt.show()

        if self.sync_options is None:
            self.feature_mapped = pm.feature_mapped
            self.manual_sync = False
        else:
            if pm.feature_mapped and self.sync_options == 1:
                self.feature_mapped = True
            else:
                self.feature_mapped = False

            if self.sync_options == 2:
                self.manual_sync = True
            else:
                self.manual_sync = False

    def make_a_curve(self, opt_params, helper_info=None, flow_change_exist=False, devel=True):
        absorbance  = self.absorbance
        uv_baseline_opt = opt_params['uv_baseline_opt']

        if uv_baseline_opt == 0:
            self.a_base = np.zeros( len(self.a_vector) )
        else:
            uv_baseline_type = opt_params['uv_baseline_type']
            self.logger.info("make_a_curve: uv_baseline_type=%d", uv_baseline_type)

            if uv_baseline_type == 1:
                self.a_base = absorbance.get_standard_elution_base()
            elif uv_baseline_type == 4:
                a_base_ok = False
                if self.enable_lrf_baseline:
                    if devel:
                        from importlib import reload
                        import Trimming.UvBaseSolver
                        reload(Trimming.UvBaseSolver)
                    from molass_legacy.Trimming.UvBaseSolver import get_lrf_elution_base
                    try:
                        self.a_base = get_lrf_elution_base(self.input_sd, self.pre_recog)
                        a_base_ok = True
                    except:
                        log_exception(self.logger, "get_lrf_elution_base failed: ", n=10)
                if not a_base_ok:
                    self.a_base = absorbance.get_shifted_elution_base()
            elif uv_baseline_type == 5:
                self.a_base = absorbance.get_integral_basecurve()
            else:
                assert False

        ac_vector       = self.a_vector - self.a_base

        debug = False

        if debug:
            # import traceback
            # traceback.print_stack()
            ax  = plt.gca()
            ax.cla()
            ax.set_title( "a_vector and a_base" )
            ax.plot( self.a_vector )
            ax.plot( self.a_base, color='red' )
            plt.tight_layout()
            plt.show()

        if flow_change_exist:
            self.flow_changes = absorbance.get_real_flow_changes()

            if helper_info is None:
                fc  = self.flow_changes
            else:
                fc  = helper_info[2]
            # flatten outside of flow_changes
            if ZEROS_TO_OUTSIDE_OF_FC:
                if fc[0] is not None:
                    ac_vector[0:fc[0]]  = 0
                if fc[1] is not None:
                    ac_vector[fc[1]+1:] = 0
            else:
                if fc[0] is not None:
                    ac_vector[0:fc[0]]  = ac_vector[ fc[0] ]
                if fc[1] is not None:
                    ac_vector[fc[1]+1:] = ac_vector[ fc[1] ]
        else:
            # this is a temp fix.
            # better be included in the above case.
             self.flow_changes = [None, None]

        max_y = None if self.orig_info is None else self.orig_info[0]
        orig_top_x = None if self.top_x_pair is None else self.top_x_pair[0]
        a_curve_temp = ElutionCurve( ac_vector, max_y=max_y, orig_top_x=orig_top_x )

        if debug:
            from CanvasDialog   import CanvasDialog
            from molass_legacy.SerialAnalyzer.ElutionCurve   import proof_plot
            def plot_func( fig, parent ):
                proof_plot( a_curve_temp, parent, fig )
            dialog = CanvasDialog( "Debug: a_curve", adjust_geometry=True )
            dialog.show( plot_func, figsize=(16, 8), parent_arg=True )

        return a_curve_temp

    def make_x_curve( self, opt_params ):
        xray_baseline_opt = opt_params[ 'xray_baseline_opt' ]
        if xray_baseline_opt == XRAY_BASE_NO:
            self.x_base = np.zeros( len(self.x_vector) )
        elif xray_baseline_opt == XRAY_BASE_CONST:
            self.x_base = self.compute_xray_baseline(opt_params)
        else:
            assert False

        xc_vector       = self.x_vector - self.x_base
        max_y = None if self.orig_info is None else self.orig_info[1]
        orig_top_x = None if self.top_x_pair is None else self.top_x_pair[1]
        x_curve_temp    = ElutionCurve( xc_vector, max_y=max_y, orig_top_x=orig_top_x )
        return x_curve_temp

    def ensure_peak_consistency( self, check_raw_info=False ):
        # for peak_info
        try:
            allowance = int( len(self.x_curve.x) * ALLOWANCE_PERCENT/100 )
            remove_info = self.check_peak_concictency( self.a_curve.peak_info, self.x_curve.peak_info, allowance )
            if len( remove_info[0] ) > 0:
                self.a_curve.remove_peaks( remove_info[0] )
                self.logger.warning( 'removed peaks ' + str(remove_info[0]) + ' from a_curve' )

            if len( remove_info[1] ) > 0:
                self.x_curve.remove_peaks( remove_info[1] )
                self.logger.warning( 'removed peaks ' + str(remove_info[1]) + ' from x_curve' )
        except:
            etb = ExceptionTracebacker()
            self.logger.error( 'failed in removing peaks:\n' + str(etb) )
            # this exception should cause the following exception 

        a_num_peaks = len( self.a_curve.peak_info )
        b_num_peaks = len( self.x_curve.peak_info )
        if a_num_peaks != b_num_peaks:
            raise RuntimeError( 'Numbers of peaks are inconsistent: %d != %d' % ( a_num_peaks, b_num_peaks ),
                                [ 1, a_num_peaks, b_num_peaks ] )

        a_num_boundaries = len( self.a_curve.boundaries )
        b_num_boundaries = len( self.x_curve.boundaries )
        if a_num_boundaries != b_num_boundaries:
            raise RuntimeError( 'Numbers of boundaries are inconsistent: %d != %d' % ( a_num_boundaries, b_num_boundaries ),
                                [ 1, a_num_boundaries, b_num_boundaries ] )

        if check_raw_info:

            # for peak_info_raw with smaller allowance to cope with such cases os 20161006/OA01
            a_peak_info_raw = self.a_curve.raw_info[PEAK_INFO_RAW]
            x_peak_info_raw = self.x_curve.raw_info[PEAK_INFO_RAW]

            try:
                allowance_raw = allowance/4     # allowance/2 is not acceptable for 20160628
                remove_info = self.check_peak_concictency( a_peak_info_raw, x_peak_info_raw, allowance_raw )
                if len( remove_info[0] ) > 0:
                    self.a_curve.remove_raw_peaks( remove_info[0] )
                    self.logger.warning( 'removed raw peaks ' + str(remove_info[0]) + ' from a_curve' )

                if len( remove_info[1] ) > 0:
                    self.x_curve.remove_raw_peaks( remove_info[1] )
                    self.logger.warning( 'removed raw peaks ' + str(remove_info[1]) + ' from x_curve' )
            except:
                etb = ExceptionTracebacker()
                self.logger.error( 'failed in removing raw peaks:\n' + str(etb) )
                # this exception should cause the following exception 

            a_peak_info_raw = self.a_curve.raw_info[PEAK_INFO_RAW]
            x_peak_info_raw = self.x_curve.raw_info[PEAK_INFO_RAW]
            if False:
                print( 'a_peak_info_raw=', a_peak_info_raw )
                print( 'x_peak_info_raw=', x_peak_info_raw )

            a_num_peaks_raw = len( a_peak_info_raw )
            b_num_peaks_raw = len( x_peak_info_raw )
            if a_num_peaks_raw != b_num_peaks_raw:
                raise RuntimeError( 'Numbers of raw peaks are inconsistent: %d != %d' % ( a_num_peaks_raw, b_num_peaks_raw ),
                                    [ 2, a_num_peaks_raw, b_num_peaks_raw ] )

    def check_peak_concictency( self, a_peak_info, x_peak_info, allowance ):
        if False:
            if len( a_peak_info ) == len( x_peak_info ):
                # better check?
                return [], []

        indeces, params = get_probably_corresponding_index( self.a_curve, self.x_curve )
        if params is None:
            return [], []

        print( 'indeces=', indeces )
        a_index, x_index = indeces

        def get_peaks_to_remove( index1, index2 ):
            # print( 'index1=', index1, 'index2=', index2 )
            to_remove = []
            for i in index1:
                if i not in index2:
                    to_remove.append( i )
            return to_remove

        a_peaks = np.array( [ info[1] for info in a_peak_info ] )
        x_peaks = np.array( [ info[1] for info in x_peak_info ] )

        e_remove = get_peaks_to_remove( list(range(len(a_peaks))), a_index )
        x_remove = get_peaks_to_remove( list(range(len(x_peaks))), x_index )

        print( 'e_remove, x_remove=', e_remove, x_remove )

        return e_remove, x_remove


    def compute_xray_baseline(self, opt_params):
        baseline_type = opt_params['xray_baseline_type']
        if baseline_type == 0:
            return np.zeros(len(self.x_vector))

        y   = copy.deepcopy( self.x_vector )
        with_bpa = opt_params['xray_baseline_with_bpa']
        if baseline_type > 0:
            # since we don't have created the self.x_curve yet, we will use self.sd_xray_curve instead
            # giving curve=temp_x_curve to improve failed LPM cases such as 20181127
            # TODO: unify creation of ScatteringBaseline objects
            ret_y = compute_baseline_using_LPM_impl( baseline_type, self.num_iterations, self.x_x, y, curve=self.sd_xray_curve,
                                                        logger=self.logger, suppress_log=False )

        if with_bpa:
            from molass_legacy.Baseline.LambertBeer import get_standard_baseline
            if self.baseline_corrected_copy is None:
                sd_copy = self.serial_data.get_copy()
                mapped_info = self.prepare_env_for_plain_LPM()
                sd_copy.apply_baseline_correction(mapped_info, basic_lpm=True)
                self.baseline_corrected_copy = sd_copy
                if self.sd_update_callback is not None:
                    self.sd_update_callback(sd_copy)
            sd_ = self.baseline_corrected_copy

            base_y = get_standard_baseline(sd_, None, logger=self.logger, debug=False)
            if False:
                x = self.x_x
                plt.push()
                fig, ax = plt.subplots()
                ax.plot(x, y, label='data')
                ax.plot(x, ret_y, label='corrected')
                ax.plot(x, ret_y + base_y, ':', label='with BPA')
                ax.legend()
                fig.tight_layout()
                plt.show()
                plt.pop()
            ret_y += base_y

        return ret_y

    def prepare_env_for_plain_LPM(self):
        self.x_curve = self.serial_data.xray_curve
        if self.x_curve_update_callback is not None:
            self.x_curve_update_callback(self.x_curve)
        self.x_base = np.zeros( len(self.x_vector) )
        self.x_base_adjustment = 0
        """
        the above three will be referenced in the following correction
        and updated after this method completes.
        i.e., correction is done without any adjustment
        """
        affine_info = self.get_affine_info()
        opt_params = {  'xray_baseline_opt':1,
                        'xray_baseline_type':1,
                        'xray_baseline_adjust':0,
                        'xray_baseline_with_bpa':1}

        # return Struct(opt_params=opt_params, affine_info=affine_info, x_curve=self.x_curve)
        return Struct(opt_params=opt_params, affine_info=affine_info)

    def compute_initial_params(self, debug=False):
        A, B = self.peak_mapper.best_info[1]    # best_params
        self.A_init = A
        self.B_init = B
        self.logger.info( 'approximate mapping params: A=%g, B=%g' % (A, B) )
        self.a_peak_ys  = [ self.a_curve.spline( info[1] ) for info in self.a_curve.peak_info ]
        self.x_peak_ys  = [ self.x_curve.spline( info[1] ) for info in self.x_curve.peak_info ]
        self.map_params = A, B
        self.init_scales = [yx/ya for ya, yx in zip(self.a_peak_ys, self.x_peak_ys)]
        if debug:
            self.logger.info("init_scales=%s", str(self.init_scales))

        if debug:
            from molass_legacy.Elution.CurveUtils import simple_plot

            A_ = 1/A
            B_ = -B/A

            with plt.Dp():
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,7))
                ax1, ax2 = axes

                fig.suptitle("compute_initial_params debug")
                simple_plot(ax1, self.a_curve, "UV", color='blue', legend=False)
                ymin, ymax = ax1.get_ylim()
                ax1.set_ylim(ymin, ymax)
                simple_plot(ax2, self.x_curve, "Xray", color='orange', legend=False)
                mapped_tops = []
                for top in self.a_curve.peak_top_x:
                    mapped_tops.append(A_*top + B_)

                ax2.plot(mapped_tops, self.x_curve.spline(mapped_tops), 'o', color='pink', label='mapped_tops')

                ax1.legend()
                ax2.legend()
                fig.tight_layout()
                fig.subplots_adjust(top=0.9)
                plt.show()

    def check_mapping_normality( self, A, B, std_diff=None ):
        """
        note that this method is usually called twice, before and after optimazation
        """
        x_ends  = self.x_x[ [0, -1] ]
        a_ends  = self.a_x[ [0, -1] ]
        mapped_x_ends   = A * x_ends + B
        mapped_x_ratio  = ( mapped_x_ends[1] - mapped_x_ends[0] ) / ( self.a_x[-1] - self.a_x[0] )
        mapped_a_ends   = ( a_ends - B ) / A
        mapped_a_ratio  = ( mapped_a_ends[1] - mapped_a_ends[0] ) / ( self.x_x[-1] - self.x_x[0] )
        # print( '---- mapped_x_ratio=', mapped_x_ratio, 'mapped_a_ratio=', mapped_a_ratio )
        if abs( mapped_x_ratio - 1 ) > MAPPED_RATIO_ALLOW or abs( mapped_a_ratio - 1 ) > MAPPED_RATIO_ALLOW:
            if True:
                self.logger.warning('abnormal mapping suspected with mapped_x_ratio=%.3g and mapped_a_ratio=%.3g' % (mapped_x_ratio, mapped_a_ratio))
            else:
                return RuntimeError( "Approximate mapping is not normal: mapped_x_ratio=%.3g, mapped_a_ratio=%.3g"
                                    % ( mapped_x_ratio, mapped_a_ratio ),
                                    [ 3, mapped_x_ratio, mapped_a_ratio, mapped_x_ends, mapped_a_ends ] )
        return None

    def compute_approximate_std_diff( self, A, B ):
        x_peak  = self.x_curve.peak_info[0][1]
        x_pk_y  = self.x_curve.spline(x_peak)
        x_y     = self.x_curve.y
        size    = len(x_y)

        a_peak  = self.a_curve.peak_info[0][1]
        a_pk_y  = self.a_curve.spline(a_peak)
        a_x     = A * self.x_x + B
        a_y     = self.a_curve.spline( a_x )
        a_y_scaled = a_y * x_pk_y / a_pk_y

        if False:
            from CanvasDialog   import CanvasDialog
            def plot_func( fig, parent ):
                ax  = fig.add_subplot( 111 )
                ax.plot( a_y_scaled )
                ax.plot( x_y, color='orange' )
                fig.tight_layout()
            dialog = CanvasDialog( "Debug: a_y_scaled", adjust_geometry=True )
            dialog.show( plot_func, figsize=(16, 8), parent_arg=True )

        std_chi_square  = np.sum( ( x_y - a_y_scaled )**2 ) / size
        std_diff        = np.sqrt( std_chi_square ) / x_pk_y * STD_DIFF_SCALE
        # print( 'approximate_std_diff=', std_diff )
        return std_diff, a_y_scaled, x_y

    def optimize_mapping(self, devel=False):
        if devel:
            from importlib import reload
            import Mapping.MappingOptimizer
            reload(Mapping.MappingOptimizer)
        from .MappingOptimizer import optimize_mapping_impl

        optimize_mapping_impl(self)

    def determine_mapping_ranges( self ):
        self.x_ranges = self.x_curve.get_ranges_by_ratio( MAPPING_RANGE_RATIO, debug=False )
        # self.mapping_ranges = self.x_curve.peak_info      # there exists a bug for 20181203
        self.mapping_ranges = self.x_ranges             # temp fix for the above bug; must be verified for other cases

        if REFINE_MAPPING_RANGES:
            try:
                self.refine_mapping_ranges()
            except:
                etb = ExceptionTracebacker()
                self.logger.warning("refine_mapping_ranges faild: %s", str(etb))

    def refine_mapping_ranges( self ):
        # not yet usable for such cases 20180206
        # print('before refine', self.mapping_ranges)

        x_ranges = self.x_ranges
        x_mapping_ranges = self.mapping_ranges

        a_mapping_ranges = self.a_curve.peak_info
        x_delta = int( len( self.x_curve.x ) * MAPPING_RANGE_DELTA + 0.5 )

        for i, a_boundary in enumerate( self.a_curve.boundaries ):
        # for i, a_boundary in enumerate( a_boundaries ):
            x_boundary  = self.x_curve.boundaries[i]
            xm_boundary = self.A_init * x_boundary + self.B_init
            x_lower     = x_mapping_ranges[i][2]
            x_upper     = x_mapping_ranges[i+1][0]
            a_lower     = a_mapping_ranges[i][2]
            a_upper     = a_mapping_ranges[i+1][0]
            am_boundary = int( ( a_boundary - self.B_init )/self.A_init + 0.5 )
            # print( 'refine_mapping_ranges (1)', [i], a_lower, a_boundary, xm_boundary, a_upper )

            if False:
                fig = plt.figure( figsize=(12,6) )
                ax1 = fig.add_subplot( 121 )
                ax2 = fig.add_subplot( 122 )

                b_colors = [ 'green', 'yellow' ]

                ax1.plot( self.a_curve.y )
                ymin1, ymax1 = ax1.get_ylim()
                ax1.set_ylim( ymin1, ymax1 )
                for x in [a_lower, a_upper]:
                    ax1.plot( [ x, x ], [ ymin1, ymax1 ], ':', color='black', alpha=0.2 )
                for k, x in enumerate([a_boundary, xm_boundary]):
                    ax1.plot( x, self.a_curve.spline(x), 'o', color=b_colors[k] )

                ax2.plot( self.x_curve.y, color='orange' )
                ymin2, ymax2 = ax2.get_ylim()
                ax2.set_ylim( ymin2, ymax2 )
                for x in [x_lower, x_upper]:
                    ax2.plot( [ x, x ], [ ymin2, ymax2 ], ':', color='black', alpha=0.2 )
                for k, x in enumerate([am_boundary, x_boundary]):
                    ax2.plot( x, self.x_curve.spline(x), 'o', color=b_colors[k] )

                fig.tight_layout()
                plt.show()

            top_x = x_mapping_ranges[i][1]
            if am_boundary < x_lower:
                am_boundary_    = am_boundary - x_delta
                upper_candidate = min( x_ranges[i][2], am_boundary_ )
                if upper_candidate > top_x:
                    x_mapping_ranges[i][2] = am_boundary_
                    x_ranges[i][2] = upper_candidate
                else:
                    self.logger.warning('failed to refine right boundary of %s to %d < %d' % ( str(x_mapping_ranges[i]), upper_candidate, top_x) )

            if am_boundary > x_upper:
                am_boundary_    = am_boundary + x_delta
                lower_candidate = max( x_ranges[i+1][0], am_boundary_ )
                if lower_candidate < top_x:
                    x_mapping_ranges[i+1][0] = am_boundary_
                    x_ranges[i+1][0] = lower_candidate
                    self.logger.warning('failed to refine ')
                else:
                    self.logger.warning('failed to refine left boundary of %s to %d > %d' % ( str(x_mapping_ranges[i]), lower_candidate, top_x) )

        self.x_ranges   = x_ranges
        self.mapping_ranges = x_mapping_ranges
        # print('after refine', self.mapping_ranges)

    def make_whole_mapped_vector(self, A, B, scales, make_inv_mapped_boundaries=False, with_original_scale=False ):
        boundaries = self.x_curve.boundaries
        # print( 'mapping_ranges=', self.mapping_ranges )
        # print( 'boundaries=', boundaries )

        debug = False
        detail_debug = False

        mapped_vector_list  = []
        if detail_debug:
            debug_list = []
        if with_original_scale:
            orig_vector_list = []

        for i, range_ in enumerate( self.mapping_ranges ):
            peak    = self.x_curve.peak_info[i][1]
            p_start = max(0, range_[0])     # temp fix to avoid range_[0] < 0 for raw Kosugi8, which should be investigated
            p_stop  = min( len(self.x_curve.y), range_[2] + 1 )
            xray_x  = np.arange( p_start, p_stop )
            S = scales[i]

            if i == 0:
                if debug: print( 'mapped vector (1)', ( 0, p_start ) )
                x_  = np.arange( 0, p_start )
                m_vec, orig_vec = self.make_mapped_vector( x_, A, B, S )
                mapped_vector_list.append( m_vec )
                if detail_debug:
                    debug_list.append([0, (0, p_start), p_start - 0, len(m_vec)])
                if with_original_scale:
                    orig_vector_list.append( orig_vec )

            else:
                if i <= len(boundaries):
                    boundary = int(boundaries[i-1])
                    if debug: print( 'mapped vector (2)', ( prev_stop, p_start ) )
                    m_vec, orig_vec   = self.make_mapped_valley_vector( i, prev_stop, boundary, p_start, prev_A, prev_B, prev_S, A, B, S, make_inv_mapped_boundaries,
                                            with_original_scale=with_original_scale)
                    mapped_vector_list.append( m_vec )
                    if detail_debug:
                        debug_list.append([2, ( prev_stop, p_start ), p_start - prev_stop, len(m_vec)])

                    if with_original_scale:
                        orig_vector_list.append( orig_vec )

            if debug: print( 'mapped vector (3)', ( p_start, p_stop ) )
            prev_A, prev_B, prev_S, prev_stop = A, B, S, p_stop
            m_vec, orig_vec = self.make_mapped_vector( xray_x, A, B, S )
            mapped_vector_list.append( m_vec )
            if detail_debug:
                debug_list.append([3, ( p_start, p_stop), p_stop - p_start, len(m_vec)])

            if with_original_scale:
                orig_vector_list.append( orig_vec )

            if i == len( self.mapping_ranges ) - 1:
                if debug: print( 'mapped vector (4)', ( p_stop, len(self.x_vector) ) )
                x_  = np.arange( p_stop, len(self.x_vector) )
                m_vec, orig_vec = self.make_mapped_vector( x_, A, B, S )
                mapped_vector_list.append( m_vec )
                if detail_debug:
                    debug_list.append([4, (p_stop, len(self.x_vector)), len(self.x_vector) - p_stop, len(m_vec)])
                if with_original_scale:
                    orig_vector_list.append( orig_vec )

            if debug:
                temp_mvec = np.hstack( mapped_vector_list )
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.set_title( "%d-th plot with range_=%s" % (i, str(range_)) )
                    ax.plot( self.x_curve.x, self.x_curve.y )
                    ax.plot( temp_mvec )
                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim(ymin, ymax)
                    x_  = len(temp_mvec)
                    ax.plot( [x_, x_], [ymin, ymax], ':', color='red' )
                    plt.tight_layout()
                    plt.show()

        if debug:
            length = 0
            for k, vec in enumerate(mapped_vector_list):
                length += len(vec)
                print([k], len(vec), length)
            print(len(self.x_x))

        mapped_vector  = np.hstack( mapped_vector_list )
        if detail_debug and len(mapped_vector) != len(self.x_curve.x):
            print('len(mapped_vector)=', len(mapped_vector))
            print('len(self.x_curve.x)=', len(self.x_curve.x))
            print('self.mapping_ranges=', self.mapping_ranges)
            print('boundaries=', boundaries)

            x = self.x_curve.x
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.plot(x, self.x_curve.y)
                ax.plot(mapped_vector)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
                mv_start = 0
                for k, mv in enumerate(mapped_vector_list):
                    info = debug_list[k]
                    print( [k], mv_start, len(mv), info )
                    if len(mv) > 0:
                        f = mv_start
                        t = mv_start + len(mv) - 1
                        for p in [f, t]:
                            ax.plot([p, p], [ymin, ymax], ':', color='yellow')
                        mv_start += len(mv)

                plt.tight_layout()
                plt.show()

        if with_original_scale:
            orig_vector = np.hstack( orig_vector_list )
        else:
            orig_vector = None

        assert len(mapped_vector) == len(self.x_x)
        return mapped_vector, orig_vector

    def make_mapped_vector( self, xray_x, A, B, S ):
        j_  = A*xray_x + B
        if False:
            ax = plt.gca()
            ax.set_title('make_mapped_vector debug')
            try:
                ax.plot(self.a_curve_spline_adjusted( j_ ), label='a_curve_spline_adjusted')
            except:
                pass
            ax.plot(self.a_curve.spline( j_ ), label='a_curve.spline')
            ax.legend()
            plt.tight_layout()
            plt.show()

        if self.in_uv_adjustment_mode:
            tgt_y   = self.a_curve_spline_adjusted( j_ )
        else:
            tgt_y   = self.a_curve.spline( j_ )
        m_vec   = tgt_y * S
        return m_vec, tgt_y

    def make_mapped_valley_vector( self, i, prev_stop, x_boundary, p_start, prev_A, prev_B, prev_S, A, B, S, make_inv_mapped_boundaries,
                                     with_original_scale=False):
        # print( 'make_mapped_valley_vector: prev_stop, x_boundary, p_start=', prev_stop, x_boundary, p_start )

        if x_boundary < prev_stop:
            # temporary fix for 20181203 when it is not an analysis copy
            self.logger.warning("boundary has been modified from %d to %d in make_mapped_valley_vector as a temporary fix." % (x_boundary, prev_stop))
            x_boundary = prev_stop
        if x_boundary > p_start:
            # temporary fix for 20171203
            self.logger.warning("boundary has been modified from %d to %d in make_mapped_valley_vector as a temporary fix." % (x_boundary, p_start))
            x_boundary = p_start

        xray_x1 = np.arange( prev_stop, x_boundary )
        xray_x2 = np.arange( x_boundary, p_start )

        if False:
            print( 'prev_stop, x_boundary, p_start=', prev_stop, x_boundary, p_start )
            print( 'xray_x1=', len(xray_x1), xray_x1 )
            print( 'xray_x2=', len(xray_x2), xray_x2 )
            x_curve = self.x_curve
            x   = x_curve.x
            y   = x_curve.y
            spline  = x_curve.spline
            fig = plt.figure()
            ax = fig.gca()
            ax.set_title("make_mapped_valley_vector debug")
            ax.plot( x, y )
            for x_ in [prev_stop, x_boundary, p_start]:
                ax.plot( x_, spline(x_), 'o' )
            plt.show()

        denom_width = p_start - prev_stop
        if denom_width > 0:
            w0 = ( x_boundary - 1 - prev_stop ) / denom_width
        else:
            # as occurred in 20191006_proteins5
            w0 = 0

        if USE_SAME_RATIO_BOUNDARY:
            # determine a_boundary so that a_boundary keeps the same ratio as x_boundary
            # in relation to ( prev_stop, p_start  ) and ( v_lower, v_upper ) respectively
            v_lower = prev_A * prev_stop + prev_B
            v_upper = A      * p_start   + B
            a_boundary  = v_lower * ( 1 - w0 ) + v_upper * w0

            # determine A_ so that it is consistent with the interval length ratio
            A_  = ( v_upper - v_lower ) / ( p_start - prev_stop )

            # determine B_ so that a_boundary be mapped to x_boundary
            B_  = a_boundary - A_ * x_boundary
        else:
            A_  = prev_A * ( 1 - w0 ) + A * w0
            B_  = prev_B * ( 1 - w0 ) + B * w0
            a_boundary  = A_ * x_boundary + B_

        S_      = prev_S * ( 1 - w0 ) + S * w0
        S_min   = min( prev_S, S )
        S_max   = max( prev_S, S )

        if make_inv_mapped_boundaries:
            self.inv_mapped_boundaries.append( a_boundary )

        if len(xray_x1) > 1:
            w1  = np.arange( len(xray_x1) ) / ( len(xray_x1) - 1)

            last_j  = prev_A*xray_x1 + prev_B
            this_j  = A_*xray_x1 + B_
            j_      = last_j * ( 1 - w1 ) + this_j * w1

            if self.in_uv_adjustment_mode:
                tgt_y1  = self.a_curve_spline_adjusted( j_ )
            else:
                tgt_y1  = self.a_curve.spline( j_ )

            if USE_SIMPLE_SCALE_FOR_VALLEY:
                # this scaling seems better as in 20170209/OA_Ald_Fer
                boundary_scale = S_
            else:
                a_by    = tgt_y1[-1]
                xj      = xray_x1[-1]
                if self.in_xray_adjustment_mode:
                    x_by    = self.x_curve_y_adjusted[ xj ]
                else:
                    x_by    = self.x_curve.y[ xj ]
                if abs( a_by ) > SMALL_VALUE_LIMIT:
                    b_scale = x_by / a_by
                    # avoid extreme values
                    abs_b_scale = max( S_min/2, min( S_max*2, abs( b_scale )  ) )
                    boundary_scale  = abs_b_scale if b_scale >= 0 else -abs_b_scale
                else:
                    boundary_scale  = S_

            # print( 'scales=', prev_S, boundary_scale, S )
            scale1  = prev_S * ( 1 - w1 ) + boundary_scale * w1
            m_vec1  = tgt_y1 * scale1
        elif len(xray_x1) == 1:
            # 20181204 
            boundary_scale = S_
            j_ = A_*xray_x1 + B_
            tgt_y1  = self.a_curve.spline( j_ )
            m_vec1  = tgt_y1 * S_
        else:
            boundary_scale = S_
            tgt_y1 = np.array([])
            m_vec1 = np.array([])

        if len(xray_x2) > 1:
            w2  = np.arange( len(xray_x2) ) / ( len(xray_x2) - 1)

            last_j  = A_*xray_x2 + B_
            this_j  = A *xray_x2 + B
            j_      = last_j * ( 1 - w2 ) + this_j * w2

            if self.in_uv_adjustment_mode:
                tgt_y2   = self.a_curve_spline_adjusted( j_ )
            else:
                tgt_y2   = self.a_curve.spline( j_ )

            scale2  = boundary_scale * ( 1 - w2 ) + S * w2
            m_vec2  = tgt_y2 * scale2
        elif len(xray_x2) == 1:
            j_ = A_*xray_x2 + B_
            tgt_y2  = self.a_curve.spline( j_ )
            m_vec2  = tgt_y2 * S
        else:
            tgt_y2 = np.array([])
            m_vec2 = np.array([])

        m_vec = np.hstack( [ m_vec1, m_vec2 ] )
        if with_original_scale:
            orig_vec = np.hstack( [ tgt_y1, tgt_y2 ] )
        else:
            orig_vec = None

        return m_vec, orig_vec

    def adjust_baselines( self, opt_params ):
        uv_baseline_adjust      = opt_params[ 'uv_baseline_adjust' ]
        xray_baseline_adjust    = opt_params[ 'xray_baseline_adjust' ]
        self.dev_allow_ratio    = opt_params[ 'dev_allow_ratio' ]
        self.a_base_adjustment  = 0
        self.x_base_adjustment  = 0
        need_re_optimization    = False

        if USE_AREA_SCALING:
            if uv_baseline_adjust == 1 and xray_baseline_adjust == 1:
                self.compute_deviation_scales()

        # iterate to get better precision for revised self.mapped_vector
        for k in range( AJUSTMENT_ITERATION ):
            adjust_both = True
            if uv_baseline_adjust == 1 and xray_baseline_adjust == 0 or ( 1 - self.dev_allow_ratio ) < VERY_SMALL_RATIO:
                adjust_both = False
                self.adjust_uv_baseline_only()
                need_re_optimization    = True

            if uv_baseline_adjust == 0 and xray_baseline_adjust == 1 or self.dev_allow_ratio < VERY_SMALL_RATIO:
                adjust_both = False
                self.adjust_xray_baseline_only()
                need_re_optimization    = True

            if uv_baseline_adjust == 1 and xray_baseline_adjust == 1 and adjust_both:
                self.adjust_both_baselines( opt_params )
                need_re_optimization    = True

            if need_re_optimization:
                self.optimize_mapping()

    def compute_deviation_scales( self ):
        # compute UV-absorbance area
        A, B, C = self.absorbance.get_baseplane_params()
        end_x   = np.array( [0, len(self.a_vector)-1] )
        floor_end_y = A * self.absorbance.std_wvlen + B * end_x + C
        sbl = ScatteringBaseline( -self.a_vector )
        D, E  = sbl.solve()
        ceil_end_y  = -D * end_x - E
        self.a_area = self.compute_area( end_x, floor_end_y, ceil_end_y )
        if False:
            from CanvasDialog   import CanvasDialog
            def debug_plot( fig ):
                ax  = fig.add_subplot( 111 )
                ax.plot( self.a_vector )
                ax.plot( end_x, floor_end_y )
                ax.plot( end_x, ceil_end_y )
                fig.tight_layout()
            dialog = CanvasDialog( "Debug: a_area", adjust_geometry=True )
            dialog.show( debug_plot, figsize=(8, 8) )

        # compute Xray-scattering area
        # giving curve=self.x_curve to improve failed LPM cases such as 20181127
        sbl = ScatteringBaseline( self.x_vector, curve=self.x_curve )
        A, B  = sbl.solve()
        end_x   = np.array( [0, len(self.x_vector)-1] )
        floor_end_y = A * end_x + B
        sbl = ScatteringBaseline( -self.x_vector )
        D, E  = sbl.solve()
        ceil_end_y  = -D * end_x - E
        self.x_area = self.compute_area( end_x, floor_end_y, ceil_end_y )
        if False:
            from CanvasDialog   import CanvasDialog
            def debug_plot( fig ):
                ax  = fig.add_subplot( 111 )
                ax.plot( self.x_vector )
                ax.plot( end_x, floor_end_y )
                ax.plot( end_x, ceil_end_y )
                fig.tight_layout()
            dialog = CanvasDialog( "Debug: x_area", adjust_geometry=True )
            dialog.show( debug_plot, figsize=(8, 8) )

    def compute_area( self, end_x, floor_end_y, ceil_end_y ):
        return end_x[-1] * ( ceil_end_y[0] - floor_end_y[0] + ceil_end_y[1] - floor_end_y[1]  ) / 2

    def get_adjusted_a_spline(self):
        # a_curve_y = self.a_curve.y
        a_curve_y = self.a_curve.spline(self.a_x)
        ac_vector = a_curve_y - self.a_base_adjustment
        return UnivariateSpline( self.a_x, ac_vector, s=0, ext=3 )

    def uv_baseline_adjuster( self, params ):
        D, E = params
        self.a_base_adjustment = D * self.a_x + E
        self.a_curve_spline_adjusted = self.get_adjusted_a_spline()
        adj_scales = self.adjust_uv_opt_results( D, E )
        mapped_vector, _ = self.make_whole_mapped_vector(*self.map_params, adj_scales)
        diff_y      = self.x_curve.y - mapped_vector

        if False:
            from CanvasDialog   import CanvasDialog
            from molass_legacy.SerialAnalyzer.ElutionCurve   import proof_plot
            print( 'D, E=', D.value, E.value, 'diff2_sum=', np.sum( diff_y**2 ) )
            def plot_func( fig ):
                ax1 = fig.add_subplot( 131 )
                ax2 = fig.add_subplot( 132 )
                ax3 = fig.add_subplot( 133 )
                ax1.plot( self.a_curve.y )
                ax1.plot( self.a_base_adjustment, color='yellow')
                # ax2.plot( ac_vector )
                ax2.plot( self.a_curve_spline_adjusted( self.a_x ), ':' )
                ax3.plot( self.x_curve.y, color='orange' )
                ax3.plot( mapped_vector )
                fig.tight_layout()
            dialog = CanvasDialog( "Debug: a_curve", adjust_geometry=True )
            dialog.show( plot_func, figsize=(24, 6) )

        return np.sum(diff_y**2)

    def adjust_uv_opt_results( self, D, E ):
        adj_results = []
        A, B = self.map_params
        for i, info in enumerate(self.x_curve.peak_info):
            x_peak  = info[1]
            x_py    = self.x_curve.spline( x_peak )
            a_peak  = A * x_peak + B
            a_py    = self.a_curve_spline_adjusted( a_peak )
            S_  = x_py / a_py
            adj_results.append(S_ )
        return adj_results

    def adjust_uv_baseline_only( self ):
        self.in_uv_adjustment_mode  = True
        self.in_xray_adjustment_mode = False

        E_init = np.percentile( self.a_curve.y, 5 )

        init_params = (0, E_init)
        bounds = ((-1e-4, 1e-4), (-0.1, 0.1))

        # result  = minimize( self.uv_baseline_adjuster, params, args=() )
        result  = optimize.minimize(self.uv_baseline_adjuster, init_params, bounds=bounds)

        D, E = result.x

        self.a_base_adjustment  =  D * self.a_x + E
        self.a_curve_spline_adjusted = self.get_adjusted_a_spline()

    def xray_baseline_adjuster( self, params ):
        F, G = params

        x_base_adjustment   = F * self.x_x + G

        adj_scales = self.adjust_xray_opt_results( F, G )
        mapped_vector, _ = self.make_whole_mapped_vector(*self.map_params, adj_scales)
        xc_vector = self.x_curve.y - x_base_adjustment
        diff_y  = xc_vector - mapped_vector
        return np.sum(diff_y**2)

    def adjust_xray_opt_results( self, F, G ):
        adj_results = []

        A, B = self.map_params
        for i, info in enumerate(self.x_curve.peak_info):
            x_peak  = info[1]
            x_py    = self.x_curve.spline( x_peak ) - ( F*x_peak + G )
            a_peak  = A * x_peak + B
            a_py    = self.a_curve.spline( a_peak )
            S_  = x_py / a_py
            adj_results.append(S_)
        return adj_results

    def adjust_xray_baseline_only( self, G_init=0 ):
        self.in_uv_adjustment_mode  = False
        self.in_xray_adjustment_mode = True

        init_params = (0, G_init)
        bounds = ((-1e-4, 1e-4), (G_init-0.1, G_init+0.1))

        # result  = minimize( self.xray_baseline_adjuster, params, args=() )
        result = optimize.minimize(self.xray_baseline_adjuster, init_params, bounds=bounds)

        F, G = result.x
        self.x_base_adjustment  = F * self.x_x + G
        self.x_curve_y_adjusted = self.x_curve.y - self.x_base_adjustment
        self.x_peak_ys_adjusted = [ self.x_curve.spline( info[1] ) - ( F * info[1] + G ) for info in self.x_curve.peak_info ]

    def both_baseline_adjuster( self, params ):
        D, E, F, G = params

        xray_y  = self.x_curve.y - ( F * self.x_x + G )

        a_base_adjustment   = D * self.a_x + E
        x_base_adjustment   = F * self.x_x + G

        self.a_base_adjustment = a_base_adjustment
        self.a_curve_spline_adjusted = self.get_adjusted_a_spline()
        self.x_curve_y_adjusted = self.x_curve.y - x_base_adjustment
        adj_scales = self.adjust_both_opt_results( D, E, F, G )
        mapped_vector, _    = self.make_whole_mapped_vector(*self.map_params, adj_scales)
        mapped_adjustment   = self.make_mapped_adjustment(*self.map_param, D, E )
        penalty_vector  = self.compute_penalty_vector( a_base_adjustment, mapped_adjustment, x_base_adjustment )

        if self.adjust_both_debug:
            from CanvasDialog   import CanvasDialog
            def plot_func( fig ):
                ax = fig.add_subplot( 111 )
                ax.plot( mapped_adjustment )
                ax.plot( x_base_adjustment )
                ax.plot( penalty_vector )
                fig.tight_layout()
            dialog = CanvasDialog( "Debug: mapped_adjustment", adjust_geometry=True )
            dialog.show( plot_func, figsize=(8, 6) )
            if not dialog.applied:
                self.adjust_both_debug = False

        diff_y  = np.abs( xray_y - mapped_vector ) + penalty_vector
        return np.sum(diff_y**2)

    def adjust_both_opt_results( self, D, E, F, G ):
        adj_results = []
        A, B = self.map_params
        for i, info in enumerate(self.x_curve.peak_info):
            x_peak  = info[1]
            x_py    = self.x_curve.spline( x_peak ) - ( F*x_peak + G )
            a_peak  = A * x_peak + B
            a_py    = self.a_curve_spline_adjusted( a_peak )
            S_  = x_py / a_py
            adj_results.append(S_)
        return adj_results

    def compute_penalty_vector( self, a_adj, m_adj, x_adj ):
        """
        the purpose of introducing a penalty vector is to prevent the optimizer
        from getting lost into useless cases where both adjustments grow large
        in the same direction.
        """
        dir_penalty  = np.zeros( len(m_adj) )
        both_positive   = np.logical_and( m_adj > 0, x_adj > 0 )
        both_negative   = np.logical_and( m_adj < 0, x_adj < 0 )
        dir_penalty[both_positive]   = np.min( [ m_adj[both_positive], x_adj[both_positive] ], axis=0 )
        dir_penalty[both_negative]   = -np.max( [ m_adj[both_negative], x_adj[both_negative] ], axis=0 )

        if USE_AREA_SCALING:
            uv_area_ratio   = np.sum( np.abs(a_adj) ) / self.a_area
            xray_area_ratio = np.sum( np.abs(x_adj) ) / self.x_area
        else:
            uv_area_ratio   = np.sum( np.abs(m_adj) )
            xray_area_ratio = np.sum( np.abs(x_adj) )

        total_ratio     = uv_area_ratio + xray_area_ratio
        uv_weight   = 1 - self.dev_allow_ratio
        xray_weight = self.dev_allow_ratio

        if total_ratio > 0:
            weight_dev_penalty = 1 + abs( uv_area_ratio/total_ratio * uv_weight - xray_area_ratio/total_ratio * xray_weight )
        else:
            weight_dev_penalty = 1

        if USE_AREA_SCALING:
            x_area  = self.x_area
        else:
            x_area  = self.x_curve.max_y * ( len( self.x_vector ) - 1 ) / 2
            # * 1/2 is an approximate scalling to make it equivalent to USE_AREA_SCALING

        keep_positive = x_area * PENALTY_KEEP_POS_VALUE

        ret_penalty = ( dir_penalty + keep_positive ) * weight_dev_penalty

        if False:
            from matplotlib.patches import Polygon
            from molass_legacy.KekLib.OurMatplotlib      import get_default_colors
            print( 'x_area=', x_area, 'keep_positive=', keep_positive )
            colors = get_default_colors()

            fig = plt.figure(figsize=(16, 8))
            ax1 = fig.add_subplot( 121 )
            ax2 = fig.add_subplot( 122 )
            ax1.set_title( "UV-absorbance adustment" )
            ax2.set_title( "Xray and mapped adustment with the penalty vector" )
            ax1.plot( a_adj, label='UV-adjustment' )
            ax2.plot( m_adj, label='mapped adjustment' )
            ax2.plot( x_adj, label='Xray adjustment' )
            ax2.plot( dir_penalty, ':', label='same direction penality' )
            ax2.plot( ret_penalty, ':', label='penalty with deviation ratio' )
            pos = np.where( dir_penalty > 0 )[0]
            if len(pos) > 0:
                null = np.array( [], dtype=int )
                pos_x = np.hstack( [ [pos[0]-1] if pos[0] > 0 else null, pos, [pos[-1]+1] if pos[-1] < len(dir_penalty)-1 else null ] )
                pos_y = dir_penalty[pos_x]
                points = list( zip( pos_x, pos_y ) )
                polygon = Polygon( points, alpha=0.2, fc=colors[2] )
                ax2.add_patch( polygon )
            ax1.legend()
            ax2.legend()
            fig.tight_layout()
            plt.show()

        return ret_penalty

    def make_mapped_adjustment(self, A, B, D, E ):
        adjustment_array = []

        effective_num_boundaries = min(len(self.x_curve.boundaries), len(self.scale_params) - 1)
        if effective_num_boundaries < len(self.x_curve.boundaries):
            """
                this case occurred in 20190221_2 with LPM(integral)
                may better be avoided by computing self.opt_results for each peak
            """
            self.logger.warning("wrong number of opt_results: len(self.opt_results) < len(self.x_curve.boundaries) + 1")

        start = 0
        for k, S in enumerate( self.scale_params ):
            if k < effective_num_boundaries:
                stop    = self.x_curve.boundaries[k]
            else:
                stop    = len(self.x_x)
            i_  = np.arange( start, stop )
            j_  = A * i_ + B
            adjustment = ( D * j_ + E ) * S
            adjustment_array.append( adjustment )
            start   = stop

        mapped_adjustment = np.hstack( adjustment_array )
        assert len(mapped_adjustment) == len(self.x_x)
        return mapped_adjustment

    def adjust_both_baselines( self, opt_params, G_init=0 ):
        self.in_uv_adjustment_mode      = True
        self.in_xray_adjustment_mode    = True

        init_params = (0, 0, 0, G_init)
        bounds = ((-1e-4, 1e-4), (-0.1, 0.1 ), (-1e-4, 1e-4 ), (G_init-0.1, G_init+0.1))

        self.adjust_both_debug = False

        # result  = minimize( self.both_baseline_adjuster, params, args=() )
        result = optimize.minimize(self.both_baseline_adjuster, init_params, bounds=bounds)

        D, E, F, G = result.x
        self.a_base_adjustment  = D * self.a_x + E
        self.a_curve_spline_adjusted = self.get_adjusted_a_spline()
        self.x_base_adjustment  = F * self.x_x + G
        self.x_curve_y_adjusted = self.x_curve.y - self.x_base_adjustment
        self.x_peak_ys_adjusted = [ self.x_curve.spline( info[1] ) - ( F * info[1] + G ) for info in self.x_curve.peak_info ]

    def compute_curve_for_eval_diff( self ):
        pass

    def get_uniformly_scaled_vector( self ):
        if self.uniformly_scaled_vector is None:
            self.uniformly_scaled_vector = self.make_uniformly_scaled_vector()

        return self.uniformly_scaled_vector

    def get_x2a_scale(self):
        x_peak_y_adjust = 0 if np.isscalar( self.x_base_adjustment) else self.x_base_adjustment[ self.x_curve.primary_peak_i ]
        a_j = int( self.A_init * self.x_curve.primary_peak_i + self.B_init + 0.5 )
        a_peak_y_adjust = 0 if np.isscalar( self.a_base_adjustment) else self.a_base_adjustment[ a_j ]

        scale   = ( self.a_curve.spline(a_j)- a_peak_y_adjust ) / ( self.x_curve.primary_peak_y - x_peak_y_adjust )
        return scale

    def make_uniformly_scaled_vector( self, scale=None, survey_fh=None):
        if scale is None:
            scale = self.get_x2a_scale()

        if np.isscalar(self.a_base_adjustment):
            adj_spline  = lambda x: self.a_base_adjustment
        else:
            adj_spline  = UnivariateSpline( self.a_x, self.a_base_adjustment, s=0, ext=3 )

        if USE_SIMPLE_MAPPING:
            use_simple_mapping = True
        else:
            scaled_vector_array = []
            start = 0
            last_joint_y = None
            use_simple_mapping = False
            A, B = self.map_params
            for k, _ in enumerate( self.x_curve.peak_info ):
                if k < len( self.x_curve.boundaries ):
                    stop    = self.x_curve.boundaries[k]
                else:
                    stop    = len(self.x_curve.x)
                i_  = self.x_curve.x[start:stop]
                j_  = A*i_ + B
                tgt_y   = self.a_curve.spline( j_ ) - adj_spline( j_ )
                if last_joint_y is not None:
                    joint_diff_ratio = abs(tgt_y[0] - last_joint_y)/self.a_curve.max_y
                    if joint_diff_ratio > 0.1:
                        use_simple_mapping = True
                        self.logger.info('a significant gap with ratio %.3g detected in uniformly scaled curve' % joint_diff_ratio)
                        break
                last_joint_y = tgt_y[-1]
                scaled_vector_array.append( tgt_y/scale )
                start   = stop

        if use_simple_mapping:
            scaled_vector_array = self.make_uniformly_scaled_vector_in_another_way(scale, adj_spline, survey_fh=survey_fh)

        return np.hstack( scaled_vector_array )

    def make_uniformly_scaled_vector_in_another_way( self, scale, adj_spline, survey_fh=None):
        # print('make_uniformly_scaled_vector_in_another_way: scale=', scale)
        if True:
            A, B = self.map_params
        else:
            if self.A_simple is None:
                self.compute_simple_params()
            A, B = self.A_simple, self.B_simple
        if survey_fh is not None:
            from .SurveyUtils import write_info
            # this case is temporary only for change influence survey purpose
            write_info(survey_fh, self)

        j_ = A * self.x_curve.x + B
        tgt_y  = ( self.a_curve.spline( j_ ) - adj_spline( j_ ) ) / scale
        # tgt_y2 = self.a_curve_spline_adjusted( j_ ) / scale
        if False:
            fig = plt.figure(figsize=(12,6))
            fig.suptitle("make_uniformly_scaled_vector_in_another_way")
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.plot( tgt_y, label='tgt_y' )
            ax1.plot( tgt_y, label='tgt_y2' )
            ax2.plot( self.x_curve.y, color='orange' )
            fig.tight_layout()
            plt.show()
        return tgt_y

    def make_scaled_xray_curve_y(self, debug=False):
        scaled_vector_list = []

        x = self.x_curve.x
        y = self.x_curve_y_adjusted

        start = 0
        A, B = self.map_params
        for k, _ in enumerate( self.x_curve.peak_info ):
            if k < len( self.x_curve.boundaries ):
                stop    = self.x_curve.boundaries[k]
            else:
                stop    = len(x)
            S = self.scale_params[k]
            scaled_vector_list.append(y[start:stop]/S)
            start = stop

        scaled_y = np.hstack(scaled_vector_list)

        if debug:
            from molass_legacy.Elution.CurveUtils import simple_plot
            from DataUtils import get_in_folder

            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))
            ax1, ax2, ax3 = axes
            ax3t = ax3.twinx()

            fig.suptitle("Proof Plot of Mapping for " + get_in_folder(), fontsize=20)
            x_ = self.inv_mapped_boundaries
            ax1.plot(x_, self.a_curve.spline(x_), 'o', color='yellow', markersize=10, label='invserse mapped')
            simple_plot(ax1, self.a_curve)
            simple_plot(ax2, self.x_curve)
            simple_plot(ax3, self.x_curve)
            ax3t.plot(x, scaled_y, ':', label='scaled_y', color='green')

            ax3t.legend()
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)

            plt.show()

        return scaled_y

    def get_mapped_info( self ):
        affine_info = self.get_affine_info()
        return MappedInfo( opt_results=self.opt_results, a_curve=self.a_curve, x_curve=self.x_curve, x_ranges=self.x_ranges,
                            affine_info=affine_info, flow_changes=self.flow_changes,
                            opt_params=self.opt_params, x_base=self.x_base )

    def get_affine_info( self ):
        peaks   = [ info[1] for info in self.x_curve.peak_info ]
        peak_x  = np.average( peaks )
        peak_y  = np.average( self.x_curve.spline( peaks ) )
        return [ self.x_vector, self.x_base, self.x_base_adjustment, (peak_x, peak_y), peaks ]

    def get_num_peaks( self ):
        return len(self.a_curve.peak_info), len(self.x_curve.peak_info)

    def get_conc_vector( self, conc_factor ):
        conc_curve_type = get_setting('conc_curve_type')
        if conc_curve_type == 0:
            ret_uv_vector = self.make_uniformly_scaled_vector(scale=1)
            curve_source = 'Mapped UV Elution'
        else:
            ret_uv_vector = self.make_scaled_xray_curve_y()
            curve_source = 'Scaled Xray Elution'
        self.logger.info("concentration curve made from " + curve_source)
        return ret_uv_vector * conc_factor

    def set_peak_eval_ranges( self ):
        ranges_ = self.x_curve.get_ranges_by_ratio(PEAK_EVAL_RANGE_RATIO)

        self.peak_eval_ranges = []
        for range_ in ranges_:
            p = range_[1]
            for mrange in self.mapping_ranges:
                if abs( p - mrange[1] ) < 5:
                    self.peak_eval_ranges.append( range_ )
                    break
 
        self.uv_peak_eval_ranges = []
        A, B = self.map_params
        for k, range_ in enumerate(self.peak_eval_ranges):
            self.uv_peak_eval_ranges.append( [ A*x + B for x in range_ ] )

    def compute_std_diff( self ):
        if self.in_xray_adjustment_mode:
            x_curve_y   = self.x_curve_y_adjusted
        else:
            x_curve_y   = self.x_curve.y

        if USE_TOTAL_STD_DIFF:
            size    = len(self.x_curve.y)
            if USE_WEIGHTED_DIFF:
                weight  = np.ones( size )
                for lower, peak_i, upper in self.peak_eval_ranges:
                    weight[lower:upper+1]   = VALLEY_DIFF_WEIGHT
            else:
                weight = 1
            std_chi_square = np.sum( weight * ( self.mapped_vector - x_curve_y )**2 ) / size
            std_diff = np.sqrt( std_chi_square ) / x_curve_y[self.x_curve.primary_peak_i]
        else:
            std_diff = 0
            size    = 0
            for lower, peak_i, upper in self.peak_eval_ranges:
                size    = upper - lower + 1
                slice_  = slice( lower, upper+1 )
                diff_y  = self.mapped_vector[slice_] - x_curve_y[slice_]
                std_chi_square  = np.sum( diff_y**2 ) / size
                std_diff += np.sqrt( std_chi_square ) / x_curve_y[peak_i]

        self.std_diff   = std_diff * STD_DIFF_SCALE

    def get_sci_list( self ):
        self.sci_list = self.scomp.compute_sci()
        return self.sci_list

    def get_min_sci(self):
        sci_array = np.array(self.get_sci_list())
        return np.min(sci_array)

    def get_opt_params_from_xe( self, xe ):
        start = 0
        i = None
        for k, b in enumerate( self.x_curve.boundaries ):
            if xe >= start and xe <= b:
                i = k
                break
            start = b
        if i is None:
             i = -1

        return self.opt_results[i]

    def get_uv_value( self, xe ):
        ue  = None
        print( 'mapping_ranges=', self.mapping_ranges )
        print( 'x_curve.boundaries=', self.x_curve.boundaries )

        A, B, S = self.get_opt_params_from_xe( xe )
        ue = A*xe + B

        print( 'get_uv_value: ', xe, ue )
        return ue, self.a_curve.spline(ue), S

    def get_opt_params_from_ue( self, ue ):
        start = 0
        i = None
        assert len(self.a_curve.boundaries) == len(self.x_curve.boundaries)
        for k, b in enumerate( self.a_curve.boundaries ):
            if ue >= start and ue <= b:
                i = k
                break
            start = b
        if i is None:
             i = -1
        return self.opt_results[i]

    def get_mapped_flow_changes( self ):
        fc_xe_list = []
        for fc in self.flow_changes:
            if fc is None:
                fc_xe = None
            else:
                A, B, S = self.get_opt_params_from_ue( fc )
                fc_xe = max(0, min( len(self.x_curve.x)-1, int( ( fc - B )/A + 0.5 ) ) )
            fc_xe_list.append( fc_xe )

        return fc_xe_list

    def get_int_ranges( self ):
        return [ [ int(p+0.5) for p in range_  ]  for range_ in self.mapping_ranges ]

    def get_adjustment_deviation( self, debug=False ):

        if debug:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            ax1, ax2 = axes
            ax1.plot(self.a_curve.y)
            ax1.plot(self.a_base, color='red')
            ax2.plot(self.x_curve.y, color='orange')
            ax2.plot(self.x_base, color='red')
            fig.tight_layout()
            plt.show()

        a_adj_dev = np.average(np.abs(self.a_base_adjustment))/self.a_curve.max_y
        x_adj_dev = np.average(np.abs(self.x_base_adjustment))/self.x_curve.max_y
        return a_adj_dev + x_adj_dev

    def make_proof_info(self):
        ret_info = []
        for curve in [self.a_curve, self.x_curve]:
            peak_tops = [int(x) for x in curve.peak_top_x]
            boundaries = [int(x) for x in curve.boundaries]
            ret_info.append([peak_tops, boundaries])
        return ret_info

    def compute_simple_params(self):
        if len(self.a_curve.peak_info) <= 1:
            return

        y = self.a_curve.peak_top_x
        x = self.x_curve.peak_top_x

        def objective(p):
            a, b = p
            return np.sum((a*x + b - y)**2)

        ret = optimize.minimize(objective, (self.A_init, self.B_init))

        self.A_simple = ret.x[0]
        self.B_simple = ret.x[1]
