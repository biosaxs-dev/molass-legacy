"""
    ElutionDecomposer.py

    Copyright (c) 2018-2023, SAXS Team, KEK-PF
"""
import time
import numpy as np
from scipy.optimize import minimize
import logging
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.BasicUtils import clear_dirs_with_retry, Struct
from molass_legacy.KekLib.NumpyUtils import np_savetxt
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from molass_legacy.Models.ElutionCurveModels import (EGHA, EMG, EMGA, model_debug_plot)
from CanvasDialog import CanvasDialog
from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
from molass_legacy._MOLASS.SerialSettings import get_setting
from FitRecord import FitRecord, sort_fit_recs
from ModelEvaluator import ModelEvaluator

DEBUG   = False
FIT_DEBUG = False

USE_TILTED_MODEL        = False
USE_PENALTY_PEAK_ERROR  = False
USE_LOGARITHMIC_RATIO   = False
LOGARITHMIC_RATIO       = 0.5
VERY_SMALL_VALUE        = 1e-5
VERY_LARGE_VALUE        = 1e10
RESID_FIND_RATIO        = 0.02  # < 0.03 for the 1st resid peak in Ald
MAX_RESID_PEAK_ONLY     = False

if USE_PENALTY_PEAK_ERROR:
    SCALE_PEAK_ERROR    = 1e-10
else:
    # SCALE_PEAK_ERROR    = 1e3     # ok
    SCALE_PEAK_ERROR    = 50

def proof_plot( decomposer, parent=None ):
    dialog = CanvasDialog( "proof_plot", parent=parent )

    def result_plot( fig ):
        ax  = fig.add_subplot(111)
        ax.plot( decomposer.x, decomposer.y, color='orange' )

        x   = decomposer.x
        residual = decomposer.y.copy()

        for _, func, _ in decomposer.fit_recs:
            ey = func( x )
            ax.plot( ey )
            residual -= ey

        ax.plot( residual )
        fig.tight_layout()

    dialog.show( result_plot, toolbar=True )

class ElutionDecomposer:
    def __init__( self, xcurve, x, y, peaks=None, model=None, d_curve=None, retry_valley=False, deeply=False, hints_dict=None, print_peaks=False, debug=False ):
        self.logger  = logging.getLogger( __name__ )

        if hints_dict is not None:
            self.logger.info('decomposition requested with hints_dict %s.' % str(hints_dict) )

        self.hints_dict = hints_dict

        if peaks is None:
            peaks = xcurve.get_emg_peaks()

        if print_peaks:
            print( 'ElutionDecomposer.__init__: peaks=', peaks )

        self.x  = x
        self.y  = y
        self.max_y = np.max(y)

        default_model = EGHA()
        if model is None:
            model   = default_model

        self.decomp_from_separation = get_setting('decomp_from_separation')

        try:
            self.try_decompose( model, xcurve, peaks, x, y, d_curve, retry_valley, deeply )
            self.logger.info('decomposition resulted into %d records with decomp_from_separation=%d.', len(self.fit_recs), self.decomp_from_separation)
        except:
            etb = ExceptionTracebacker()
            print( etb )
            self.logger.warning( "%s.fit failed. resorting to default_model(EGHA)." % ( model.name ) )
            model = default_model
            self.try_decompose( model, xcurve, peaks, x, y, d_curve, retry_valley, deeply )
            self.logger.warning( "%s.fit done." % ( model.name ) )

        self.result_model_name = model.get_name()

        if debug:
            from OptRecsUtils import debug_plot_opt_recs
            debug_plot_opt_recs(xcurve, self.fit_recs, title="ElutionDecomposer Result")

    def rectify_peaks(self, xcurve, peaks, d_curve):
        rectified = False
        diff_top_x = d_curve.primary_peak_i
        if diff_top_x is None:
            return peaks, rectified

        debug = False

        if debug:
            from molass_legacy.Elution.CurveUtils import simple_plot
            fig = plt.figure()
            ax = fig.gca()
            simple_plot(ax, d_curve)

        for k, peak in enumerate(peaks):
            y = peak.get_model_y(self.x)
            if debug:
                ax.plot(y)
            spx = peak.sigma_points
            if spx[0] < diff_top_x and diff_top_x < spx[1]:
                if debug:
                    ax.plot(diff_top_x, d_curve.spline(diff_top_x), 'o', color='cyan')
                    for x_ in spx:
                        y_ = peak.get_model_y([x_])
                        ax.plot(x_, y_[0], 'o', color='yellow')
                peaks = self.add_peak_at_diff_top_x(xcurve, peaks, k, diff_top_x)
                rectified = True
                break
        if debug:
            fig.tight_layout()
            plt.show()

        return peaks, rectified

    def add_peak_at_diff_top_x(self, xcurve, peaks, k, diff_top_x):
        # task: remove this

        from molass_legacy.ElutionCurveDivider import egha_divide

        d_peaks = egha_divide(xcurve, peaks, k, diff_top_x)
        peaks[k] = d_peaks[0]
        peaks.insert(k+1, d_peaks[1])
        self.logger.warning("one peak has been separated into two peaks from the separation between UV and Xray elutions.")

        return peaks

    def try_decompose( self, model, xcurve, peaks, x, y, d_curve, retry_valley, deeply, debug=False):
        if d_curve is None or not self.decomp_from_separation:
            rectified = False
        else:
            peaks, rectified = self.rectify_peaks(xcurve, peaks, d_curve)

        if rectified:
            self.set_fit_res_just_from_peaks(peaks)
            return

        init_fit_recs = self.compute_fit_recs( model, peaks, x, y, debug=debug)
        if FIT_DEBUG:
            self.debug_plot( x, y, init_fit_recs, title="init_fit_recs with " + model.name )

        if retry_valley:
            resid_y = self.compute_residual_curve( init_fit_recs, x, y, debug=debug )
            try:
                valleys = self.get_valleys_to_modify( xcurve, resid_y )
            except:
                self.logger.warning( "failed to get_valleys_to_modify" )
                valleys = []

            if len(valleys) > 0:
                # TODO: optimize this ratio
                def obj_func( ratios ):
                    y_ = self.modify_y( valleys, resid_y, y, ratios )
                    temp_fit_recs = self.compute_fit_recs( model, peaks, x, y_ )    # usually, do not debug this call to avoid many repeats.
                    temp_resid_y = self.compute_residual_curve( temp_fit_recs, x, y )
                    error = np.sum( [ np.sum( temp_resid_y[slice_]**2 ) for slice_ in valleys ] )
                    return error

                init_ratios = np.ones( len(valleys) )
                result = minimize( obj_func, init_ratios )
                opt_ratios = result.x

                y_ = self.modify_y( valleys, resid_y, y, opt_ratios )
                opt_fit_recs = self.compute_fit_recs( model, peaks, x, y_, debug=debug)
                if False:
                    self.debug_plot( x, y_, opt_fit_recs, title="modified y" )
                    self.debug_plot( x, y, opt_fit_recs, title="original y" )
                self.init_fit_recs = opt_fit_recs
            else:
                self.init_fit_recs = init_fit_recs
        else:
            self.init_fit_recs = init_fit_recs

        if deeply:
            fit_recs = self.add_other_peaks( model, peaks, x, y, self.init_fit_recs )
            self.fit_recs = fit_recs
        else:
            self.fit_recs = self.init_fit_recs

        if FIT_DEBUG:
            self.debug_plot( x, y, self.fit_recs, title="fit_recs with " + model.name )

    def set_fit_res_just_from_peaks(self, peaks):
        fit_recs = []
        for kno, peak in enumerate(peaks):
            # chisqr_n = out.chisqr/self.max_y/(stop-start) # TODO
            chisqr_n = 0
            fit_rec = FitRecord(kno, ModelEvaluator(peak.model, peak.opt_params, sign=peak.sign), chisqr_n, peak)
            fit_recs.append( fit_rec )
        self.init_fit_recs = fit_recs
        self.fit_recs = fit_recs

    def compute_residual_curve( self, fit_recs, x, y, debug=False ):
        resid_y = y.copy()
        for rec in fit_recs:
            func = rec[1]
            resid_y -= func( x )

        if debug:
            self.debug_plot( x, y, fit_recs )

        return resid_y

    def debug_plot( self, x, y, fit_recs, title=None ):
        plt.push()
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.cla()
        if title is not None:
            ax.set_title( title )
        ax.plot( x, y, label='input' )
        resid_y = y.copy()
        for k, rec in enumerate(fit_recs):
            func= rec[1]
            fy = func(x)
            ax.plot( x, fy, label='fit-%d' % k )
            resid_y -= fy
        ax.plot( x, resid_y, label='residuals' )
        ax.legend()
        plt.show()
        plt.pop()

    def compute_residual_error( self, fit_recs=None ):
        if fit_recs is None:
            fit_recs = self.fit_recs
        resid_y = self.compute_residual_curve( fit_recs, self.x, self.y )
        return np.sum( resid_y**2 )

    def get_valleys_to_modify( self, xcurve, resid_y ):
        very_small_abs = xcurve.max_y * 0.01
        abs_resid_y = np.abs( resid_y )

        valleys = []
        for j, info in enumerate( xcurve.peak_info ):
            if j > 0:
                k = j - 1
                b = xcurve.boundaries[k]
                # print( [k], 'b, abs_resid_y[b]=', b, abs_resid_y[b] )
                if abs_resid_y[b] > -xcurve.max_y * 0.05:
                    # print( 'need modify ', b )
                    w   = np.where( abs_resid_y[0:b] < very_small_abs )[0]
                    # print( 'w=', w[-5:] )
                    b_lower = w[-1]
                    w   = b + np.where( abs_resid_y[b:] < very_small_abs )[0]
                    # print( 'w=', w[0:5] )
                    b_upper = w[0]
                    # print( 'b_lower, b_upper=', b_lower, b_upper )
                    valleys.append( slice( b_lower, b_upper ) )
        return valleys

    def modify_y( self, valleys, resid_y, y, ratios ):
        y_ = y.copy()

        for k, slice_ in enumerate(valleys):
            y_[slice_] += resid_y[slice_] * ratios[k]

        return y_

    def compute_fit_recs( self, model, peaks, x, arg_y, k_sign=1, simple_try=True, return_chisqr=False, debug=False ):

        y = arg_y.copy()

        fit_recs = []

        ret_chisqr = 0

        # try fit from the largest to the smallest of peak.top_y
        for k, peak in sorted([(k, rec) for k, rec in enumerate(peaks)], key=lambda p: -p[1].top_y):
            # print([k], peak.top_y)
            f, t    = peak.get_fit_limits()
            start   = f
            stop    = t + 1
            slice_  = slice( start, stop )
            x_  = x[slice_]
            y_  = y[slice_]

            if self.hints_dict is None:
                tau_hint = None
            else:
                xkey = peak.get_xkey()
                tau_hint = self.hints_dict.get(xkey)
                # self.logger.info( '[%d] %s tau_hint=%s' % (k, xkey, str(tau_hint)) )

            params  = model.guess(y_, x=x_, tau_hint=tau_hint)

            if simple_try:
                # out = model.fit(y_, params, x=x_, method='least_squares')
                try:
                    out = model.fit(y_, params, x=x_ )
                except:
                    # self.logger.warning( "%s.fit failed. the %dth element has been ignored." % ( model.name, k ) )
                    # logging is not appropriate in an optimizer loop
                    # 20180206 case, which causes this exception, should be investigated
                    etb = ExceptionTracebacker()
                    print( str(etb) )
                    if debug:
                        try:
                            model_debug_plot( model, out.params,  x_, y_, x, arg_y, tau_hint, "after fit", residual=return_chisqr, before_params=params, print_callstack=True )
                        except:
                            etb = ExceptionTracebacker()
                            print( str(etb) )
                    continue
            else:
                try:
                    out = model.fit(y_, params, x=x_)
                except:
                    etb = ExceptionTracebacker()
                    print( str(etb) )
                    # EMG model sometimes raises ValueError("The input contains nan values")
                    self.logger.warning( "%s.fit default method failed. resorting to 'least_squares'." % ( model.name ) )
                    out = model.fit(y_, params, x=x_, method='least_squares')
                    if debug:
                        try:
                            model_debug_plot( model, out.params,  x_, y_, x, arg_y, tau_hint, "after fit", residual=return_chisqr, before_params=params )
                        except:
                            etb = ExceptionTracebacker()
                            print( str(etb) )

            y -= model.eval(out.params, x=x)

            if debug:
                try:
                    model_debug_plot(model, out.params,  x_, y_, x, arg_y, tau_hint, "after fit", residual=return_chisqr, before_params=params, work_y=y)
                except:
                    etb = ExceptionTracebacker()
                    print( str(etb) )

            # print( [k], out.params )
            kno = k*k_sign if k_sign > 0 else (k+1)*k_sign
            chisqr_n = out.chisqr/self.max_y/(stop-start)

            ret_chisqr += chisqr_n

            # ModelEvaluator is too often constructed. why?
            fit_rec = FitRecord(kno, ModelEvaluator(model, out.params, sign=peak.sign), chisqr_n, peak)
            fit_recs.append( fit_rec )

        ret_fit_recs = sort_fit_recs(fit_recs)
        if return_chisqr:
            return ret_fit_recs, ret_chisqr
        else:
            return ret_fit_recs

    def add_other_peaks( self, model, peaks, x, y, init_fit_recs ):
        resid_y = self.y.copy()

        fit_recs = []

        for k, rec in enumerate( init_fit_recs ):
            func = rec[1]
            fit_recs.append(rec)
            ey = func( self.x )
            resid_y -= ey

        """
        exclude cases where resid_y becomes all NaNs as in 0201006_1
        """
        assert not np.isnan(resid_y[0])

        resid_max_y = np.max( resid_y )
        find_max_y = self.max_y * RESID_FIND_RATIO
        if resid_max_y < find_max_y:
            print('find_giveup_ratio=', find_max_y/self.max_y)
            return init_fit_recs

        try:
            debug_plot = False
            abs_resid_y = np.abs(resid_y)
            resid_curve = ElutionCurve( abs_resid_y, low_quality=True, delay_emg_peaks=True, debug_plot=debug_plot )

            max_h   = None
            max_peak = None
            resid_peaks = resid_curve.get_emg_peaks(max_y=find_max_y, allow_wider=True, orig_y=resid_y, debug=False )

            if False:
                plt.push()
                ax = plt.gca()
                ax.set_title( "Peak recognition in the residual curve" )
                ax.plot( x, resid_y, color='brown', alpha=0.3 )
                ax.plot( x, resid_curve.spline(x), color='green' )
                ax.plot( x[[0,-1]], [find_max_y, find_max_y], ':', color='yellow' )
                for info in resid_curve.peak_info:
                    top_x = info[1]
                    ax.plot( top_x, resid_curve.spline(top_x), 'o', color='yellow' )
                for peak in resid_peaks:
                    top_x = peak.top_x
                    ax.plot( top_x, resid_curve.spline(top_x), 'o', color='red' )
                plt.show()
                plt.pop()

            if MAX_RESID_PEAK_ONLY:
                for peak in resid_peaks:
                    h   = resid_curve.spline( peak.top_x )
                    if max_h is None or h > max_h:
                        max_h = h
                        max_peak = peak

            if len(resid_peaks) > 0:
                # set simple_try=False for the residual curve of May11
                peaks_ = [max_peak] if MAX_RESID_PEAK_ONLY else resid_peaks
                added_fit_recs, ret_chisqr = self.compute_fit_recs( model, peaks_, x, abs_resid_y, k_sign=-1, simple_try=False, return_chisqr=True )
                ry = self.compute_residual_curve( added_fit_recs, x, resid_y )
                chisqr = np.sum( ry**2 )
                # TODO: why ret_chisqr is so small even when chisqr is very large?
                if chisqr < VERY_LARGE_VALUE:
                    if FIT_DEBUG:
                        self.debug_plot( x, resid_y, added_fit_recs, title="added_fit_recs with " + model.name)

                    fit_recs += added_fit_recs

        except Exception as exc:
            etb = ExceptionTracebacker()
            # print(etb)
            self.logger.warning( '%s occurred in ElutionDecomposer.add_other_peaks' % etb.last_line() )

            if False:
                # print( etb )
                plt.plot( x, resid_y )
                plt.plot(  max_peak.top_x, max_peak.top_y, 'o', color='red' )
                for x in [ max_peak.flimL, max_peak.flimR ]:
                    plt.plot(  x, resid_y[int(x)], 'o', color='yellow' )
                plt.show()

        return sort_fit_recs(fit_recs)

    def debug_plot_fitted( self, x, y, fit_rec ):
        plt.plot( x, y, color='orange' )
        func = fit_rec[1]
        plt.plot( x, func(x) )
        plt.show()

    def determine_intervals_deprecated( self, peak_info, x, fit_recs ):

        intervals = []
        for k, info in enumerate( peak_info ):
            x_  = x[0:info[2]+1]
            y_  = fit_recs[k]( x_ )
            n   = np.where( y_ < 1e-6 )[0]
            start = n[-1]
            x_  = x[info[0]:]
            y_  = fit_recs[0][k]( x_ )
            n   = np.where( y_ < 1e-6 )[0]
            stop = info[0] + n[0] + 1
            intervals.append( [start, stop] )
        return intervals

    def decompose_toAB( self ):
        Y_list = []
        C_list = []
        for _, func, _ in self.fit_recs:
            Y = func( self.x )
            Y_list.append( Y )
            C_ = Y / np.max( Y )
            C_list.append( C_ )
            C_list.append( C_**2 )

        if True:
            C_list.append( np.ones( len(self.x) ) )
        else:
            Ye = self.y - Y_list[0] - Y_list[1]
            C_list.append( Ye/np.max( Ye ) )
        self.C_matrix = np.vstack( C_list ).T
        C_pinv = np.linalg.pinv( self.C_matrix )
        return np.dot( C_pinv, self.data )

    def save_components( self, folders, entire_elution=True ):
        assert len(folders) == len(self.fit_recs)
        assert entire_elution

        clear_dirs_with_retry( folders  )

        evector = np.zeros( len(self.qvector) )

        for k in range( len(self.fit_recs) ):
            if not entire_elution and k < 2:
                start, stop = self.intervals[k]
            else:
                start, stop = 0, len(self.y)
            folder  = folders[k]
            # for i in range( 0, self.data.shape[0] ):
            k_slice = slice( k*2, k*2 + 2 )
            for i in range( start, stop ):
                curve = np.dot( self.C_matrix[i,k_slice], self.AB[k_slice,:] ).flatten()
                file = folder + '/SCATTER_%05d.dat' % i
                np_savetxt( file, np.vstack( [ self.qvector, curve, evector ] ).T )
