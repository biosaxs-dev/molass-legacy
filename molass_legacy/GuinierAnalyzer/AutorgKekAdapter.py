# coding: utf-8
"""
    AutorgKekAdapter.py

    Copyright (c) 2017-2021, SAXS Team, KEK-PF
"""
import numpy as np
from SimpleGuinier      import SimpleGuinier
from Result             import Result
from ErrorResult        import ErrorResult
from WLS_Rg             import WLS_Rg
from Quality            import STDEV_SCORE_SCALE
from LightObjects       import AnyObject, LightIntensity
from scipy.interpolate  import UnivariateSpline
from molass_legacy._MOLASS.SerialSettings     import get_setting

DEBUG   = False

def get_new_factor_weights():
    return get_setting( 'quality_weighting' )

def new_q_rg_score_func( qrg ):
    if qrg <= 0.2: return 1
    if qrg >= 1.2: return 0
    return ( 1.2 - qrg ) / 1.0

class AdapterQualiy:
    def __init__( self, sg, wls ):
        self.basic_condition    = 0
        self.stdev_ratio    = stdev_ratio = wls.sigmaRg / wls.Rg
        self.basic_quality  = basic_quality = sg.basic_quality
        positive_ratio      = self.compute_positive_ratio( sg.y_ )
        self.positive_score = 2 * ( max( 0.5, positive_ratio ) - 0.5 )
        self.fit_consistency    = sg.end_consistency
        self.fit_consistency_pure   = 0
        stdev_score         = np.exp( -stdev_ratio*STDEV_SCORE_SCALE ) if stdev_ratio > 0 else 0
        self.stdev_score    = stdev_score * basic_quality
        q_rg_score_         = new_q_rg_score_func( sg.x[sg.guinier_start]*sg.Rg )
        self.q_rg_score     = q_rg_score_ * basic_quality
        self.fit_score      = 0
        self.update_quality()

    def compute_positive_ratio( self, I ):
        N = I.shape[0]

        y05, y95 = np.percentile( I, [ 5, 95 ] )
        if y05 > 0:
            if DEBUG: print( 'y05, y95=', y05, y95 )
            dy = ( y95 - y05 )/( N - 1 )
            weights = np.arange( y95, y05 - dy/10, -dy )
            if DEBUG: print( 'N=', N, 'weights.shape=', weights.shape )
        else:
            weights = np.ones( ( N, ) )

        # print( 'weights.shape=', weights.shape, 'N=', N )
        assert( weights.shape[0] == N )

        weights /= np.sum( weights )
        ratio = np.sum( weights[ I > 0 ] )
        if DEBUG: print( 'positive_ratio=', ratio )
        return ratio

    def get_raw_factors( self ):
        return  [   self.basic_quality,
                    self.positive_score,
                    self.fit_consistency,
                    self.fit_consistency,
                    self.stdev_score,
                    self.q_rg_score,
                    self.fit_score,
                ]

    def get_factors( self ):
        weights = get_new_factor_weights()
        return  [   weights[0] * self.basic_quality,
                    weights[1] * self.positive_score,
                    0,
                    weights[2] * self.fit_consistency,
                    weights[3] * self.stdev_score,
                    weights[4] * self.q_rg_score,
                ]

    def get_factors_with_fit_score( self ):
        return self.get_factors() + [ self.fit_score ]

    def update_quality( self ):
        self.quality =  np.sum( self.get_factors() )
        if np.isnan( self.quality ):
            # TODO: remove this case
            print( 'quality factors=', self.get_factors() )
            import logging
            logger = logging.getLogger( __name__ )
            logger.warning( 'quality is set to 0; quality factors=' + str(self.get_factors()) )
            self.quality = 0

        if self.quality <= 0.3:
            self.quality_signal = 'R'
        elif self.quality >= 0.7:
            self.quality_signal = 'G'
        else:
            self.quality_signal = 'Y'

class AutorgKekAdapter:
    def __init__( self, file, guinier=None ):
        if guinier is None:
            self.sg = SimpleGuinier( file )
        else:
            self.sg = guinier
        self.intensity  = LightIntensity( AnyObject( orig_array=self.sg.data ) )

    def run( self, robust=None, optimize=None, fit_result=False, write_log=False ):
        ret = self.run_impl( robust, optimize, fit_result  )

        if write_log:
            from molass_legacy.Test.Tester import write_to_tester_log
            sg = self.sg
            write_to_tester_log( 'guinier result: type=%s(%s), Rg=(%s, %s), basic_quality=%g, score_vector=%s\n'
                        % ( sg.peak_type, str(sg.try_ret), str(ret.Rg), str(sg.Rg), sg.basic_quality, str(sg.score_vector) ) )

        return ret

    def run_impl( self, robust=None, optimize=None, fit_result=False ):
        if self.sg.Rg is None:
            # raise RuntimeError()
            return ErrorResult()

        slice_ = slice( self.sg.guinier_start, self.sg.guinier_stop )
        x_  = self.sg.x2[slice_]
        y_  = self.sg.log_y[slice_]
        e_  = self.sg.log_y_err[slice_]

        w_ = 1/e_**2
        wls = WLS_Rg( x_, y_, w_ )
        if wls.Rg is None:
            w_ = np.ones( len(x_) )
            wls = WLS_Rg( x_, y_, w_ )
            if wls.Rg is None:
                # raise RuntimeError()
                return ErrorResult()

        quality_obj = AdapterQualiy( self.sg, wls )
        if fit_result:
            spline = UnivariateSpline( x_, y_, s=0, ext=3 )
            fit = AnyObject( Rg=99, I0=0, degree=1, q1=0.1, model=spline, result=AnyObject( aic=-1, bic=-1 ) )
        else:
            fit = AnyObject( Rg=0, I0=0, degree=0, result=AnyObject( aic=-1, bic=-1 ) )

        fr_ = self.sg.guinier_start
        to_ = self.sg.guinier_stop - 1

        return Result(
            type = 'A',     # Adapter
            Rg          = wls.Rg,
            Rg_         = wls.Rg,
            Rg_stdev    = wls.sigmaRg,
            I0          = wls.I0,
            I0_         = wls.I0,
            I0_stdev    = wls.sigmaI0,
            From        = fr_,
            To          = to_,
            min_q       = self.sg.min_q,
            max_q       = self.sg.max_q,
            a   = wls.B,
            b   = wls.A,
            fit         = fit,
            min_qRg     = self.sg.min_qRg,
            max_qRg     = self.sg.max_qRg,
            bico_mono_ratio = 1,
            head_trend  = 0,
            IpI         = 0,
            bicomponent = 0,
            result_type = 0,
            quality_object  = quality_obj,
            basic_quality = quality_obj.basic_quality,
            Quality     = quality_obj.quality,
            rg_ratio_for_better_peak = self.sg.rg_ratio_for_better_peak,
            head_gap_ratio = self.sg.head_gap_ratio,
            )
