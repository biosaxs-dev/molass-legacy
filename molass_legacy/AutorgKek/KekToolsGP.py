# coding: utf-8
"""
    KekToolsGP.py

    Copyright (c) 2016-2017, Masatsuyo Takahashi, KEK-PF
"""

import sys
import logging
import traceback
import time
import numpy                as np
from bisect                 import bisect_right
from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker
from IntensityData          import IntensityData, ACCEPTABLE_BASIC_QUALITY
from GuinierPorodFit        import GuinierPorodFit
from Guinier                import Guinier, AIC_SCORE_BOUNDARY
from GuinierInterval        import GuinierInterval
from QrgLimitsAdjuster      import QrgLimitsAdjuster
from Quality                import is_better_quality, fit_score_func
from ErrorResult            import ErrorResult

DEBUG = False

"""
    AutorgKek   ->  GuinierPorodFit     fitting
                    SmoothLineGP        fit result in x, y
                    GuinierInterval     optimization
"""

class AutorgKek:
    def __init__( self, file=None ):
        self.logger = logging.getLogger( __name__ )

        if file is None: return

        if type( file ) == str or type( file ) == np.ndarray:
            try:
                intensity = IntensityData( file, add_smoother=True )
            except ( ValueError, IndexError ):
                # these errors are known for very low quality data
                # which include many negative values.
                etb = ExceptionTracebacker()
                self.logger.error( etb )
                intensity = None
        elif type( file ) == IntensityData:
            intensity = file
        else:
            assert( False )

        self.intensity = intensity
        self.interval = None
        self.optimal_interval_q = None

    def run( self, robust=False, optimize=True,
                qrg_limits=[ 0.0, 1.3 ],
                retry_mode=False
           ):

        if DEBUG: print( 'AutorgKek.run: qrg_limits=', qrg_limits )

        if self.intensity is None or not self.intensity.has_acceptable_quality():
            self.result = ErrorResult()
            return self.result

        better_fit = None
        force_low_quality = False

        self.fit = fit = GuinierPorodFit( self.intensity )
        fit.fit()

        x, y, e = self.intensity.get_guinier_valid_xy()
        self.guinier = Guinier( x, y, e, positive_ratio=self.intensity.positive_ratio )

        # boundary_q = qrg_limits[1] / fit.Rg * np.sqrt( 3*fit.degree/2 )
        boundary_q = qrg_limits[1] / fit.Rg

        if DEBUG: print( 'basic_quality=', self.intensity.basic_quality )

        is_sufficient_baisic_quality = self.intensity.basic_quality >= ACCEPTABLE_BASIC_QUALITY and not force_low_quality

        if is_sufficient_baisic_quality:
            allow_ratio_list    = [ 1 ]
        else:
            allow_ratio_list    = [ 1, 1.2, 1.5, 4.0 ]

        for allow_ratio in allow_ratio_list:
            if DEBUG: print( '[0] --------------------------------- without interval optimization: allow_ratio=', allow_ratio )

            t_ = bisect_right( self.intensity.Q, boundary_q * allow_ratio )

            # print( 'boundary_q=', boundary_q, '; fit.f0, t_=', [ fit.f0, t_  ] )

            """
            if is_sufficient_baisic_quality:
                f0 = fit.f0
            else:
                f0 = 0
            """

            f0 = fit.f0

            if t_ > f0:
                try:
                    self.guinier.estimate_rg( [ f0, t_ ], qrg_limits, fit.Rg )
                    result1 = self.guinier.get_result( fit, result_type=1 )
                    if DEBUG: print( '[ f0, t_ ]=', [ f0, t_ ], ', result1: Rg=', result1.Rg, ', Quality=', result1.Quality )
                except Exception as e:
                    print( e )
                    result1 = None
            else:
                result1 = None

            if result1 is not None and result1.Rg is not None:
                break

        if DEBUG: print( 'result1=', result1 )

        if is_sufficient_baisic_quality:
            opt_result  = None
            opt_quality = None
            boundary_q = qrg_limits[1] / fit.Rg
            # allow_ratio_list    = [ 0.8, 0.9, 1 ]
            allow_ratio_list    = [ 0.9, 1, 1.1, 1.2, 1.5, 2.0 ]

            for i, allow_ratio in enumerate( allow_ratio_list ):
                if DEBUG: print( '[1] --------------------------------- with interval optimization: allow_ratio=', allow_ratio )
                f0 = fit.f0
                t_ = bisect_right( self.intensity.Q, boundary_q * allow_ratio )
                # print( 'f0, t_=', [ f0, t_ ] )
                if f0 >= t_:
                    continue

                try:
                    slice_ = slice( f0, t_+1 )
                    gx = self.intensity.Q[slice_]
                    gy = self.intensity.I[slice_]
                    gw = self.intensity.W[slice_]
                    self.interval = GuinierInterval( self.intensity.Q, self.intensity.I, gx, gy, gw, fit, qrg_limits )
                    min_q, max_q, f_, t_ = self.interval.get_optimal_interval()
                    if DEBUG: print( 'f0, t_=', ( f0, t_ ), '; f_, t_=', ( f_, t_ ) )

                    self.guinier.estimate_rg( [ f0, t_ ], qrg_limits, fit.Rg )
                    result_ = self.guinier.get_result( fit, result_type=2 )
                    if result_ is None or result_.Rg is None:
                        continue

                    self.optimal_interval_q = [ min_q, max_q ]
                    if DEBUG: print( 'optimization is ok. Rg=%.4g, Quality=%.4g, max_qRg=%.4g' % ( result_.Rg, result_.Quality, result_.max_qRg ) )

                except Exception as e:
                    if True:
                        ( exc_type_type_, exc_value, exc_traceback ) = sys.exc_info()
                        e_seq = traceback.format_exception( exc_type_type_, exc_value, exc_traceback )
                        print( 'Optimization Error! ' + ''.join( e_seq ) )
                    result_ = None
                    continue

                if self.intensity.basically_ok:
                    # evaluate quality without q_rg_score
                    if result_ is None:
                        quality_ = 0
                    else:
                        quality_ = np.sum( result_.quality_object.get_factors()[0:5] )

                    if opt_quality is None or quality_ > opt_quality:
                        opt_quality = quality_
                        opt_result  = result_
                else:
                    if opt_result is None or is_better_quality( result_, opt_result, qrg_limits, allow_qrg=0.1 ):
                        # allow_qrg=0.1 makes 'AIMdim01_00158_sub.dat' slightly better ( 21.7 => 23.9 )
                        opt_result = result_
                        # print( 'opt_result.Rg=', opt_result.Rg )

                if i == 4 and opt_result is not None and opt_result.Rg is not None:
                    break

            result2 = opt_result
        else:
            result2 = None

        if result2 is None:
            result2 = self.guinier.get_result( fit, result_type=4 )

        if True:
            if result1 is None:
                self.result = result2
            else:
                if self.intensity.basically_ok:
                    if DEBUG: print( 'fit_consistency1=',  result1.quality_object.fit_consistency_pure)
                    if DEBUG: print( 'fit_consistency2=',  result2.quality_object.fit_consistency_pure)
                    if result2.quality_object.fit_consistency_pure > 0.3:
                        self.result = result2
                    else:
                        if result1.quality_object.fit_consistency_pure > result2.quality_object.fit_consistency_pure:
                            self.result = result1
                        else:
                            self.result = result2
                else:
                    if is_better_quality( result2, result1, qrg_limits ):
                        self.result = result2
                    else:
                        self.result = result1
        else:
            if is_better_quality( result2, result1, qrg_limits ):
                self.result = result2
            else:
                self.result = result1

        assert( qrg_limits is not None )
        try:
            # if self.result is not None: print( 'before adjust result.Rg=', self.result.Rg )

            self.result = self.adjust_to_qrg_limits( qrg_limits, self.result )

            # if self.result is not None: print( 'adjusted result.Rg=', self.result.Rg )
        except Exception as e:
            # when result.Rg is None
            if DEBUG:
                ( exc_type_type_, exc_value, exc_traceback ) = sys.exc_info()
                e_seq = traceback.format_exception( exc_type_type_, exc_value, exc_traceback )
                print( ''.join( e_seq ) )
            pass

        if self.result is None:
            self.result = ErrorResult()
        else:
            self.result = self.guinier.update_result_with_bico_solver( self.result )

        if not retry_mode and self.result.IpI == 1:
            # print( 'retry for IpI' )
            # TODO: 
            start_ipi_retry = self.guinier.determine_ipi_interval()
            # print( 'start_ipi_retry=', start_ipi_retry )

            intensity_for_retry = self.intensity.copy( start=start_ipi_retry )
            retry_obj = AutorgKek( intensity_for_retry )
            retried_result = retry_obj.run( retry_mode=True )
            # print( 'retried_result.Rg=', retried_result.Rg, ', retried_result.Aggregated=', retried_result.Aggregated )

            if retried_result.IpI != 0:
                # keep the low quality as IpI
                retried_result.quality_object.degrade_by_anomalies( 0.5 )
            head_cvmin = self.result.head_cvmin
            self.result = retried_result
            self.result.IpI         = 1
            self.result.head_cvmin  = head_cvmin
            self.result.From        += start_ipi_retry
            self.result.To          += start_ipi_retry

        self.result.Rg_ = 0 if self.result.Rg is None else self.result.Rg
        self.result.I0_ = 0 if self.result.I0 is None else self.result.I0

        return self.result

    def adjust_to_qrg_limits( self, qrg_limits, result, ):
        adjuster = QrgLimitsAdjuster( self.fit, self.guinier, qrg_limits, result )
        return adjuster.get_adjusted_result()

    def proof_plot( self, plot_class, title=None, sleep=0 ):
        plot = plot_class( self.fit, self.interval, self.guinier,  self.optimal_interval_q, title=title )
        block_ = sleep == 0
        plot.show( block=block_ )
        if sleep > 0:
            time.sleep( sleep )
        plot.close()

def autorg( file, robust=False, optimize=True,
            qrg_limits=[ 0, 1.3 ], qrg_limits_apply=True ):
    autorg_ = AutorgKek( file  )
    autorg_.run( robust=robust, optimize=optimize,
                    qrg_limits=qrg_limits )
    return autorg_.result, autorg_.intensity
