# coding: utf-8
"""
    Guinier.py

    Copyright (c) 2016, Masatsuyo Takahashi, KEK-PF
"""
import sys
import numpy                as np
from bisect                 import bisect_right
from scipy.interpolate      import UnivariateSpline
from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker
from WLS_Rg                 import WLS_Rg, WLS_HeadTrend
from Quality                import Quality
from Result                 import Result
from SmoothSpline           import SmoothSpline
from BiComponentSolver      import BiComponentSolver, REQUIRED_BASIC_QUALITY

ANALYZABLE_STDEV_RATIO      = 0.04
IPI_MAX_STDEV_RATIO         = 0.01
# IPI_REQ_BASIC_QUALITY       = 0.9
IPI_CURVATURE_BOUNDARY      = -0.3      # -2.08 for ts008_ave_sub.dat
FIT_CONSISTENCY_PURE_BOUNDARY   = 0.9
AIC_SCORE_BOUNDARY          = 9.0
REQUIRED_QUALITY            = 0.2
HEAD_TREND_POLY_BOUNDARY    = -0.5
POLY_QRG_SCORE_BOUNDARY     = 1e-5
# POLY_QRG_SCORE_BOUNDARY     = 0.6     # 0.51 for ts008_ave_sub.dat
POLY_REQ_BASIC_CONDITION    = 0.9       # for Mei_subtRNA_00189_sub.dat, etc.
POLY_STRICT_STDEV_RATIO     = 0.007     # 0.009 for 
POLY_NARROW_RATIO_BOUNDARY  = 0.05      # 0.07 for AIMdim01_00226_sub.dat
POLY_WIDE_RATIO_BOUNDARY    = 0.8       # 0.700 for ts008_ave_sub.dat
POLY_ASSERTABLE_PROPORTION  = 0.92
POLY_ASSERTABLE_DIFF_RATIO  = 0.08      # Rg_diff= 0.137645266692 for OA_1.0_ave_sub.dat

DEBUG = False

class Guinier:
    def __init__( self, x, y, e, positive_ratio=None ):
        self.x      = x
        self.y      = y
        self.e      = e
        self.wls        = None
        self.interval   = None
        self.positive_ratio = positive_ratio
        self.sx_array   = None
        self.anomalies  = None

    def estimate_rg( self, interval, qrg_limits=None, Rg=None ):
        if Rg is None or qrg_limits is None:
            f_, t_ = interval
        else:
            f_, t_ = self.get_qrg_limits_indeces( interval, qrg_limits, Rg )

        # temp fix
        if t_ >= len( self.x ):
            t_ = len( self.x ) - 1

        if DEBUG: print( 'estimate_rg: (1) f_, t_=', ( f_, t_ ) )
        if f_ >= t_:
            raise Exception( 'f_(%d) >= t_(%d)' % ( f_, t_  ) )

        slice_ = slice( f_, t_+1 )
        x = self.x[ slice_ ]
        y = self.y[ slice_ ]
        e = self.e[ slice_ ]

        w = 1/e**2
        wls = WLS_Rg( x, y, w )

        self.wls        = wls
        self.interval   = [ f_, t_ ]
        if DEBUG: print( 'estimate_rg: (2) ', (f_, t_), self.wls.Rg )

    def get_qrg_limits_indeces( self, interval, qrg_limits, Rg  ):
        f_, t_ = interval

        limits = list( map( lambda q: bisect_right( self.x, q/Rg ), qrg_limits ) )
        if limits[0] > f_ and limits[0] < t_:
            f_ = limits[0]
        if limits[1] < t_ and limits[1] > f_:
            t_ = limits[1]

        if DEBUG: print( interval, '=>', (f_, t_) )

        return f_, t_

    def get_quality( self, fit ):
        # TODO: interval conversion
        if self.wls is None:
            quality_obj = Quality( fit, None, None, self.interval, self.positive_ratio, self.x )
        else:
            quality_obj = Quality( fit, self.wls.Rg, self.wls.sigmaRg, self.interval, self.positive_ratio, self.x )
        return quality_obj

    def get_basic_result( self, fit=None ):
        return (
                self.wls.B, self.wls.A,
                self.wls.Rg, self.wls.sigmaRg,
                self.wls.I0, self.wls.sigmaI0,
                )

    def get_result( self, fit=None, result_type=None, result=None ):

        self.fit = fit

        if DEBUG: print( 'get_result----------------------------------------------------------' )

        if result is None:
            if DEBUG: print( 'Guinier: get_result None' )

            if self.wls is None:
                Rg          = None
                Rg_stdev    = None
                I0          = None
                I0_stdev    = None
                a, b        = None, None
            else:
                wls         = self.wls
                Rg          = wls.Rg
                Rg_stdev    = wls.sigmaRg
                I0          = wls.I0
                I0_stdev    = wls.sigmaI0
                a, b        = wls.B, wls.A

            if self.interval is None:
                f_, t_      = None, None
                minQ        = None
                maxQ        = None
            else:
                f_, t_      = self.interval
                minQ        = np.sqrt( self.x[f_] )
                maxQ        = np.sqrt( self.x[t_] )

        else:
            if DEBUG: print( 'Guinier: get_result from Struct' )

            assert( hasattr( result, 'wls' ) )
            self.wls    = result.wls
            wls         = result.wls
            Rg          = wls.Rg
            Rg_stdev    = wls.sigmaRg
            I0          = wls.I0
            I0_stdev    = wls.sigmaI0
            a, b        = wls.B, wls.A

            f_, t_      = result.From, result.To
            self.interval = [ f_, t_ ]
            minQ        = np.sqrt( self.x[f_] )
            maxQ        = np.sqrt( self.x[t_] )

            if result_type is None:
                result_type = result.result_type

        curvatures_array = None
        IpI             = None
        head_cvmin      = None
        bicomponent  = None
        bico_mono_ratio    = None
        anomalies       = None

        if fit is None or self.wls is None:
            quality_obj     = None
            quality         = None
            quality_signal  = None
            head_trend      = None
            anomaly_index   = 0
            head_trend      = None
            head_proportion = None
            remarks         = None
        else:
            remarks         = fit.remarks

            quality_obj     = self.get_quality( fit )

            head_trend, head_proportion = self.compute_head_data_trend( fit )
            if DEBUG: print( 'head_trend, head_proportion=', head_trend, head_proportion )

            try:
                anomalies   = self.compute_anomaly_indeces( fit, quality_obj, minQ, I0, Rg, head_trend )
                if anomalies is not None:
                    IpI             = anomalies[2]
                    head_cvmin      = anomalies[3]
                    bicomponent  = anomalies[4]      # not yet set
                    bico_mono_ratio    = anomalies[5]      # not yet set
            except:
                etb = ExceptionTracebacker()
                print(etb)

            quality         = quality_obj.quality
            quality_signal  = quality_obj.quality_signal

        if fit is None:
            gpfit_RG    = None
            gpfit_I0    = None
            gpfit_d     = None
            gpfit_minQ  = None
            gpfit_maxQ  = None
        else:
            gpfit_RG    = fit.Rg
            gpfit_I0    = fit.I0
            gpfit_d     = fit.degree
            # gpfit_minQ  = fit.Q[fit.f0]
            # gpfit_maxQ  = fit.Q[fit.t0]
            gpfit_minQ  = fit.Q[0]
            gpfit_maxQ  = fit.Q[-1]

        if quality_obj is None:
            basic_quality = None
            positive_score  = None
            fit_cover_ratio = None
            fit_consistency = None
            stdev_score     = None
            q_rg_score      = None
        else:
            basic_quality = quality_obj.basic_quality
            positive_score  = quality_obj.positive_score
            fit_cover_ratio = quality_obj.fit_cover_ratio
            fit_consistency = quality_obj.fit_consistency
            stdev_score     = quality_obj.stdev_score
            q_rg_score      = quality_obj.q_rg_score

        if anomalies is None or anomalies[-1] is None:
            bico_G1, bico_G2, bico_Rg1, bico_Rg2, bico_d1, bico_d2 = [ None ] * 6
        else:
            bico_G1, bico_G2, bico_Rg1, bico_Rg2, bico_d1, bico_d2 = anomalies[-1] 

        if DEBUG: print( 'bicomponent=', bicomponent )

        result = Result(
                    type='E',
                    Rg=Rg, Rg_stdev=Rg_stdev,
                    I0=I0, I0_stdev=I0_stdev,
                    From=f_, To=t_,
                    Quality=quality,
                    Aggregated=head_trend,
                    IpI=IpI,
                    head_cvmin=head_cvmin,
                    bicomponent=bicomponent,
                    bico_mono_ratio=bico_mono_ratio,
                    min_qRg=None if Rg is None else minQ * Rg,
                    max_qRg=None if Rg is None else maxQ * Rg,
                    min_curvature=None,
                    max_curvature=None,
                    a=a,
                    b=b,
                    remarks=remarks,
                    fit=fit,

                    gpfit_Rg    = gpfit_RG,
                    gpfit_I0    = gpfit_I0,
                    gpfit_d     = gpfit_d,
                    gpfit_minQ  = gpfit_minQ,
                    gpfit_maxQ  = gpfit_maxQ,

                    basic_quality = basic_quality,
                    positive_score  = positive_score,
                    fit_cover_ratio = fit_cover_ratio,
                    fit_consistency = fit_consistency,
                    fit_consistency_pure = None if quality_obj is None else quality_obj.fit_consistency_pure,
                    stdev_score     = stdev_score,
                    q_rg_score      = q_rg_score,

                    bico_G1  = bico_G1,
                    bico_G2  = bico_G2,
                    bico_Rg1 = bico_Rg1,
                    bico_Rg2 = bico_Rg2,
                    bico_d1  = bico_d1,
                    bico_d2  = bico_d2,
                    bico_result = None,

                    signal=quality_signal,
                    min_q=minQ,
                    max_q=maxQ,

                    # quality details
                    quality_object=quality_obj,
                    head_trend=head_trend,
                    head_proportion=head_proportion,
                    curvatures_array = curvatures_array,
                    result_type=result_type,
                    anomalies=anomalies,
                    )

        if DEBUG:
            fit_consistency_pure = None if quality_obj is None else quality_obj.fit_consistency_pure
            print( [f_, t_], ' Rg=', Rg, ', Quality=', quality, ', fit_pure=', fit_consistency_pure )

        return result

    def update_result_with_bico_solver( self, result ):

        if DEBUG: print( 'update_result_with_bico_solver: result.anomalies=', result.anomalies )

        bicomponent, bico_mono_ratio, bico_result = self.compute_bicomponent_info( result )
        # print( 'bicomponent=', bicomponent, '; bico_mono_ratio=', bico_mono_ratio )

        result.bicomponent    = bicomponent
        result.bico_mono_ratio  = bico_mono_ratio
        result.bico_result      = bico_result

        if bico_result is not None:
            G1, G2, Rg1, Rg2, d1, d2 = bico_result[1]
            G_ = G1 + G2
            result.bico_G1  = int(G1/G_*100)
            result.bico_G2  = int(G2/G_*100)
            result.bico_Rg1 = Rg1
            result.bico_Rg2 = Rg2
            result.bico_d1  = d1
            result.bico_d2  = d2

        return result

    def compute_head_data_trend( self, fit ):
        if self.interval is None:
            return None, None

        # f0_ = fit.start_point
        f0_      = fit.f0
        f2_, t2_ = self.interval

        head_size = f2_ - f0_ + 1

        if DEBUG: print( 'head_size=', head_size, ', self.wls.sigmaB=', self.wls.sigmaB )

        if head_size >= 8:
            # avoid calculation for too short intervals
            # print( 'head trend: f0_, f2_=', (f0_, f2_) )
            slice_ = slice( f0_, f2_ )
            x_ = self.x[slice_]
            y_ = self.y[slice_]
            w_ = 1/self.e[slice_]**2

            ht = WLS_HeadTrend( x_, y_, w_ )
            guin_size = self.x[t2_] - self.x[f2_]
            trend, prop = ht.compute_trend( self.wls.B, guin_size )

        else:
            trend, prop = 0, 1

        return trend, prop

    def compute_anomaly_indeces( self, fit, quality_obj, minQ, I0, Rg, head_trend ):

        # f0_ = fit.start_point
        f0_ = fit.f0

        t_array = []

        """"
        if quality_obj.fit_cover_ratio < 0.5:
            d2_ex = 1
        else:
            d2_ex = 1.5 if fit.degree >= 3 else 2
            # d2_ex = 1.5
        d_params = [ [ 1, 1 ], [ fit.degree, d2_ex ] ]
        if quality_obj.fit_consistency_pure < 0.5 or quality_obj.fit_cover_ratio < 0.5:
            # if the consistency is low, check also in a wider interval
            d4_ex = 1.5 if fit.degree >= 3 else 2       # if fit.degree is small, you must make the interval wider.
            d_params.append( [ 4, d4_ex ] )
        """

        for d, ex in [ [ 1, 0.5 ] ]:
            q_ = 1/fit.Rg * np.sqrt( 3*d/2 )
            qq_ = q_**2 * ex
            """
                ex=1.5 for d=2 is too narrow    for subtract_SSC
                ex=2   for d=4 is too wide      for OA
            """
            t_ = bisect_right( self.x, qq_ )
            t_array.append( [ d, t_ ] )

        self.sx_array = []
        self.spline_array = []
        self._y_array = []
        self.sy_array = []
        self.cy_array = []
        self.curvatures_array = []

        if DEBUG: print( 'compute_anomaly_indeces: t_array=', t_array, ', y.shape[0]=', self.y.shape[0] )

        for i, param in enumerate( t_array ):

            d, t = param
            t_ = max( f0_ + 10, t )

            if t_ >= self.y.shape[0]:
                t_ = self.y.shape[0] - 1

            if DEBUG: print( '[%d] d=%g, f0_=%d, t_=%d' % ( i, d, f0_, t_ ) )

            slice_ = slice( f0_, t_+1 )

            sx = self.x[slice_]
            _y = self.y[slice_]
            w_ = 1/self.e[slice_]

            try:
                # with_smoother = i == 0
                with_smoother = False       # smoother not required by IPI_MAX_STDEV_RATIO check
                smooth_spline = SmoothSpline( sx, _y, w=w_, curvature=True, with_smoother=with_smoother )
            except Exception as e:
                if DEBUG:
                    etb = ExceptionTracebacker()
                    print(etb)

                self.sx_array = None
                raise e

            spline  = smooth_spline.spline
            self.spline_array.append( spline )
            sy = spline( sx )
            cy = smooth_spline.cy
            curvatures = smooth_spline.cv_array

            self.sx_array.append( sx )
            self._y_array.append( _y )
            self.sy_array.append( sy )
            self.cy_array.append( cy )
            self.curvatures_array.append( curvatures )

        # self.sx = self.sx_array[0]
        # self._y = self._y_array[0]

        cvmin   = self.curvatures_array[0][0]

        if DEBUG:
            print(  'basic_quality=', quality_obj.basic_quality,
                    ', stdev_ratio=', quality_obj.stdev_ratio,
                    ', quality=', quality_obj.quality )

        if ( quality_obj.aic_score >= AIC_SCORE_BOUNDARY    # quality_obj.aic_score < AIC_SCORE_BOUNDARY for agg1 or IpI
            and (   quality_obj.basic_quality   < REQUIRED_BASIC_QUALITY
                or  quality_obj.quality         < REQUIRED_QUALITY
                )
            ):
            if DEBUG: print( "Can't analize anomalies." )
            return None

        if DEBUG: print( 'cvmin=', cvmin, '; curvatures_array=', self.curvatures_array )

        # ipi_value   = 0 if Rg is None or quality_obj.fit_cover_ratio > 0.5 else cvmin
        ipi_value   = cvmin
        if (  ( quality_obj.stdev_ratio <= IPI_MAX_STDEV_RATIO
                # and quality_obj.basic_quality >= IPI_REQ_BASIC_QUALITY
                # cvmin == -18 for AIMdim01_00199_sub.dat
                # or fit.intensity.ha_end >= 10
              )
                and ipi_value < IPI_CURVATURE_BOUNDARY ):
            ipi_index   = 1
        else:
            ipi_index   = 0

        if self.wls.Rg is not None:
            stdev_ratio = self.wls.sigmaRg / self.wls.Rg
        else:
            stdev_ratio = 1

        bico_mono_ratio   = None
        bico_params  = None

        if DEBUG: print(    'head_trend=', head_trend,
                            ', fit_consistency_pure=', quality_obj.fit_consistency_pure,
                            ', q_rg_score=', quality_obj.q_rg_score )

        if ( ( quality_obj.basic_condition >= POLY_REQ_BASIC_CONDITION
                or quality_obj.basic_condition >= POLY_REQ_BASIC_CONDITION - 0.1
                    and head_trend < HEAD_TREND_POLY_BOUNDARY    # relaxed for agg2/TYINAs105_ave_sub.dat
              )
            and ( quality_obj.fit_consistency_pure < FIT_CONSISTENCY_PURE_BOUNDARY
                or quality_obj.aic_score < AIC_SCORE_BOUNDARY
                or head_trend < HEAD_TREND_POLY_BOUNDARY
                or quality_obj.q_rg_score < POLY_QRG_SCORE_BOUNDARY  # for TY03s304_ave0-4_sub.dat, TY30s305_ave0-4_sub.dat
                or False     # True for bico_mono_ratio's for all
                )
            ):

            aggregation = None
            quality_obj.needs_bico_solver = True
            if head_trend < HEAD_TREND_POLY_BOUNDARY:
                aggregation = -1
        else:
            aggregation = 0

        # anomalies = [ aggregation, ipi_index, bico_mono_ratio, bico_params ]
        num_anomalous_points = fit.intensity.ha_end
        anomalies = [ head_trend, num_anomalous_points, ipi_index, cvmin, aggregation, bico_mono_ratio, bico_params ]

        if DEBUG: print( 'anomalies=', anomalies )

        self.anomalies = anomalies

        return anomalies

    def comupute_bico_mono_ratio( self, fit ):

        bico_result = None

        solver = BiComponentSolver( fit.intensity, no_assert=True )
        try:
            bico_result = solver.solve( fit=fit )
        except:
            etb = ExceptionTracebacker()
            print(etb)
            return None, bico_result

        bico_ee, bico_params, I0_est = bico_result

        G1, G2, Rg1, Rg2, d1, d2 = bico_params

        G_  = G1 + G2
        Rg_ = ( G1*Rg1 + G2*Rg2 ) / G_
        d_  = ( G1*d1  + G2*d2 )  / G_
        gp_ee_a = solver.evaluate_gp( G_, Rg_, d_ )
        gp_ee_f = solver.evaluate_gp( fit.I0, fit.Rg, fit.degree )
        gp_ee = min( gp_ee_a, gp_ee_f )

        bico_mono_ratio = bico_ee / gp_ee
        if DEBUG:
            print( 'Guinier: gp_ee_a=', gp_ee_a, ', gp_ee_f', gp_ee_f )
            print( 'Guinier: bico_ee=', bico_ee )
            print( 'Guinier: gp_ee=',  gp_ee )
            print( 'Guinier: bico_mono_ratio=',  bico_mono_ratio )

        return bico_mono_ratio, bico_result

    def compute_bicomponent_info( self, result ):

        bico_result = None

        # print( 'compute_bicomponent_info: needs_bico_solver=', result.quality_object.needs_bico_solver  )
        if result.quality_object is not None and result.quality_object.needs_bico_solver:
            bico_mono_ratio, bico_result = self.comupute_bico_mono_ratio( result.fit )
        else:
            bico_mono_ratio = None

        head_trend      = result.head_trend
        head_proportion = result.head_proportion
        quality_obj     = result.quality_object
        anomalies       = result.anomalies

        if head_trend is None:
            bicomponent  = None
        else:
            if DEBUG:
                print( 'anomalies=', anomalies )
                print( 'fit_cover_ratio=', quality_obj.fit_cover_ratio, ', fit_score=', quality_obj.fit_score, ', head_trend=', head_trend )

            if quality_obj.basic_condition < REQUIRED_BASIC_QUALITY:

                bicomponent  = None

            else:
                # TODO: clarify the case where bico_mono_ratio is None

                polycomp_score = 0

                if quality_obj.stdev_ratio < ANALYZABLE_STDEV_RATIO:
                    if head_trend < HEAD_TREND_POLY_BOUNDARY and head_proportion > 0.5:
                        polycomp_score  += 1

                if bico_mono_ratio is not None:

                    if quality_obj.stdev_ratio < POLY_STRICT_STDEV_RATIO:
                        if bico_mono_ratio < POLY_WIDE_RATIO_BOUNDARY:
                            polycomp_score  += 1    # for ts008_ave_sub.dat
                    else:
                        if bico_mono_ratio < POLY_NARROW_RATIO_BOUNDARY:
                            polycomp_score  += 1    # except for AIMdim01_00226_sub.dat

                    G1, G2, Rg1, Rg2, d1, d2 = bico_result[1]
                    I0 = G1 + G2
                    pc1 = G1 / I0
                    pc2 = G2 / I0
                    Rg_diff = abs( Rg1 - Rg2 ) / ( Rg1 + Rg2 )
                    # print( 'Rg_diff=', Rg_diff )

                    if (    pc1 < POLY_ASSERTABLE_PROPORTION
                        and pc2 < POLY_ASSERTABLE_PROPORTION
                        and Rg_diff > POLY_ASSERTABLE_DIFF_RATIO
                       ):
                       polycomp_score  += 1

                # print( 'polycomp_score=', polycomp_score )
                if polycomp_score >= 2:
                    bicomponent   = 1
                else:
                    bicomponent   = 0

                if result.anomalies is not None:
                    result.bicomponent    = bicomponent
                    result.anomalies[5]     = bico_mono_ratio

        return bicomponent, bico_mono_ratio, bico_result

    def determine_ipi_interval( self ):
        # print( 'determine_ipi_interval' )
        gp_slope = - self.fit.Rg**2 / 3

        start_ipi_retry = None

        sx = self.sx_array[0]
        sy = self.sy_array[0]
        for i in range( len( sx ) - 10 ):
            x0 = sx[i]
            y0 = sy[i]
            x1 = sx[i+10]
            y1 = sy[i+10]
            slope = ( y1 - y0 ) / ( x1 - x0 )
            slope_ratio = slope/gp_slope
            # print( '[%d] slope ratio=%g' % ( i, slope_ratio ) )
            if slope_ratio >= 0.95:
                break

        # this includes the case when slope_ratio < 0.95
        start_ipi_retry = i

        return start_ipi_retry
