# coding: utf-8
"""
    Quality.py

    Copyright (c) 2016-2017, Masatsuyo Takahashi, KEK-PF
"""
import numpy        as np
from Settings       import get_setting


DEBUG = False

STDEV_SCORE_SCALE   = 10.0

RAW_BASIC_QUALITY   = 0
RAW_FIT_SCORE       = 6
RAW_STDEV_SCORE     = 4

FIT_CONSISTENCY     = 2

def get_factor_weights():
    return get_setting( 'quality_weighting' )

def q_rg_score_func( qrg ):
    if qrg <= 0.3: return 1
    if qrg >= 0.8: return 0
    return ( 0.8 - qrg ) / 0.5

base_part       = 0.4
bonus_part      = 1 - base_part
bonus_rate      = 10
bonus_boundary  = 1 - 1 / bonus_rate    # 0.9

AIC_SCORE_MIN   =  8
AIC_SCORE_MAX   = 10

def fit_score_func( aic ):
    aic_score = np.log( -aic ) if aic < 0 else 0
    aic_ = min( max( AIC_SCORE_MIN,  aic_score ), AIC_SCORE_MAX )
    fit_score = ( aic_ - AIC_SCORE_MIN ) / ( AIC_SCORE_MAX - AIC_SCORE_MIN )
    return fit_score

def fit_consistency_func( fit_Rg, Rg, adopt_ratio=1 ):
    rg_diff_score   = max( 0, 1 - abs( fit_Rg - Rg ) / fit_Rg )

    if rg_diff_score <= bonus_boundary:
        consistency = base_part * rg_diff_score
    else:
        consistency = base_part * rg_diff_score + bonus_part * ( rg_diff_score - bonus_boundary ) * bonus_rate

    return consistency * adopt_ratio

def compute_atsas_fit_consistency( fit_Rg, atsas_Rg, raw_factors ):
    adopt_ratio = np.power( raw_factors[RAW_FIT_SCORE] * raw_factors[RAW_STDEV_SCORE] * raw_factors[RAW_BASIC_QUALITY], 1/4 )
    atsas_fit_consistency   = fit_consistency_func( fit_Rg, atsas_Rg, adopt_ratio=adopt_ratio ) * get_factor_weights()[ FIT_CONSISTENCY ]
    return atsas_fit_consistency

def fit_consistency_with_stdev( fit_Rg, Rg, Rg_stdev ):
    stdev_ratio         = Rg_stdev / Rg
    stdev_score         = np.exp( -stdev_ratio*STDEV_SCORE_SCALE ) if stdev_ratio > 0 else 0

    fit_consistency_pure    = fit_consistency_func( fit_Rg, Rg )
    fit_consistency     = fit_consistency_pure * stdev_score
    return fit_consistency

class Quality:
    def __init__( self, fit, Rg, Rg_stdev, interval, positive_ratio, x ):

        # self.fit    = fit         # this is bad for garbage collection
        self.fit_Rg = fit.Rg

        self.Rg     = Rg
        basic_condition     = fit.intensity.basic_condition
        basic_quality       = fit.intensity.basic_quality
        positive_score      = 2 * ( max( 0.5, positive_ratio ) - 0.5 )

        fit_score   = fit_score_func( fit.result.aic )

        if Rg is None:
            fit_consistency_pure    = 0
            fit_consistency     = 0
            stdev_ratio         = 1
            stdev_score         = 0
            q_rg_score          = 0
        else:
            stdev_ratio         = Rg_stdev / Rg
            if DEBUG: print( 'stdev_ratio=', stdev_ratio )
            stdev_score         = np.exp( -stdev_ratio*STDEV_SCORE_SCALE ) if stdev_ratio > 0 else 0

            fit_consistency_pure= fit_consistency_func( fit.Rg, Rg )
            fit_consistency     = fit_consistency_pure * np.power( fit_score * stdev_score * basic_quality, 1/4 )

            f2_, t2_ = interval
            # size_score      = min( 20, t2_ - f2_ + 1 ) / 20
            q_rg_score = q_rg_score_func( np.sqrt( x[f2_] ) * Rg )
            q_rg_score *= basic_quality

        if DEBUG:
            print( 'fit.cover_ratio=', fit.cover_ratio )
            print( 'fit_score=', fit_score )
            print( 'fit_consistency=', fit_consistency )
            print( 'stdev_score=', stdev_score )
            print( 'q_rg_score=', q_rg_score )

        self.aic_score          = np.log( -fit.result.aic )     # internal use only
        self.basic_condition    = basic_condition               # internal use only

        self.basic_quality      = basic_quality
        self.positive_score     = positive_score
        self.fit_cover_ratio    = fit.cover_ratio * basic_quality
        self.fit_score          = fit_score
        self.fit_consistency_pure   = fit_consistency_pure
        self.fit_consistency    = fit_consistency
        self.stdev_ratio        = stdev_ratio
        self.stdev_score        = stdev_score * basic_quality
        self.q_rg_score         = q_rg_score
        self.needs_bico_solver   = False

        self.update_quality()

    def update_quality( self ):
        self.quality =  np.sum( self.get_factors() )

        if self.Rg is None:
            self.quality = max( 0, self.quality - 0.1 )

        if self.quality <= 0.3:
            self.quality_signal = 'R'
        elif self.quality >= 0.7:
            self.quality_signal = 'G'
        else:
            self.quality_signal = 'Y'

    def get_raw_factors( self ):
        return  [   self.basic_quality,
                    self.positive_score,
                    self.fit_cover_ratio,
                    self.fit_consistency,
                    self.stdev_score,
                    self.q_rg_score,
                    self.fit_score,
                ]

    def get_factors( self ):
        weights = get_factor_weights()
        return  [   weights[0] * self.basic_quality,
                    weights[1] * self.positive_score,
                    0 * self.fit_cover_ratio,
                    weights[2] * self.fit_consistency,
                    weights[3] * self.stdev_score,
                    weights[4] * self.q_rg_score,
                ]

    def get_factors_with_fit_score( self ):
        return self.get_factors() + [ self.fit_score ]

    def degrade_by_anomalies( self, deg_score ):
        self.fit_score = max( 0, self.fit_score - deg_score )
        if self.Rg is not None:
            self.fit_consistency = fit_consistency_func( self.fit_Rg, self.Rg,
                                    adopt_ratio=np.power( self.fit_score * self.stdev_score * self.basic_quality, 1/4 ) )
        self.update_quality()

def is_better_quality( r1, r2, qrg_limits, allow_qrg=0.0 ):
    # print( 'is_better_quality: r1=', r1, ', r2=', r2 )

    if r1 is None:
        return False

    if r2 is None:
        return True

    if r1.Rg is None:
        return False

    if r2.Rg is None:
        return True

    qrg_limit = qrg_limits[1] + allow_qrg
    if r1.max_qRg > qrg_limit:
        if DEBUG: print( 'r1.max_qRg(%g) > qrg_limit(%g)' % ( r1.max_qRg, qrg_limit  ) )
        if r2.max_qRg <= qrg_limit:
            return False

    if r2.max_qRg > qrg_limit:
        if DEBUG: print( 'r2.max_qRg(%g) > qrg_limit(%g)' % ( r2.max_qRg, qrg_limit  ) )
        return True

    if DEBUG:
        print( 'Rg             : %.4g, %.4g' % ( r1.Rg, r2.Rg ) )
        print( 'Aggregated     : %s, %s' % ( r1.Aggregated, r2.Aggregated ) )
        print( 'Quality        : %.4g, %.4g' % ( r1.Quality, r2.Quality ) )
        print( 'basic_quality: %.4g, %.4g' % ( r1.quality_object.basic_quality, r2.quality_object.basic_quality ) )
        print( 'fit_consistency: %.4g, %.4g' % ( r1.quality_object.fit_consistency_pure, r2.quality_object.fit_consistency_pure ) )

    """
    if r1.Aggregated is None:
        if r2.Aggregated is None:
            pass
        else:
            return False
    else:
        if r2.Aggregated is None:
            return True
        else:
            if r1.Aggregated == r2.Aggregated:
                pass
            else:
                # take it if any anomaly is detected
                if abs( r1.Aggregated ) > abs( r2.Aggregated ):
                    return True
    """

    if r1.quality_object.basic_quality > 0.5 and r2.quality_object.fit_consistency_pure > 0.2:
        # print( "is_better_quality fit_consistency_pure:", r1.quality_object.fit_consistency_pure, r2.quality_object.fit_consistency_pure )
        if r1.quality_object.fit_consistency_pure > r2.quality_object.fit_consistency_pure:
            return True
        else:
            return False
    else:
        if r1.Quality > r2.Quality:
            return True
        else:
            return False
