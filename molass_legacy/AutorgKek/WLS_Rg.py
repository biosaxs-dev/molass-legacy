# coding: utf-8
"""
    WLS_Rg.py

    Copyright (c) 2016, Masatsuyo Takahashi, KEK-PF
"""
import numpy            as np
import statsmodels.api  as sm
from SAngler_WLS        import SAngler_WLS

class LS_Rg:
    def set_result( self, res ):
        A, B = res.params

        self.A  = A
        self.B  = B

        sigmaA, sigmaB = np.sqrt( np.diag( res.cov_params() ) )
        # print( 'sigmaA, sigmaB=', sigmaA, sigmaB )

        self.sigmaA = sigmaA
        self.sigmaB = sigmaB
        self.Rg     = np.sqrt( 3*(-B) ) if B < 0 else None
        self.I0     = np.exp( A )
        self.sigmaRg    = 1/2 / self.Rg * 3 * sigmaB if B < 0 else None
        self.sigmaI0    = self.I0 * sigmaA

class WLS_Rg( LS_Rg ):
    def __init__( self, x, y, w ):
        X = sm.add_constant(x)
        mod = sm.WLS( y, X, weights=w )
        res = mod.fit()
        self.set_result( res )

class OLS_Rg( LS_Rg ):
    def __init__( self, x, y ):
        X = sm.add_constant(x)
        mod = sm.OLS( y, X )
        res = mod.fit()
        self.set_result( res )

class SAngler_WLS_Rg( LS_Rg ):
    def __init__( self, x, y, w ):
        mod = SAngler_WLS( y, x, w )
        res = mod.fit()
        self.set_result( res )

class ODR_WLS_Rg( LS_Rg ):
    def __init__( self, x, y, w ):
        from ODR_WLS            import ODR_WLS
        mod = ODR_WLS( y, x, w )
        res = mod.fit()
        self.set_result( res )

class WLS_HeadTrend:
    def __init__( self, x, y, w ):
        self.x  = x
        X = sm.add_constant(x)
        mod = sm.WLS( y, X, weights=w )
        self.res = res = mod.fit()

    def compute_trend( self, guin_b, guin_size ):

        guin_angle = np.arctan2( guin_b, 1 )
        # print( 'guin_b=', guin_b, ', guin_anglee=', guin_angle )

        head_b = self.res.params[1]
        head_angle = np.arctan2( head_b, 1 ) 
        # print( 'head_b=', head_b, ', head_angle=', head_angle )

        trend_angle = ( head_angle - guin_angle ) * 180/np.pi
        # print( 'trend_angle=', trend_angle )
        trend_angle_ = max( -0.5, min( +0.5, trend_angle ) )

        head_size = self.x[-1] - self.x[0]
        proportion = min( 1, head_size/guin_size )
        # print( 'head_size, guin_size, proportion=', head_size, guin_size, proportion )

        if proportion > 0.5:
            trend = min( 1, trend_angle_ * proportion * 10 )
        else:
            # proportion==0.318 for IN_c380_00015_sub.dat
            trend = 0
        # print( 'trend=', trend )

        return trend, proportion
