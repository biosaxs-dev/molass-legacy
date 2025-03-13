# coding: utf-8
"""
    SAngler_WLS.py

    Copyright (c) 2016, Masatsuyo Takahashi, KEK-PF
"""
import numpy        as np

class SAngler_WLS_Result:
    def __init__( self, A, B, sigmaA2, sigmaB2 ):
        self.params = np.array( [ A, B ] )
        self.std_errors = np.sqrt( [ sigmaA2, sigmaB2 ] )
        self.sigmaA2    = sigmaA2
        self.sigmaB2    = sigmaB2

    def cov_params( self ):
        # this is diagonal only for proof purpose
        return np.diag( [ self.sigmaA2, self.sigmaB2 ] )

class SAngler_WLS:
    def __init__( self, y, x, w ):
        sum_w   = np.sum( w )
        wx_     = np.inner( w, x )
        x2_     = x**2
        wx2_    = np.inner( w, x2_ )
        delta   = sum_w*wx2_ - wx_**2
        wy_     = np.inner( w, y )
        wxy_    = np.inner( w, x*y )
        d_      = 1/delta
        self.A  = d_*( wx2_*wy_ - wx_*wxy_ )
        self.B  = d_*( sum_w*wxy_ - wx_*wy_ )

        N = y.shape[0]
        e = y - ( self.A + self.B * x )
        S = np.inner( w, e**2 )
        sigma2 = S / ( N - 2 )

        self.sigmaA2 = d_ * wx2_ * sigma2
        self.sigmaB2 = d_ * sum_w * sigma2

    def fit( self ):

        return SAngler_WLS_Result( self.A, self.B, self.sigmaA2, self.sigmaB2 )
