# coding: utf-8
"""
    OurStatsModels.py

    Copyright (c) 2016-2018, Masatsuyo Takahashi, KEK-PF
"""
import numpy            as np

def add_constant( x ):
    return np.vstack( [ np.ones( ( len(x), ) ), x ] ).T

class Result:
    def __init__( self, **entries ): 
        self.__dict__.update(entries)

    def cov_params( self ):
        return self.cov_params_

class WLS:
    def __init__( self, y, X, weights=None ):
        if type(y) != np.ndarray:
            y = np.array(y)
        if weights is None:
            weights = np.ones( ( len(y), ) )
        else:
            if type(weights) != np.ndarray:
                weights = np.array(weights)

        assert( len( y.shape ) == 1 )
        assert( len( X.shape ) == 2 )
        assert( len( weights.shape ) == 1 )
        assert( y.shape[0] == X.shape[0] )
        assert( y.shape[0] == weights.shape[0] )

        self.y  = y
        self.A          = X
        self.N  = N = y.shape[0]
        shape_ = ( N, 1 )
        # print( self.A )
        self.y_         = np.reshape( y, shape_ )
        self.w          = weights
        self.W          = np.diag( weights )

    def fit( self ):
        At      = np.transpose( self.A )
        AtW     = np.dot( At, self.W )
        self.XtWX_inv = AtWA_inv = np.linalg.inv( np.dot( AtW, self.A ) )
        self.XtWX_inv_XtW = AtWX_inv_AtW = np.dot( AtWA_inv, AtW )
        params  = np.dot( AtWX_inv_AtW, self.y_ )
        # print( params )
        """
            As for the covariance matrix of params, see
            Linear least squares (mathematics)
            https://en.wikipedia.org/wiki/Linear_least_squares_%28mathematics%29#Weighted_linear_least_squares
        """
        e       = ( self.y_ - np.dot( self.A, params ) ).squeeze()
        # print( 'e_=', e[0:5] )
        # print( 'e =', 1/np.sqrt( self.w[0:5] ) )
        ee = e**2
        S       = np.inner( self.w,  ee)
        sigma2  = S / ( self.N - 2 )
        # print( 'S=', S, ', N=', self.N, ', sigma2=', sigma2 )
        cov_params = AtWA_inv * sigma2

        ssr = np.sum(ee)
        # TODO: this ssr value is differenct from that of statsmodels.api

        return Result(
                params=np.reshape( params, ( params.shape[0], ) ),
                cov_params_=cov_params,
                ssr=ssr,
                )

class OLS(WLS):
    def __init__(self, y, X):
        WLS.__init__(self, y, X)
