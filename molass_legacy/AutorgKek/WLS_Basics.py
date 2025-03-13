# coding: utf-8
"""
    WLS_Basics.py

    Copyright (c) 2016-2018, Masatsuyo Takahashi, KEK-PF
"""
import numpy            as np

class Result:
    def __init__( self, **entries ): 
        self.__dict__.update(entries)

    def cov_params( self ):
        return self.cov_params_

class OLS:
    def __init__( self, y, x ):
        assert( len( y.shape ) == 1 )
        assert( len( x.shape ) == 1 )
        assert( y.shape[0] == x.shape[0] )

        self.N = N = y.shape[0]
        shape_ = ( N, 1 )
        self.A          = np.hstack( [ np.ones( shape_ ), np.reshape( x, shape_ ) ] )   # add constant
        # print( self.A )
        self.b          = np.reshape( y, shape_ )

    def fit( self ):
        At      = np.transpose( self.A )
        AtA_inv = np.linalg.inv( np.dot( At, self.A ) )
        params  = np.dot( np.dot( AtA_inv, At ), self.b )
        # print( params )
        e       = self.b - np.dot( self.A, params )
        S       = np.sum( e**2 )
        sigma2  = S / ( self.N - 2 )
        cov_params = AtA_inv * sigma2
        # print( cov_params )
        return Result(
                params=np.reshape( params, ( params.shape[0], ) ),
                cov_params_=cov_params,
                )

class WLS:
    def __init__( self, y, x, w ):
        assert( len( y.shape ) == 1 )
        assert( len( x.shape ) == 1 )
        assert( len( w.shape ) == 1 )
        assert( y.shape[0] == x.shape[0] )
        assert( y.shape[0] == w.shape[0] )

        self.x  = x
        self.y  = y
        self.N  = N = y.shape[0]
        shape_ = ( N, 1 )
        self.A          = np.hstack( [ np.ones( shape_ ), np.reshape( x, shape_ ) ] )   # add constant
        # print( self.A )
        self.b          = np.reshape( y, shape_ )
        # w = w / np.sum( w )
        self.w          = w
        self.W          = np.diag( w )

    def fit( self ):
        At      = np.transpose( self.A )
        AtW     = np.dot( At, self.W )
        AtWA_inv = np.linalg.inv( np.dot( AtW, self.A ) )
        AtWb    = np.dot( AtW, self.b )
        params  = np.dot( AtWA_inv, AtWb )
        # print( params )
        """
            As for the covariance matrix of params, see
            Linear least squares (mathematics)
            https://en.wikipedia.org/wiki/Linear_least_squares_%28mathematics%29#Weighted_linear_least_squares
        """
        e       = ( self.b - np.dot( self.A, params ) ).squeeze()
        ee = e**2
        print( 'e_=', e[0:5] )
        print( 'e =', 1/np.sqrt( self.w[0:5] ) )
        S       = np.inner( self.w, ee )
        sigma2  = S / ( self.N - 2 )
        print( 'S=', S, ', N=', self.N, ', sigma2=', sigma2 )
        cov_params = AtWA_inv * sigma2

        ssr = np.sum(ee)
        # TODO: this ssr value is differenct from that of statsmodels.api

        return Result(
                params=np.reshape( params, ( params.shape[0], ) ),
                cov_params_=cov_params,
                ssr=ssr,
                )
