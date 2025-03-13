# coding: utf-8
"""
    ConvolutionDemo.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.gridspec  as gridspec

NEGATIVE_INF    = -20
POSITIVE_INF    =  20

class Convolution:
    def __init__( self, f, g, lower_limit=NEGATIVE_INF, upper_limit=POSITIVE_INF, zero_neg=False ):
        self.f  = f
        self.g  = g
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.zero_neg = zero_neg

    def eval_scaler( self, t ):
        h = lambda x: self.f( x ) * self.g( t - x )
        if self.zero_neg:
            return quad( h, self.lower_limit, t )[0]
        else:
            return quad( h, self.lower_limit, self.upper_limit )[0]

    def __call__( self, t ):
        if np.isscalar( t ):
            return self.eval_scaler( t )
        else:
            return np.array( [ self.eval_scaler(x) for x in t ] )

class Gaussian:
    def __init__( self, mu, sigma ):
        self.mu     = mu
        self.sigma  = sigma
        self.c      = 1 / np.sqrt( 2 * np.pi * sigma**2 )

    def __call__( self, x ):
        return self.c * np.exp( - (x - self.mu)**2/( 2*self.sigma**2 ) )

class Exponential:
    def __init__( self, k):
        self.k  = k

    def __call__( self, x ):
        if np.isscalar( x ):
            return 0 if x < 0 else self.k * np.exp( -self.k * x )
        else:
            x_pos   = x >= 0
            v_pos   = self.k * np.exp( -self.k * x[x_pos] )
            return np.hstack( [ np.zeros( len(x) - len(v_pos) ), v_pos ] )

class EmgConvolutionDemo:
    def __init__( self ):
        k   = 1
        m   = 3
        s   = 0.5
        n   = Gaussian( m, s )
        e   = Exponential( k )
        h   = Convolution( n, e, zero_neg=True )

        x   = np.linspace( -3, 10, 1000 )

        fig = plt.figure( figsize=(12, 9) )

        ax1 = fig.add_subplot( 111 )

        ax1.set_title( "EMG - Convolution of Gaussian and Exponential" )

        ax1.plot( x, n(x), label='G(%g, %g)' % (m, s) )
        ax1.plot( x, e(x), label='E(%g)' % (k) )
        ax1.plot( x, h(x), label='EMG - Convolution of G(%g, %g) and E(%g)' % (m, s, k) )

        ax1.legend()
        fig.tight_layout()
        plt.show()
