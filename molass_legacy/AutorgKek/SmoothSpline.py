# coding: utf-8
"""
    SmoothSpline.py

    Copyright (c) 2016, Masatsuyo Takahashi, KEK-PF
"""
import numpy                as np
from scipy.interpolate      import UnivariateSpline

DEBUG = False

class SmoothSpline:
    def __init__( self, x, y, w=None, curvature=False, sf_log_base=36, sf=None, is_guinier=True, with_smoother=False ):

        if DEBUG: print( 'SmoothSpline: x.shape[0]=', x.shape[0], ', x[0], x[-1]=', x[0], x[-1] )

        self.is_guinier = is_guinier

        yvar = np.var( y )
        if DEBUG: print( 'yvar=',  yvar, ', y.shape[0]=', y.shape[0] )
        if yvar == 0:
            yvar = 1e2
            if DEBUG:
                print( 'SmoothSpline: yvar(==0) is replaced to ', yvar )

        if sf is None:
            logvar = np.log( yvar * y.shape[0] )
            sf = np.exp( sf_log_base + logvar )
            if DEBUG: print( 'logvar=%.3g' % logvar, ', sf=%.3e' % sf )

        if with_smoother:
            from GaussianProcessDep import Smoother
            smoother = Smoother( x, y  )
            y_ = smoother( x )
        else:
            y_ = y

        # TODO: validity of min( 3, len(y)-1 )
        spline  = UnivariateSpline( x, y_, w=w, k=min( 3, len(y)-1 ), s=sf )
        if DEBUG: print( 'UnivariateSpline ok, curvature=', curvature )

        self.spline = spline

        if not curvature: return
        if DEBUG: print( 'derivatives' )

        yˈ  = spline.derivative(1)( x )
        yˈˈ = spline.derivative(2)( x )
        cy = yˈˈ / np.power( 1 + yˈ**2, 3/2 )

        self.yˈ  = yˈ
        self.yˈˈ = yˈˈ
        self.cy  = cy
        self.cv_array   = np.percentile( cy, [ 5, 20, 50, 80, 95 ] )
        cvmax = np.max( np.abs( self.cv_array[1:-1] ) )

        if DEBUG: print( 'cvmax=', cvmax, ', cv_array=', self.cv_array  )
        if cvmax > 100:
            # cvmax == 6.99 for IpI/exampleA_8mg.dat
            raise Exception( 'Unexpected cvmax=%g' % cvmax )

    def __call__( self, x ):
        return self.spline( x )
