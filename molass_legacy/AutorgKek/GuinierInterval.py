# coding: utf-8
"""
    GuinierInterval.py

    Copyright (c) 2016-2017, Masatsuyo Takahashi, KEK-PF
"""
import sys
import numpy                as np
from bisect                 import bisect_right
from scipy.interpolate      import UnivariateSpline
from lmfit                  import minimize, Parameters, Parameter, report_fit
from SmoothSpline           import SmoothSpline

DEBUG = False

WeightsArray = [
    np.array( [ 100, 10,  1,  1 ] ),
    ]

MIN_SIZE = 1e-6

class InvervalEvaluationModel:
    def __init__( self, Rg, degree, x, smoother, weights, qmax_limit ):
        self.Rg     = Rg
        self.degree = degree
        self.x      = x
        self.dx     = x[1] - x[0]
        # self.qrg_q_limit    = np.sqrt( 3 * degree / 2 )
        self.qmax_limit = qmax_limit
        if DEBUG: print( 'InvervalEvaluationModel: qmax_limit=', qmax_limit )
        self.smoother = smoother
        self.weights = weights

    def __call__( self, params ):
        min_q   = params[ 'min_q' ].value
        max_q   = params[ 'max_q' ].value
        # max_q   = self.qmax_limit

        size_ = max_q - min_q

        Rg_array = []
        Rg_diff_array = []
        min_qq = min_q**2

        if size_ > MIN_SIZE:
            min_2   = min_q + self.dx*2     # *2 seems to make it a little better
            min_22  = min_2**2

            if self.smoother.is_guinier:
                y1  = self.smoother( min_qq )
                y2  = self.smoother( min_22 )
            else:
                y1 = np.log( self.smoother( min_q ) )
                y2  = np.log( self.smoother( min_2 ) )
            a1   = ( y2 - y1 ) / ( min_22 - min_qq )
            a_array = [a1]

            xmax_ = min( max_q, self.qmax_limit )
            if xmax_ > min_2:
                max_qq = xmax_**2
                if self.smoother.is_guinier:
                    y3  = self.smoother( max_qq )
                else:
                    y3 = np.log( self.smoother( max_q ) )
                a2  = ( y3 - y2 ) / ( max_qq - min_22 )
                a_array.append( a2 )
                # print( 'a_array=', a_array )
            else:
                a_array.append( a1 )

            for a in a_array:
                if a < 0:
                    Rg = np.sqrt( -a*3 )
                    Rg_diff = abs( Rg - self.Rg )**2
                    """
                    ex = max_q * Rg - self.qrg_q_limit
                    if ex > 0:
                        Rg_diff += ex
                    """
                else:
                    # raise Exception( 'a=%g >= 0' % a )
                    Rg = None
                    Rg_diff = 9999 + size_**2

                Rg_array.append( Rg )
                Rg_diff_array.append( Rg_diff )
        else:
            Rg = np.nan
            Rg_diff = 999 + size_**2
            Rg_array = [ Rg, Rg ]
            Rg_diff_array = [ Rg_diff, Rg_diff ]

        size_factor = 1e-2/size_**2 if size_ > MIN_SIZE else 1e-2/MIN_SIZE**2 + size_**2

        # start_factor = min_q ** 2 / max( 0.01, Rg_diff ) / size_factor * 1e5
        # start_factor = min_q / size_factor * 10
        # start_factor = min_q ** 2 / size_factor * 10000
        # start_factor = min_q ** 4 * size_factor * 1e5
        # start_factor = min_q ** 2 * size_factor**2 * 1e3
        # start_factor = min_q**2 * np.sqrt( size_factor ) * 10000
        # start_factor = min_q ** 4 * 10000
        start_factor = min_qq * 1e4

        # Rg_diff_factor = np.sqrt( np.sum( ( self.weights[0:2] * Rg_diff_array)**2 ) )
        # ret = np.array( [ Rg_diff_factor, self.weights[2]*size_factor, self.weights[3]*start_factor ] )
        ret = self.weights * np.array( [ Rg_diff_array[0], Rg_diff_array[1], size_factor, start_factor ] )

        if DEBUG:
            Rg_str = 'None' if Rg is None else '%0.3g' % Rg
            """
            print( '%.2g' % self.degree, '(%0.3g, %0.3g)' % (min_q, max_q),
                    ', Rg=%s' % Rg_str, ', Rg_diff=(%0.3g, %0.3g)' % tuple( Rg_diff_array ) )
            """
            opt_v = np.inner( ret, ret )
            # print( 'min_q=%.4g => (%0.3g, %0.3g) ret=[ %.3g %.3g %.3g %0.3g ], %.4g' % ( ( min_q, ) + tuple( Rg_diff_array ) + tuple( ret ) + ( opt_v, ) ) )
            print( 'Rg=%.4g, min_q=%.4g => (%.3g, %.3g), %.4g' % ( ( self.Rg, min_q, ) + tuple( Rg_array ) + ( opt_v, ) ) )

        return ret

class GuinierInterval:
    def __init__( self, Q, I, x, y, w, fit, qrg_limits ):
        self.Q      = Q
        self.I      = I
        self.x      = x
        self.y      = y
        self.w      = w
        self.fit    = fit
        self.Rg     = fit.Rg
        self.degree = fit.degree
        self.qmax_limit = min( qrg_limits[1] / fit.Rg, x[-1] )
        self.qmax_index = bisect_right( self.Q, self.qmax_limit )
        try:
            # self.spline = UnivariateSpline( x, y, k=3, s=5e8 )
            # be aware that x and y are non-transformed values,
            # i.e. not x**2 nor log(y)
            # self.spline = UnivariateSpline( x, y, w=w, k=3, s=1e8 )
            # smooth_spline = SmoothSpline( x, y, w=w, sf_log_base=30 )
            self.smoother = SmoothSpline( x, y, w=w, is_guinier=False )
            # self.smoother = SmootherChain( x, y, is_guinier=False )
            # self.smoother = fit.smoother
        except Exception as e:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            if True:
                etb = ExceptionTracebacker()
                print( etb )
                print( 'y=', y )
            # TODO: raise Exception
            self.smoother = None

    def get_optimal_interval( self ):

        # print( 'qmax_limit=', self.qmax_limit )
        min_q_inits = list( map( lambda p: (1 - p) * self.x[0] + p * self.qmax_limit, [ 0.1, 0.2, 0.5 ] ) )
        # print( 'min_q_inits=', min_q_inits )

        for weights in WeightsArray:

            opt_redchi = None
            opt_result = None
            opt_q_init = None

            for min_q_init in min_q_inits:
                if DEBUG: print( 'minimize with min_q_init', min_q_init )

                model   = InvervalEvaluationModel( self.Rg, self.degree, self.x, self.smoother, weights, self.qmax_limit )

                params  = Parameters()

                params.add('min_q',    value= min_q_init,       min=self.x[0], max=self.qmax_limit )
                params.add('max_q',    value= self.qmax_limit,  min=self.x[0], max=self.qmax_limit )

                result = minimize( model, params, args=() )

                if opt_redchi is None or result.redchi < opt_redchi:
                    opt_redchi = result.redchi
                    opt_result = result
                    opt_q_init = min_q_init

            self.min_q = opt_result.params[ 'min_q' ].value

            if DEBUG:
                print( 'minimized with opt_q_init=%.4g' % opt_q_init )
                print( 'optimal min_q=%.4g' % self.min_q )

            self.max_q = result.params[ 'max_q' ].value
            # self.max_q = self.qmax_limit

            if DEBUG: print( self.min_q, self.max_q )            

            f_ = bisect_right( self.Q, self.min_q )
            t_ = bisect_right( self.Q, self.max_q )
            if DEBUG: print( 'get_optimal_interval', self.min_q, self.max_q, f_, t_, self.qmax_index )
            # sys.exit()

            # TODO: better check
            if f_ < self.qmax_index:
                if DEBUG: print( self.min_q, self.max_q, f_, t_ )
                return self.min_q, self.max_q, f_, t_

        raise Exception( 'Guninier interval optimazation failed!' )
