# coding: utf-8
"""

    ScatteringBasespline.py

    scattering baseline solver

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF

"""
import numpy            as np
from scipy                  import stats
from scipy.interpolate      import UnivariateSpline
from bisect                 import bisect_right
from molass_legacy.Baseline.ScatteringBaseline     import ScatteringBaseline

DEBUG_PLOT              = False
PERCENTILE_SECOND       = 25
PERCENTILE_FINAL        = 10

VERY_SMALL_ANGLE_LIMIT  = 1.0
NUM_REGRESSION_POINTS   = 20

def get_spline_smoothing_for_base( y, max_y ):
    x   = np.arange(NUM_REGRESSION_POINTS)
    slope, intercept, r_value, p_value, std_err = stats.linregress( x, y[0:NUM_REGRESSION_POINTS] )
    s   = std_err**2 * 1e3 * len( y )
    # print( 'std_err=', std_err )
    # print( 'max_y=', max_y, 's=%.1e' % s )
    return s

class ScatteringBasespline:
    def __init__( self, y, q = 0.02, q_min=0.004 ):
        self.q  = q
        self.q_min = q_min
        self.y  = y
        self.x = np.arange( len(self.y) )
        self.line   = ScatteringBaseline( y )
        # np.savetxt( 'curve.csv', y, delimiter=',' )

    def solve( self, p_final=PERCENTILE_FINAL ):
        A, B = self.line.solve( p_final=p_final )
        # A, B = self.line.solve( p_final=PERCENTILE_FINAL )
        self.A  = A
        self.B  = B
        self.baseline = self.A*self.x + self.B

        if self.q < VERY_SMALL_ANGLE_LIMIT:
            max_y   = np.max( self.y )
            smoothing   = get_spline_smoothing_for_base( self.y, max_y )
            y_  = self.y - self.baseline
            ypps   = np.percentile( y_, [ PERCENTILE_SECOND, p_final ] )
            self.lowp0 = y_ < ypps[0]
            self.curve  = UnivariateSpline( self.x[self.lowp0], self.y[self.lowp0], s=smoothing )
            # TODO: shift using p_final (the following is not tested and probably not correct )
            self.final_shift = ypps[0] - ypps[1]
            print( 'self.final_shift=', self.final_shift )

        return 0, 0, 0

    def get_baseline( self, x ):
        baseline    = self.baseline
        if self.q < VERY_SMALL_ANGLE_LIMIT:
            # basecurve = self.curve( x ) + self.curve_base
            basecurve = self.curve( x ) + self.final_shift
            # w = VERY_SMALL_ANGLE_LIMIT - self.q_min
            # composite = basecurve * ( VERY_SMALL_ANGLE_LIMIT - self.q )/w + baseline * ( self.q - self.q_min )/w
            if DEBUG_PLOT:
                from DebugCanvas    import DebugCanvas
                def debug_plot( fig ):
                    ax = fig.add_subplot( 111 )
                    ax.set_title( 'q=' + str(self.q) )
                    ax.plot( self.y, label='data' )
                    ax.plot( baseline, label='baseline' )
                    ax.plot( basecurve, label='basecurve' )
                    ax.plot( self.x[self.lowp0], self.y[self.lowp0], 'yo' )
                    knots = self.curve.get_knots()
                    ax.plot( knots, self.curve(knots), 'co' )
                    # ax.plot( composite, label='composite' )
                    # ax.plot( self.lowp1, self.y[self.lowp1], 'o', color='red' )
                    ax.legend()
                    fig.tight_layout()
                dc = DebugCanvas( "Debug", debug_plot, parent=None, figsize=(8,6) )
                dc.show( cancelable=True )
            return basecurve
        else:
            return baseline
