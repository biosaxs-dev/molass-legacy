# coding: utf-8
"""
    LinearityScore.py

    Objects to facilitate garbage collection

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""
import numpy        as np
from scipy          import stats

DEBUG_PLOT  = False
if DEBUG_PLOT:
    import matplotlib.pyplot    as plt

FACTOR_WEIGHT       = 20

def linearity_score100( values ):
    return [ float( '%.3g' % ( np.power(abs(x),4)*FACTOR_WEIGHT) ) for x in values ]

def stderror_score100( values ):
    return [ float( '%.3g' % ( np.exp(-x*20)*FACTOR_WEIGHT ) ) for x in values ]

class LinearityScore:
    def __init__( self, x, y ): 
        base_angle = np.pi/4

        try:
            self.slope = slope = stats.linregress( x, y )[0]
            eval_angle = np.arctan( slope )

            # rotate to the base_angle so that slopes near zero won't have too low linearities
            angle = base_angle - eval_angle
            c = np.cos( angle )
            s = np.sin( angle )
            x_ = c*x - s*y
            y_ = s*x + c*y
            self.r_value, _, self.stderr = stats.linregress( x_, y_ )[2:5]

            if DEBUG_PLOT:
                print( slope, angle )
                plt.plot( x, y )
                plt.plot( x_, y_ )
                plt.show()
        except:
            import logging
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            logger = logging.getLogger( __name__ )
            logger.warning( 'caused an error in computing the linearity score: ' + etb.last_line() )
            self.slope      = np.nan
            self.r_value    = 0
            # self.stderr     = np.inf   # would result in an error in ExcelCOM.py
            self.stderr     = 1e10

    def get_params( self ):
        return self.slope, self.r_value, self.stderr
