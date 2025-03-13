# coding: utf-8
"""

    LinRegSmoother.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""
import numpy        as np
from scipy          import stats

REGRESS_SIZE    = 9
REGHALF_SIZE    = REGRESS_SIZE // 2

class Smoother:
    def __init__( self, x, y ):
        sy_list = []
        start   = 0
        stop    = len(x) - REGRESS_SIZE
        for i in range( start, stop ):
            slice_ = slice( i, i + REGRESS_SIZE )
            x_ = x[slice_]
            y_ = y[slice_]
            slope, intercept, r_value, p_value, std_err = stats.linregress( x_, y_ )
            if i == start:
                for j in range( 0, REGHALF_SIZE ):
                    sy = slope * x_[j] + intercept
                    sy_list.append( sy )

            sy = slope * x_[ REGHALF_SIZE ] + intercept
            sy_list.append( sy )

            if i == stop-1:
                for j in range( REGHALF_SIZE, REGRESS_SIZE ):
                    sy = slope * x_[j] + intercept
                    sy_list.append( sy )

        self.sy = np.array( sy_list )

    def __call__( self, x ):
        return self.predict( x )

    def predict( self, x ):
        return self.sy
