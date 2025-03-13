# coding: utf-8
"""
    HeadAnomalies.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""

import numpy        as np
from scipy          import stats

HEAD_SIZE               = 10
NEXT_SIZE               = HEAD_SIZE * 3
ACCEPTABLE_DEVIATION    = 1.0
NUM_CONSECUTIVE_ACCEPTS = 2

DEBUG = False

class HeadAnomalies:
    def __init__( self, y, X, Y ):
        self.X  = X
        self.Y  = Y

    def remove_head_anomalies( self ):
        try:
            self.remove_head_anomalies_impl()
        except:
            # this case will be caused e.g. when X, X are empty
            self.start_point = 0

    def remove_head_anomalies_impl( self ):
        start_candidate = np.argmax( self.Y )
        stop_candidate  = len(self.Y)//2
        if DEBUG: print( 'start_candidate=', start_candidate, len(self.Y) )

        last_accept = None
        num_accepts = 0
        for i in range( start_candidate, stop_candidate ):
            next_start  = i + HEAD_SIZE
            head_slope  = self.evaluate_slope( i, next_start )
            next_slope  = self.evaluate_slope( next_start, next_start + NEXT_SIZE )
            deviation   = abs( head_slope/next_slope - 1 )
            if DEBUG: print( i, 'deviation=', deviation )
            if deviation <= ACCEPTABLE_DEVIATION:
                if last_accept is not None and i == last_accept + 1:
                    num_accepts += 1
                    if num_accepts == NUM_CONSECUTIVE_ACCEPTS:
                        start_candidate = i - num_accepts + 1
                        break
                else:
                    num_accepts = 1
                last_accept = i

        self.start_point = start_candidate

    def evaluate_slope( self, start, stop ):
        x, y = self.X[start:stop], self.Y[start:stop]
        slope, intercept, r_value, p_value, std_err = stats.linregress( x, y )
        return slope
