# coding: utf-8
"""

    DataModelsAgg.py

    Copyright (c) 2016, Masatsuyo Takahashi, KEK-PF
"""

import numpy                as np


def _GuinierPorodAgg( C, Rg, d, x ):

    y = np.zeros( (len(x),) )
    for i in range(2):
        C_  = C[i]
        Rg_ = Rg[i]
        d_  = d[i]

        Q1  = 1/Rg_ * np.sqrt( 3*d_/2 )
        D   = C_ * np.exp( -d_/2 ) * np.power( 3*d_/2, d_/2 ) / np.power( Rg_, d_ )
        y_  = np.hstack( [  C_ * np.exp( - x[ x < Q1 ]**2 * Rg_**2 / 3 ),
                            D  / np.power( x[ x >= Q1 ], d_ )
                         ] )
        y += y_
    return y

class GuinierPorodAgg:
    def __init__( self, C, Rg, d ):
        self.C  = C
        self.Rg = Rg
        self.d  = d

    def __call__( self, x, sigma=0 ):
        # TODO: sigma=0 to be deleted
        return _GuinierPorodAgg( self.C, self.Rg, self.d, x )

