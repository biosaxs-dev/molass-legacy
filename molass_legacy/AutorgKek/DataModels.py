# coding: utf-8
"""

    DataModels.py

    Copyright (c) 2016, Masatsuyo Takahashi, KEK-PF

"""

import numpy                as np


def _GuinierPorod( G, Rg, d, x ):
    q1  = 1/Rg * np.sqrt( 3*d/2 )
    D   = G * np.exp( -d/2 ) * np.power( 3*d/2, d/2 ) / np.power( Rg, d )
    y   = np.hstack( [  G * np.exp( - x[ x < q1 ]**2 * Rg**2 / 3 ),
                        D / np.power( x[ x >= q1 ], d )
                     ] )
    return y

class GuinierPorod:
    def __init__( self, G, Rg, d ):
        self.G  = G
        self.Rg = Rg
        self.d  = d

    def __call__( self, x, sigma=0 ):
        # TODO: sigma=0 to be deleted
        return _GuinierPorod( self.G, self.Rg, self.d, x )

class GuinierPorodLmfit:
    def __init__( self, def_func=None ):
        if def_func is None:
            self.def_func   = _GuinierPorod
        else:
            self.def_func   = def_func

    def __call__( self, params, x, y, w  ):
        G   = params['G'].value
        Rg  = params['Rg'].value
        d   = params['d'].value
        y_  = self.def_func( G, Rg, d, x )
        return w * ( y_ - y )
        # return y_ - y

def _GuinierPorodGeneral( G, Rg, d, s, x ):
    if d > s:
        q1  = 1/Rg * np.sqrt( (d - s)*(3 - s)/2 )
        D   = G  / np.power( Rg, d - s ) * np.exp( -(d - s)/2 ) * np.power( (d - s)*(3 - s)/2, (d - s)/2 )
        x_  = x[ x < q1 ]
        y   = np.hstack( [  G /  np.power( x_, s ) * np.exp( - x_**2 * Rg**2 / (3 - s) ),
                            D / np.power( x[ x >= q1 ], d )
                         ] )
    else:
        y   = np.ones( len(x) ) * 9999.9999
    return y

class GuinierPorodGeneral:
    def __init__( self, G, Rg, d, s ):
        self.G  = G
        self.Rg = Rg
        self.d  = d
        self.s  = s

    def __call__( self, x ):
        return _GuinierPorodGeneral( self.G, self.Rg, self.d, self.s, x )

class GuinierPorodGeneralLmfit:
    def __init__( self, def_func=None ):
        if def_func is None:
            self.def_func   = _GuinierPorodGeneral
        else:
            self.def_func   = def_func

    def __call__( self, params, x, y, w  ):
        G   = params['G'].value
        Rg  = params['Rg'].value
        d   = params['d'].value
        s   = params['s'].value
        y_  = self.def_func( G, Rg, d, s, x )
        return w * ( y_ - y )
