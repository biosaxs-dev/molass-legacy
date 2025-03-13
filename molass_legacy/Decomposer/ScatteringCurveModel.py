# coding: utf-8
"""
    ScatteringCurveModel.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import numpy                as np
import opticspy
import matplotlib.pyplot    as plt

class RapidDebye:
    def __init__( self ):
        pass

    def fit( self, x, y ):
        self.b  = 1
        self.G  = y / ( np.exp( -self.b*q**2 )**2 )

    def eval( self, x ):
        return np.exp( -self.b*x**2 )**2 * self.G
