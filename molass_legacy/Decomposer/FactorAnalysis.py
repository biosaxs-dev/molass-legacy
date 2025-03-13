# coding: utf-8
"""
    FactorAnalysis.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import numpy                as np
import matplotlib.pyplot    as plt
from sklearn.decomposition  import PCA, FastICA, FactorAnalysis as SKFA

DEBUG   = False

class FactorAnalysis:
    def __init__( self, a_curve, array ):

        num_factors = len( a_curve.peak_info )

        # fa = FastICA( n_components=num_factors, max_iter=5000 )
        fa = PCA( n_components=num_factors )
        X = fa.fit_transform( array )
        for i in range(num_factors):
            plt.plot(X[:,i])

        plt.show()
