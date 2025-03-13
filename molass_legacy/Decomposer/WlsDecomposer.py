# coding: utf-8
"""
    WlsDecomposer.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import numpy                as np
import statsmodels.api      as sm
import matplotlib.pyplot    as plt

DEBUG   = False

class WlsDecomposer:
    def __init__( self, ecurve, data, drift=False, in_folder=None ):
        self.data   = data

        peaks_ = [ int(info[1]) for info in ecurve.peak_info ]
        if drift:
            peaks_ += [-1]

        print( peaks_ )
        A = data[:,peaks_]
        print( A.shape )

        params_list = []
        for j in range( self.data.shape[1] ):
            params = self.solve_an_elution( A, j )
            params_list.append( params )

        solution = np.array( params_list ).T
        for k in range( solution.shape[0] ):
            plt.plot( solution[k,:], label='component' + str(k) )

        plt.legend()
        plt.tight_layout()
        plt.show()

        if drift:
            A[:, -1] *= 10

        Apinv = np.linalg.pinv( A )
        print( Apinv.shape )
        C   = np.dot( Apinv, data )

        if in_folder is not None:
            plt.title( in_folder )

        num_components   = len( peaks_ )
        for k in range( num_components ):
            plt.plot( C[k,:], label='component' + str(k) )

        plt.legend()
        plt.tight_layout()
        plt.show()

    def solve_an_elution( self, A, j ):
        y   = self.data[:,j]
        x   = A
        X   = sm.add_constant(x)
        mod = sm.OLS( y, X )
        res = mod.fit()
        # print( 'res.params=', res.params )
        return res.params
