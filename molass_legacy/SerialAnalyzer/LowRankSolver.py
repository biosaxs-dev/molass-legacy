# coding: utf-8
"""
    LowRankSolver.py

    Copyright (c) 2017-2018, Masatsuyo Takahashi, KEK-PF
"""
import numpy                as np
import copy


class LowRankSolver:
    def __init__( self, data_matrix ):
        self.M  = data_matrix

    def solve_base_plane( self, cols ):
        rank = len(cols)

        BP = self.generate_baseplane( A, B, C )
        M_ = self.M - BP
        P = col_curve_matrix - BP[:,cols]
        Pinv = np.linalg.pinv( P )
        C = np.dot( Pinv, M_ )

    def generate_baseplane( self, A, B, C ):
        pass
