# coding: utf-8
"""
    ElutionStainModel..py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""

class ElutionStainModel:
    def __init__( self, qvector, num_eluitions, start, stop ):
        assert num_eluitions > num_params

        self.qvec   = qvector
        self.j      = np.arange( num_eluitions )
        self.slice_ = slice( start, stop )
        self.num_params = stop - start

    def learn( self, baselines ):
        pass
