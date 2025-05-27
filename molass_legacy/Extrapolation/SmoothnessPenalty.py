# coding: utf-8
"""
    SmoothnessPenalty.py

    Copyright (c) 2018-2019, SAXS Team, KEK-PF
"""

import numpy as np

try:
    from SmoothnessPenaltyNumba import *
    NUMBA_IS_AVAILABLE  = True
except:
    NUMBA_IS_AVAILABLE  = False

assert NUMBA_IS_AVAILABLE

import molass_legacy.KekLib.DebugPlot as plt

"""
    [NumPy k-th diagonal indices]
        (https://stackoverflow.com/questions/10925671/numpy-k-th-diagonal-indices)
"""
def kth_diag_indices(a, k):
    rowidx, colidx = np.diag_indices_from(a)
    colidx = colidx.copy()  # rowidx and colidx share the same buffer

    if k > 0:
        colidx += k
    else:
        rowidx -= k
    k = np.abs(k)

    return rowidx[:-k], colidx[:-k]


"""
    http://ttic.uchicago.edu/~klivescu/papers/wang_SLT2014.pdf
    RECONSTRUCTION OF ARTICULATORY MEASUREMENTS
    WITH SMOOTHED LOW-RANK MATRIX COMPLETION
"""
def norm_diff_matrix(n):
    dm = np.identity(n)*2
    dm[0,0] = 1
    dm[-1,-1] = 1
    dm[kth_diag_indices(dm,1)] = -1
    dm[kth_diag_indices(dm,-1)] = -1
    return dm

def diff_matrix(n):
    dm = np.identity(n)
    dm[kth_diag_indices(dm,1)] = -1
    return dm

def get_penalty_impl(NDM, P, penalty_weights):
    return [P[:,j].T @ NDM @ P[:,j] * penalty_weights[j%2]   for j in range(P.shape[1]) ]

def get_penalty_diff_impl(NDM, P, penalty_weights):
    n = P.shape[1]
    NDP = np.array( [ ( (NDM * 2) @ P[:,j] * penalty_weights[j%2] ).T  for j in range(n) ] ).T
    return NDP

class SmoothnessPenalty:
    def __init__(self, m, conc_dependence=2, add_const=0):
        self.NDM = norm_diff_matrix(m)
        self.conc_dependence = conc_dependence
        self.add_const = add_const

    def get_reshaped(self, flatP):
        m = self.NDM.shape[0]
        assert len(flatP) % m == 0
        n = len(flatP)//m
        self.col_size = n - self.add_const
        return flatP.reshape((m, n))

    def get_penalties(self, flatP, penalty_weights):
        P = self.get_reshaped(flatP)
        if NUMBA_IS_AVAILABLE:
            return get_penalty_impl_numba(self.conc_dependence, P, penalty_weights, self.col_size)
        else:
            return get_penalty_impl(self.NDM, P, penalty_weights)

    def get_penalties_bq_ignore(self, flatP, penalty_weight):
        P = self.get_reshaped(flatP)
        return get_penalty_bq_ignore_impl_numba(P, penalty_weight, self.col_size)

    def get_penalty_diff(self, flatP, penalty_weights, debug=False):
        P = self.get_reshaped(flatP)
        m = self.NDM.shape[0]
        n = P.shape[1]
        if NUMBA_IS_AVAILABLE:
            NDP = get_penalty_diff_impl_numba(self.conc_dependence, P, penalty_weights, self.col_size)
        else:
            NDP = get_penalty_diff_impl(self.NDM, P, penalty_weights)

        if debug:
            DM = diff_matrix(m)
            DP = np.array( [ ( DM @ P[:,j] ).T  for j in range(n) ] ).T

            plt.plot(P)
            # plt.plot(DP)
            plt.plot(NDP)
            plt.show()

        return NDP.flatten()

    def get_penalty_diff_bq_ignore(self, flatP, penalty_weights, debug=False):
        P = self.get_reshaped(flatP)
        NDP = get_penalty_diff_bq_ignore_impl_numba(P, penalty_weights, self.col_size)
        return NDP.flatten()
