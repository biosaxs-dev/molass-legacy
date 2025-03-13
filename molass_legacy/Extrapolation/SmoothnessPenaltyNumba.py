# coding: utf-8
"""
    SmoothnessPenaltyNumba.py

    Copyright (c) 2018-2020, SAXS Team, KEK-PF
"""

import numpy as np
try:
    # for numba 1.49 or later
    from numba.core.decorators import jit
except:
    from numba.decorators import jit

@jit(nopython=True)
def get_penalty_impl_numba(conc_dependence, P, penalty_weights, col_size):
    penalties = np.zeros(col_size)
    for j in range(col_size):
        pwi = j%conc_dependence
        if pwi > 1:
            continue

        x = P[:,j]
        penalty = 0
        for k in range(len(x)-1):
            penalty += (x[k] - x[k+1])**2
        penalties[j] = penalty * penalty_weights[pwi]
    return penalties

@jit(nopython=True)
def get_penalty_diff_impl_numba(conc_dependence, P, penalty_weights, col_size):
    NDP = np.zeros(P.shape)
    for j in range(col_size):
        pwi = j%conc_dependence
        if pwi > 1:
            continue

        scale = penalty_weights[pwi]
        x = P[:,j]
        NDP[ 0,j]   = scale*2*(x[0] - x[1])
        for k in range(1, len(x)-1):
            NDP[k,j]    = scale*2*(2*x[k] - x[k-1] - x[k+1])
        NDP[-1,j]   = -scale*2*(x[-2] - x[-1])

    return NDP

@jit(nopython=True)
def get_penalty_bq_ignore_impl_numba(P, penalty_weight, col_size):
    penalties = np.zeros(col_size)
    for j in range(col_size):
        x = P[:,j]
        penalty = 0
        for k in range(len(x)):
            penalty += (x[k] - x[k+1])**2
        penalties[j] = penalty * penalty_weight
    return penalties

@jit(nopython=True)
def get_penalty_diff_bq_ignore_impl_numba(P, weight, col_size):
    NDP = np.zeros(P.shape)
    for j in range(col_size):
        x = P[:,j]
        NDP[ 0,j]   = weight*2*(x[0] - x[1])
        for k in range(1, len(x)-1):
            NDP[k,j]    = weight*2*(2*x[k] - x[k-1] - x[k+1])
        NDP[-1,j]   = -weight*2*(x[-2] - x[-1])

    return NDP
