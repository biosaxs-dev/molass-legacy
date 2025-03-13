# coding: utf-8
"""
    WeightedLRF.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import copy
import numpy as np

SMALL_POSITIVE_VALUE = 1e-10

def compute_reciprocally_weighted_matrices(conc_dependence, C, D):
    assert C.shape[0] <= 2 and C.shape[0] == conc_dependence
    assert C.shape[1] == D.shape[1]

    row_ones = np.ones(C.shape[1])
    n = C.shape[0]//conc_dependence
    c = copy.deepcopy(C[0,:])
    c[c < SMALL_POSITIVE_VALUE] = SMALL_POSITIVE_VALUE

    c_rows = [row_ones]
    if conc_dependence > 1:
        c_rows.append(c)
    C_ = np.array(c_rows)

    cinv = 1/c
    d_rows = []
    for i in range(D.shape[0]):
        d_rows.append(D[i,:]*cinv)

    D_ = np.array(d_rows)
    return C_, D_
