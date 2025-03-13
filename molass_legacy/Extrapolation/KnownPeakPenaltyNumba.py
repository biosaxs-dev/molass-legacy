# coding: utf-8
"""
    KnownPeakPenaltyNumba.py

    Copyright (c) 2019-2020, SAXS Team, KEK-PF
"""

import numpy as np
try:
    # for numba 1.49 or later
    from numba.core.decorators import jit
except:
    from numba.decorators import jit

@jit(nopython=True)
def compute_known_error_impl(conc_dependence, known_peak_info, P):
    error = 0
    for k, data in enumerate(known_peak_info):
        if data is None:
            continue
        n = k*conc_dependence
        error += np.sum((P[:,n] - data[:,1])**2)
    return error

@jit(nopython=True)
def compute_known_error_grad_impl(conc_dependence, known_peak_info, P):
    error = 0
    for k, data in enumerate(known_peak_info):
        if data is None:
            continue
        n = k*conc_dependence
        error += np.sum(2*(P[:,n] - data[:,1]))
    return error
