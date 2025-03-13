# coding: utf-8
"""
    KnownPeakPenalty.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""

import numpy as np

def compute_known_error_impl(conc_dependence, known_peak_info, P):
    error = 0
    for k, data in enumerate(known_peak_info):
        if data is None:
            continue
        n = k*conc_dependence
        error += np.sum((P[:,n] - data[:,1])**2)
    return error

def compute_known_error_grad_impl(conc_dependence, known_peak_info, P):
    error = 0
    for k, data in enumerate(known_peak_info):
        if data is None:
            continue
        n = k*conc_dependence
        error += np.sum(2*(P[:,n] - data[:,1]))
    return error
