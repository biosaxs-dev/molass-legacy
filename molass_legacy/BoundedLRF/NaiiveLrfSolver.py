"""
    NaiiveLrfSolver.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt
from SvdDenoise import get_denoised_data

class NaiiveLrfSolver:
    def __init__(self, qv, D, E, i=None, denoise=True):
        if i is None:
            from bisect import bisect_right
            i = bisect_right(qv, 0.02)

        self.qv = qv
        self.i = i
        if denoise:
            D_ = get_denoised_data(D, rank=2)
        else:
            D_ = D
        self.D_ = D_
        self.E = E
        cy = np.average(D_[i-5:i+6,:], axis=0)
        self.j = np.argmax(cy)
        self.Cinit = np.array([cy, cy**2])

    def solve(self, debug=False):
        qv = self.qv
        i = self.i
        j = self.j
        C = self.Cinit
        D_ = self.D_
        Cinv = np.linalg.pinv(C)
        P = D_ @ Cinv
        return P, C
