# coding: utf-8
"""
    ExSolver.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from SvdDenoise import get_denoised_data

class ExSolver:
    def __init__(self):
        pass

    def solve(self, M, c, cd=1):
        M_ = get_denoised_data(M, rank=cd)
        if cd == 1:
            C = np.array([c])
        elif cd == 2:
            C = np.array([c, c**2])
        else:
            assert False

        P = M_ @ np.linalg.pinv(C)
        return P[:,0]
