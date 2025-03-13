"""
    Peaks.MomentsUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np

def compute_moments(x, y):
    W = np.sum(y)
    M1 = np.sum(x*y)/W              # raw moment
    M2 = np.sum(y*(x-M1)**2)/W      # central moment
    M3 = np.sum(y*(x-M1)**3)/W      # central moment
    return W, M1, M2, M3