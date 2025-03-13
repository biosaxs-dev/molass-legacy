"""
    Distance.NormalizedRmsd.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np

def normalized_rmsd(y1, y2, weights=None, adjust=0, debug=False):
    diff = y1 - y2
    if weights is not None:
        diff *= weights
    return np.log10(np.sqrt(np.mean(diff**2))) + adjust