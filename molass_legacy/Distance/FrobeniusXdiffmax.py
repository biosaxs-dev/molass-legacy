"""
    Distance.FrobeniusXdiffmax.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np

def frobenius_xdiffmax(y1, y2, adjust=0, max_mask=None, debug=False):
    diff = y1 - y2
    d1 = np.linalg.norm(diff) 
    if max_mask is None:
        diff_ = diff
    else:
        diff_ = diff[max_mask]  # to avoid over-esiamtion from outliers
    std = np.std(diff_)
    d2 = std + np.max(np.abs(diff_))/std
    if debug:
        print("d1=", d1, "d2=", d2)
    return np.log10(d1*d2) + adjust