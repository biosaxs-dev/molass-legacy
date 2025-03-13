"""
    XrayLrfConc.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right

def estimate_true_concentration(qv, D):
    i = bisect_right(qv, 0.02)

    cy_init = D[i,:]
    cy = cy_init
    cy2 = cy**2
    C_ = np.array([cy, cy2])

    C_list = [C_]

    for n in range(3):
        C_inv = np.linalg.pinv(C_)
        P_ = D @ C_inv
        P_inv = np.linalg.pinv(P_)
        C_ = P_inv @ D
        C_list.append(C_)

    return C_
