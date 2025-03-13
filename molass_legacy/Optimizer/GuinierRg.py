"""
    GuinierRg.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier

def compute_rg_params_impl(lrf_info):
    print("compute_rg_params_impl")
    qv = lrf_info.qv
    P = lrf_info.matrices[0]
    P_ = P[:,:-1]           # excluding baseline
    D = lrf_info.xrD
    E = lrf_info.xrE
    print(P.shape, D.shape, E.shape)
    """
        M = PC
        P = MC⁺
        Pe = sqrt(E**2 @ W**2)
    """
    D_pinv = np.linalg.pinv(D)
    W   = np.dot(D_pinv, P_)
    Pe  = np.sqrt(np.dot(E**2, W**2))

    ret_rgs = []
    qualities = []
    for y, ye in zip(P_.T, Pe.T):
        sg = SimpleGuinier(np.array([qv, y, ye]).T)
        ret_rgs.append(sg.Rg)
        qualities.append(sg.basic_quality)

    return np.array(ret_rgs), np.array(qualities)
