"""
    LrfRgUtils.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier

def compute_rg_from_qvDEP(qv, D, E, P, return_sg=False):
    aq = P.T[0]
    Dinv = np.linalg.pinv(D)
    W = np.dot(Dinv, P)
    Pe = np.sqrt(np.dot(E**2, W**2))
    data = np.array([qv, aq, Pe[:,0]]).T

    sg = SimpleGuinier(data)
    if return_sg:
        return sg
    else:
        return sg.Rg
