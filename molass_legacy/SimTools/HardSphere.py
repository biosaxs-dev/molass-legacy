"""
    SimTools.HardSphere.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from Theory.SolidSphere import phi
from molass_legacy.Models.ElutionCurveModels import egha

def formfactor_factor(qv, R):
    return phi(qv, R)**2

def structure_factor(qv, K, R):
    return 1 - K*phi(qv, 2*R)

def get_model_data(qv, jv, rg, h, mu, sigma, tau, a=0, K=0, error=0, return_error_matrix=False):
    R = np.sqrt(5/3)*rg

    fy = formfactor_factor(qv, R)
    sy = structure_factor(qv, K, R)

    if error > 0:
        fy += np.random.normal(0, np.max(fy)*error, len(qv))
        sy += np.random.normal(0, np.max(np.abs(sy))*error, len(qv))

    aq = fy
    bq = fy*(sy - 1)
    P = np.array([aq, bq]).T

    cy = egha(jv, h, mu, sigma, tau, a)

    if error > 0:
        cy += np.random.normal(0, np.max(cy)*error, len(jv))

    C = np.array([cy, cy**2])

    M = P @ C

    if return_error_matrix:
        from SvdDenoise import get_denoised_error
        M_, P_, C_ = get_model_data(qv, jv, rg, h, mu, sigma, tau, a=a, K=K, error=0)
        E = M - M_
        E_ = get_denoised_error(M_, M, E)   # better method wanted
        return M, E_, P, C
    else:
        return M, P, C
