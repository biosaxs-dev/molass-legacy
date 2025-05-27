"""
    Theory.SynthesizedLRF.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from scipy.optimize import minimize
from matplotlib.patches import Rectangle
from time import time
from molass_legacy.KekLib.SciPyCookbook import smooth
import molass_legacy.KekLib.DebugPlot as plt
from SvdDenoise import get_denoised_data
from .SolidSphere import phi
from .Rg import compute_corrected_Rg
from .SolidSphere import get_boundary_params_simple

def synthesized_lrf_spike(qv, M, c, M2, P2, boundary=None, k=1):
    w = 1/(1+np.exp(-k*(qv-boundary)))

    M1 = get_denoised_data(M, rank=1)
    C1 = np.array([c])
    P1 = M1 @ np.linalg.pinv(C1)

    aq = P2[:,0] * (1-w) + P1[:,0] * w
    bq = P2[:,1] * (1-w) + np.zeros(P1.shape[0]) * w
    return np.array([aq, bq]).T

def demo(root, sd, pno=0, debug=False):
    from .BoundedLRF import demo as demo_impl
    demo_impl(root, sd, pno=pno, synthesized_lrf=synthesized_lrf_spike, bounded_lrf=True, debug=debug)

def get_reduced_conc(C_, cdl_list):
    c_list = []
    for k, cd in enumerate(cdl_list):
        if cd == 2:
            c_list.append(C_[k,:])
    return np.array(c_list)

def synthesized_lrf(qv, D, C, rank, Cred, Rg):
    t0 = time()

    D2 = get_denoised_data(D, rank=rank)
    P2 = D2 @ np.linalg.pinv(C)

    red_rank = len(Cred)
    D1 = get_denoised_data(D, rank=red_rank)
    P1 = D1 @ np.linalg.pinv(Cred)

    b1, b2, k = get_boundary_params_simple(Rg)
    w = 1/(1+np.exp(-k*(qv-b1)))

    aq = P2[:,0] * (1-w) + P1[:,0] * w
    bq = P2[:,1] * (1-w) + np.zeros(P1.shape[0]) * w

    # this is to be compatible with ExtrapolationSolver.fit
    # opt_info = [num_iterations, func_calls, time_elapsed]
    opt_info = [0, 0, time() - t0]

    P = np.array([aq, bq]).T
    return P, opt_info, D2, b1
