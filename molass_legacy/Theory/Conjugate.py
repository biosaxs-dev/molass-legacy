"""
    Theory.Conjugate.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt

def compute_conjugate_solutin(M, C, aq, bq):
    def obj_func(p):
        P_ = np.array([p, -bq]).T
        return np.linalg.norm(P_@C - M)

    res = minimize(obj_func, aq)
    return res.x, res.fun

def spike(preview_results):
    from SimpleGuinier import SimpleGuinier

    A1 = np.loadtxt(preview_results + '/A1.dat')
    B1 = np.loadtxt(preview_results + '/B1.dat')
    C1 = np.loadtxt(preview_results + '/C1.dat')
    M1 = np.loadtxt(preview_results + '/M1.dat')
    a1 = A1[:,1]
    b1 = B1[:,1]
    P1 = np.array([a1, b1]).T
    c = C1[0,:]
    C = np.array([c, c**2])
    norm1 = np.linalg.norm(P1@C - M1)
    print(norm1)

    a1_, minnorm = compute_conjugate_solutin(M1, C, a1, b1)
    print(minnorm, abs(minnorm - norm1)/np.linalg.norm(M1))

    qv = A1[:,0]
    sg = SimpleGuinier(np.array([qv, a1_, A1[:,2]]).T)
    print('Rg=', sg.Rg)

    fig, (ax1, ax2)= plt.subplots(ncols=2, figsize=(14, 7))

    ax1.set_yscale('log')
    ax1.plot(qv, a1, label='$A(q)$')
    ax1.plot(qv, a1_, label='$A^-(q)$')

    ax2.plot(qv, b1, label='$B(q)$')
    ax2.plot(qv, -b1, label='$-B(q)$')

    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    plt.show()
