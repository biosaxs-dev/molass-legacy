# coding: utf-8
"""
    EFA.RotationDemo.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import matplotlib.animation as animation
from CorrectedData import CorrectedXray
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.OurMatplotlib import get_color
from MatrixData import simple_plot_3d
from ModeledData import ModeledData
from Tutorials.MatrixFactorization import draw_PQ, generate_random_points
from Tutorials.SvdPrecision import plot_text

FONTSIZE = 160
A_SIZE = 600

def cpc_demo(symmetric=False):
    q = np.linspace(0.01, 0.6, A_SIZE)
    rg_list = [50, 24]
    d_list = [1, 3]

    if symmetric:
        E_SIZE = 300
        md = ModeledData(q, E_SIZE, rg_list=rg_list, d_list=d_list)
        f1, t1 = 0, 200
        f2, t2 = 100, E_SIZE
    else:
        E_SIZE = 500
        md = ModeledData(q, E_SIZE, rg_list=rg_list, d_list=d_list,
                            h_list=[1, 0.5], mu_list=[130, 220], sigma_list=[20, 30], tau_list=[40, 20])
        f1, t1 = 0, 350
        f2, t2 = 120, E_SIZE

    R = np.zeros(md.C.shape)
    R[0,f1:t1] = 1
    R[1,f2:t2] = 1

    def normalize_C(C):
        C = R*C
        for k in range(C.shape[0]):
            C[k,:] /= np.sum(C[k,:])
        return C

    def one_iteration(C):
        P = M_ @ np.linalg.pinv(normalize_C(C))
        C_ = np.linalg.pinv(P) @ M_
        return C_

    fig = plt.figure(figsize=(22,11))
    fig.suptitle("Early (1-4) C-P-C Iterations", fontsize=40)
    gs = GridSpec(3,6)
    axes = []
    for i in range(3):
        ax_row = []
        for j in range(6):
            ax = fig.add_subplot(gs[i,j], projection='3d' if i+j == 0 else None)
            ax_row.append(ax)
        axes.append(ax_row)

    rank = len(rg_list)
    axes = np.array(axes)

    ax00 = axes[0,0]
    M = md.get_data()
    U, s, VT = np.linalg.svd(M)
    M_ = U[:,0:rank] @ np.diag( s[0:rank] ) @ VT[0:rank,:]
    simple_plot_3d(ax00, M_, x=q)

    ax10 = axes[1,0]
    for p in md.P.T:
        ax10.plot(q, p)

    ax20 = axes[2,0]
    for c in md.C:
        ax20.plot(c)

    poly1 = Polygon([(f1,0), (f1,1), (t1,1), (t1,0)], color='cyan', alpha=0.2)
    ax20.add_patch(poly1)
    poly2 = Polygon([(f2,0), (f2,1), (t2,1), (t2,0)], color='pink', alpha=0.2)
    ax20.add_patch(poly2)

    def early_iterations(i, C):
        ax0 = axes[i,1]
        for c in C:
            ax0.plot(c)

        for j in range(2,6):
            ax = axes[i,j]
            C = one_iteration(C)
            for c in C:
                ax.plot(c)
            plot_text(ax, str(j-1), fontsize=FONTSIZE)

    early_iterations(0, VT[0:rank,:])
    plot_text(axes[0,1], "$V^T$", fontsize=FONTSIZE)

    w = np.linspace(0, 1, E_SIZE)
    W = np.vstack([w, 1-w])
    early_iterations(1, W)

    r = np.random.uniform(0, 1, E_SIZE)
    R = np.vstack([r, 1-r])
    early_iterations(2, R)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()
