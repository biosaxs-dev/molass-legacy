# coding: utf-8
"""
    SimultaneousLRF.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from matplotlib.gridspec import GridSpec
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition

def concatinate_data(pre_recog, X, U):
    M = np.zeros((X.shape[0]+U.shape[0], X.shape[1]))
    M[0:X.shape[0],:] = X

    slope = pre_recog.cs.slope
    intercept = pre_recog.cs.intercept
    for j in range(X.shape[1]):
        f = max(0, int(round(slope*(j-0.5) + intercept)))
        t = min(U.shape[1], int(round(slope*(j+0.5) + intercept)))
        v = np.average(U[:,f:t], axis=1)
        M[X.shape[0]:,j] = v
    return M

def demo(sd):
    from MatrixData import simple_plot_3d
    U = sd.conc_array
    X, E, qv, ecurve = sd.get_xr_data_separate_ly()
    pre_recog = PreliminaryRecognition(sd)
    M = concatinate_data(pre_recog, X, U)

    plt.push()
    fig = plt.figure(figsize=(21,11))
    gs = GridSpec(4,4)
    ax1 = fig.add_subplot(gs[0:2,0], projection="3d")
    ax2 = fig.add_subplot(gs[2:4,0], projection="3d")
    ax3 = fig.add_subplot(gs[1:3,1], projection="3d")

    simple_plot_3d(ax1, X)
    simple_plot_3d(ax2, U)
    simple_plot_3d(ax3, M)

    fig.tight_layout()
    plt.show()
    plt.pop()
