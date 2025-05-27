# coding: utf-8
"""
    Boundary.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import os
import numpy as np
from bisect import bisect_right
from matplotlib.patches import Rectangle
import molass_legacy.KekLib.DebugPlot as plt
from SvdDenoise import get_denoised_data
from DataUtils import get_in_folder

WIDTH = 40
BOUNDARY_RATIO = 0.8

def get_boundary(qv, M, max_q):
    qlim_i = bisect_right(qv, max_q)

    sv_ratios = []
    if qlim_i+WIDTH > len(qv):
        qlim_i = len(qv) - WIDTH

    # print(qlim_i, qv[qlim_i])

    for i in range(0,qlim_i):
        s = np.linalg.svd(M[i:i+WIDTH,:])[1]
        sv_ratios.append(s[1]/s[0])

    sv_ratios = np.array(sv_ratios)
    fwd_ratios = sv_ratios[WIDTH:]/sv_ratios[0:-WIDTH]
    k = int(len(fwd_ratios)*BOUNDARY_RATIO)
    p = np.argpartition(fwd_ratios, k)
    m = np.min(p[k:])
    return m+WIDTH, fwd_ratios, qlim_i, sv_ratios, p[k:]

def demo(sd):
    M, E, qv, ecurve = sd.get_xr_data_separate_ly()

    range_ = ecurve.get_ranges_by_ratio(0.5)[0]
    f = range_[0]
    p = range_[1]
    t = range_[2]+1
    eslice = slice(f,t)

    m, fwd_ratios, qlim_i, sv_ratios, pk_ = get_boundary(qv, M[:,eslice], 0.3)
    aslice = slice(0,qlim_i)

    x = ecurve.x
    y = ecurve.y
    c = y[f:t].copy()
    c /= np.max(c)

    M_ = M[aslice,eslice]
    M1 = get_denoised_data(M_, rank=1)
    C1 = np.array([c])
    P1 = M1 @ np.linalg.pinv(C1)

    M2 = get_denoised_data(M_, rank=2)
    C2 = np.array([c, c**2])
    P2 = M2 @ np.linalg.pinv(C2)

    y1 = P1[:,0]
    y2 = P2[:,0]
    bq = P2[:,1]

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(21, 7))
    ax2.set_yscale('log')
    # ax3.set_yscale('log')

    fig.suptitle("Rank Boundary Determination for " + get_in_folder(), fontsize=20)
    ax1.set_title("Elution Curve and Analysis Range", fontsize=16)
    ax2.set_title("Scattering Curves", fontsize=16)
    ax3.set_title("S1/S0 Ratio Variation and B(q)", fontsize=16)

    ax1.set_xlabel('Eno')
    ax2.set_xlabel('Q')
    ax3.set_xlabel('Q')

    ax1.plot(x, y, color='orange')

    ymin, ymax = ax1.get_ylim()
    patch = Rectangle(
            (f, ymin),      # (x,y)
            t - f,          # width
            ymax - ymin,    # height
            facecolor   = 'cyan',
            alpha       = 0.2,
        )

    ax1.add_patch(patch)

    qv_ = qv[aslice]
    ax2.plot(qv_, M[aslice,p], alpha=0.3, label='peak top')
    ax2.plot(qv_, y1, alpha=0.3, label='rank(1,1)')
    ax2.plot(qv_, y2, alpha=0.3, label='rank(2,2)')

    dw = (qv[WIDTH]-qv[0])      # assuming regular intervals
    ax3.plot(qv[0:qlim_i]+dw, sv_ratios, label='$bs_1/bs_0$')
    ax3.plot(qv[0:qlim_i-WIDTH]+dw, fwd_ratios, label='$(fs_1/fs_0)/(bs_1/bs_0)$')
    ax3.plot(qv[pk_]+dw, fwd_ratios[pk_], 'o', color='yellow', markersize=3, label='over %d %% points' % (100*BOUNDARY_RATIO))

    bx = qv[m]
    for ax in [ax2, ax3]:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
        ax.plot([bx, bx], [ymin, ymax], ':', color='red', label='rank boundary')

    axt = ax3.twinx()
    axt.plot(qv_, bq, color='pink', label='B(q)')
    axt.grid(False)

    ax3.set_ylabel('Ratio')
    axt.set_ylabel('B(q) Value')

    ax2.legend()
    ax3.legend(loc='center right')
    axt.legend(loc='upper right')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    plt.show()

def boundary_demo(sd):
    M, E, qv, ecurve = sd.get_xr_data_separate_ly()

    range_ = ecurve.get_ranges_by_ratio(0.5)[0]
    f = range_[0]
    p = range_[1]
    t = range_[2]+1
    eslice = slice(f,t)

    _, fwd_ratios, qlim_i, sv_ratios, pk_ = get_boundary(qv, M[:,eslice], 0.3)

    n = np.argmax(fwd_ratios)
    m = np.where(fwd_ratios[n:] < 1)[0][0] + n + 40

    fig, ax = plt.subplots()
    ax.set_yscale('log')

    y = M[:,p]
    ax.plot(qv, y)

    ymin, ymax = ax.get_ylim()

    for rank, fq, tq, color in [(2, qv[m-40], qv[m], 'pink'), (1, qv[m], qv[m+40], 'cyan')]:
        patch = Rectangle(
                (fq, ymin),     # (x,y)
                tq - fq,        # width
                ymax - ymin,    # height
                facecolor   = color,
                alpha       = 0.2,
                label = 'Rank %d' % rank,
            )

        ax.add_patch(patch)

    ax.legend()
    fig.tight_layout()
    plt.show()
