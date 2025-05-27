# coding: utf-8
"""
    EFA.Demo.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from CorrectedData import CorrectedXray
import molass_legacy.KekLib.DebugPlot as plt
from MatrixData import simple_plot_3d

def demo(in_folder):
    print(in_folder)
    xr = CorrectedXray(in_folder)
    D = xr.data
    print(D.shape)
    print('D[0:3,0:5]=', D[0:3,0:5])
    E = np.average(xr.error, axis=1)
    print(E.shape)
    print('E[0:3]=', E[0:3])
    A = (D.T/E.T).T
    print(A.shape)
    print('A[0:3,0:5]=', A[0:3,0:5])
    U, s, VT = np.linalg.svd(A)
    print('s.shape=', s.shape)

    num_elutions = A.shape[1]
    num_sings = 3
    fwd_sing_array = np.zeros((num_elutions, num_sings))*np.nan
    bwd_sing_array = np.zeros((num_elutions, num_sings))*np.nan
    for k in range(num_elutions):
        _, fs, _ = np.linalg.svd(A[:,0:k+1])
        _, bs, _ = np.linalg.svd(A[:,-(k+1):])
        fs_ = fs[0:num_sings]
        bs_ = bs[0:num_sings]
        fwd_sing_array[k,0:len(fs_)] = fs_
        bwd_sing_array[-(k+1),0:len(bs_)] = bs_

    gs = GridSpec(3,3)
    fig = plt.figure(figsize=(21, 12))
    ax00 = fig.add_subplot(gs[0,0], projection='3d')
    ax01 = fig.add_subplot(gs[0,1], projection='3d')
    ax02 = fig.add_subplot(gs[0,2])
    ax10 = fig.add_subplot(gs[1,0])
    ax11 = fig.add_subplot(gs[1,1])
    ax12 = fig.add_subplot(gs[1,2])
    ax12t = ax12.twinx()
    ax20 = fig.add_subplot(gs[2,0])
    ax21 = fig.add_subplot(gs[2,1])
    ax22 = fig.add_subplot(gs[2,2])
    ax22t = ax22.twinx()
    xr.plot_3d(ax00)
    simple_plot_3d(ax01, A)
    ax02.plot(s[0:5], '-o')
    for i in range(num_sings):
        fy = fwd_sing_array[:,i]
        by = bwd_sing_array[:,i]
        for ax in [ax10, ax12, ax20, ax22]:
            ax.plot(fy, label='fwd s%d' % i)

        for ax in [ax11, ax12t, ax21, ax22t]:
            ax.plot(by, label='bwd s%d' % i)

    ylim_array = np.array([ax.get_ylim() for ax in [ax10, ax11, ax12]])
    ymin = min(ylim_array[:,0])
    ymax = max(ylim_array[:,1])

    ey = xr.ecurve.y / xr.ecurve.max_y * ymax
    ex = np.arange(len(ey))
    points = [(0,0)] + list(zip(ex,ey))+ [(ex[-1],0)]

    for ax in [ax10, ax11,ax12]:
        poly = Polygon(points, alpha=0.1)
        ax.add_patch(poly)

    ymin_ = ymin*0.1
    ymax_ = ymax*0.1
    ey_ = xr.ecurve.y / xr.ecurve.max_y * ymax_
    points_ = [(0,0)] + list(zip(ex,ey_))+ [(ex[-1],0)]

    for ax in [ax20, ax21,ax22, ax22t]:
        ax.set_ylim(ymin_, ymax_)

    for ax in [ax20, ax21,ax22]:
        poly = Polygon(points_, alpha=0.1)
        ax.add_patch(poly)

    fig.tight_layout()
    plt.show()
