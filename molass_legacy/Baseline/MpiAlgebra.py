# coding: utf-8
"""
    Baseline.MpiAlgebra.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from MeasuredData import MeasuredData
import molass_legacy.KekLib.DebugPlot as plt
from MatrixData import simple_plot_3d
from .Baseline import compute_baseline
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from CurveDecomposer import decompose
from molass_legacy.Models.ElutionCurveModels import egh
from DataModels import GuinierPorod
from.PatchUtils import make_area_points

TITLE_FONTSIZE = 16

def drifted_elution(x):
    y1 = egh(x, 1, 100, 25, 5)
    y2 = egh(x, 0.8, 180, 20, 10)
    y_ = y1 + y2
    d = np.cumsum(y_*0.001)
    return np.array([y1, y2, d])

def demo(in_folder=None, split=True):
    q = np.arange(600)*0.001
    i = bisect_right(q, 0.02)
    x = np.arange(300)
    C = drifted_elution(x)

    p_list = []
    for k, params in enumerate([(50, 3, 100), (30, 2, 180), (10, 1, 299) ]):
        rg, d, j = params
        gp = GuinierPorod(1, rg, d)
        y = gp(q)
        p_list.append(y*C[k,j]/y[i])
    P = np.array(p_list).T
    M = np.dot(P, C)

    fig = plt.figure(figsize=(21,11))
    gs = GridSpec(2,3)
    ax00 = fig.add_subplot(gs[0,0], projection='3d')
    ax01 = fig.add_subplot(gs[0,1])
    ax02 = fig.add_subplot(gs[0,2])
    ax11 = fig.add_subplot(gs[1,1])
    ax12 = fig.add_subplot(gs[1,2])

    ax00.set_title("3D Data View", fontsize=TITLE_FONTSIZE, y=1.09)
    ax01.set_title("Component Elution Curves", fontsize=TITLE_FONTSIZE)
    ax02.set_title("Component Scattering Curves", fontsize=TITLE_FONTSIZE)

    if split:
        paren1 = "(split)"
        paren2 = "(overlaid with split ones)"
    else:
        paren1 = "(summed)"
        paren2 = "(overlaid with summed one)"
    ax11.set_title("Component Elution Curves " + paren1, fontsize=TITLE_FONTSIZE)
    ax12.set_title("Component Scattering Curves " + paren2, fontsize=TITLE_FONTSIZE)

    simple_plot_3d(ax00, M, x=q)
    qi = np.ones(len(x))*q[i]
    y = x
    z = M[i,:]
    ax00.plot(qi, y, z, color='orange')

    ty = np.zeros(len(x))

    for k, c in enumerate(C):
        ty += c
        for ax in [ax01, ax11]:
            ax.plot(x, c, ':', label='C[%d,:]' % k)

    for ax in [ax01, ax11]:
        ax.plot(x, ty, color='orange', label='total')

    for k, y in enumerate(p_list):
        for ax in [ax02, ax12]:
            ax.plot(q, y, ':', label='P[:,%d]' % k)

    Pmpi = np.dot(M, np.linalg.pinv(C))

    if False:
        for y in Pmpi.T:
            ax12.plot(q, y)

    if split:
        w = np.random.uniform(0,1,300)
        ca = C[0,:]*w
        cb = C[0,:]*(1-w)
        points = make_area_points(x, C[0,:], ca)
        area = Polygon(points, alpha=0.3, color='cyan')
        ax11.add_patch(area)
        points = make_area_points(x, ca)
        area = Polygon(points, alpha=0.3, color='pink')
        ax11.add_patch(area)
        w = np.random.uniform(0,1,300)
        cc = C[1,:]*w
        cd = C[1,:]*(1-w)
        points = make_area_points(x, C[1,:], cc)
        area = Polygon(points, alpha=0.3, color='yellow')
        ax11.add_patch(area)
        points = make_area_points(x, cc)
        area = Polygon(points, alpha=0.3, color='green')
        ax11.add_patch(area)
        C_ = np.array([ca, cb, cc, cd, C[2,:]])
        P_ = np.dot(M, np.linalg.pinv(C_))

        for k, y in enumerate(P_.T):
            if k < 4:
                ul = '_upper' if k%2 == 0 else '_lower'
            else:
                ul = ''
            ax12.plot(q, y, '-', markersize=1, label='P_[:,%d]%s' % (k//2, ul))

    else:
        cs = C[0,:]+C[1,:]
        ax11.plot(x, cs, label='C[0,:]+C[1,:]')

        C_ = np.array([cs, C[2,:]])
        P_ = np.dot(M, np.linalg.pinv(C_))

        for k, y in enumerate(P_.T):
            ax12.plot(q, y, '-', markersize=1, label='P_[:,%d]' % k)

    for ax in [ax01, ax02, ax11, ax12]:
        ax.legend()

    fig.tight_layout()
    plt.show()
