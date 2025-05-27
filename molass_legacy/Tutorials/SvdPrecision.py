# coding: utf-8
"""
    SvdPrecision.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from matplotlib.gridspec import GridSpec
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.OurMatplotlib import get_color
from ModeledData import ModeledData, simple_plot_3d
from XrayData import XrayData
from molass_legacy.ElutionDecomposer import ElutionDecomposer
from ExistingNmf import ExistingNmf

def svd_variation_by_noise():
    qvector = np.linspace(0.01, 0.6, 600)
    rg_list = [50, 24]
    d_list = [1, 3]
    md = ModeledData(qvector, 300, rg_list=rg_list, d_list=d_list)
    md.plot_components()
    M = md.get_data()
    fig = plt.figure(figsize=(40,24))
    case_type = 'Weak' if d_list[0] == d_list[1] else 'Strong'
    fig.suptitle('Matrix Factorization Comparison in case of %s Linear Independence' % case_type, fontsize=40)
    title_fontsize = 16
    gs = GridSpec(8,5)
    axes1 = []
    axes2 = []
    E = np.ones(M.shape)
    mu = 0
    sigma = 1
    for j in range(5):
        ax1 = fig.add_subplot(gs[0,j], projection='3d')
        ax2 = fig.add_subplot(gs[1,j])
        ax3 = fig.add_subplot(gs[2,j])
        ax4 = fig.add_subplot(gs[3,j])
        ax5 = fig.add_subplot(gs[4,j])
        ax6 = fig.add_subplot(gs[5,j])
        ax7 = fig.add_subplot(gs[6,j])
        ax8 = fig.add_subplot(gs[7,j])

        noise = j*0.05
        ax1.set_title('Noise %d %%' % (int(noise*100)), fontsize=title_fontsize)
        ax2.set_title('SVD major U spectra', fontsize=title_fontsize)
        ax3.set_title('SVD major Sigular Values', fontsize=title_fontsize)
        ax5.set_title('Elution Curve Decomposition (SA)', fontsize=title_fontsize)
        ax6.set_title('Elution Curve Decomposition (PMF)', fontsize=title_fontsize)
        ax7.set_title('Component Scattering Curves (SA)', fontsize=title_fontsize)
        ax8.set_title('Component Scattering Curves (PMF)', fontsize=title_fontsize)

        axes1.append(ax1)
        axes2.append(ax2)
        E_ = E + np.random.normal(mu, sigma, M.shape) * noise
        M_ = M * E_
        simple_plot_3d(ax1, M_)
        U, s, VT = np.linalg.svd(M_)
        V = VT.T
        for k in range(2):
            ax2.plot(U[:,k])
        ax3.plot(s[0:5], ':')
        ax3.plot(s[0:5], 'o')
        for k in range(2):
            ax4.plot(V[:,k])
        ax1.set_zlim(-0.1, 1.2)
        ax3.set_ylim(-3, 65)
        xd = XrayData(None, q=qvector, data=M_, error=E_)
        e_curve = xd.e_curve
        x = e_curve.x
        y = e_curve.y
        ax5.plot(y)
        decomp = ElutionDecomposer(e_curve, x, y)
        c_list = []
        for rec in decomp.fit_recs:
            f = rec.evaluator
            c = f(x)
            ax5.plot(x, c)
            c_list.append(c)

        ax6.plot(y)
        nmf = ExistingNmf(M_)
        H = nmf.H
        W = nmf.W

        h1 = H[0,:]
        h2 = H[1,:]
        w1 = W[:,0]
        w2 = W[:,1]

        i = int(e_curve.peak_top_x[0])
        numbered_order = h1[i] > h2[i]
        h_list = [h1, h2] if numbered_order else [h2, h1]
        w_list = [w1, w2] if numbered_order else [w2, w1]

        scales = solve_scale(y, h_list)
        for k, h in enumerate(h_list):
            ax6.plot(x, h*scales[k])

        C_ = np.array(c_list)
        rank = 2
        Us_ = np.dot( U[:,0:rank], np.diag( s[0:rank] ) )
        MD  = np.dot( Us_, VT[0:rank,:] )
        Cinv = np.linalg.pinv(C_)
        P_ = np.dot(MD, Cinv)

        for k in range(P_.shape[1]):
            color = get_color(k+1)
            ax7.plot(qvector, md.P[:, k], ':', color=color, label='model truth[%d]' % k)
            ax7.plot(qvector, P_[:,k], color=color, label='decomposed[%d]' % k)
            w = w_list[k]
            ax8.plot(qvector, md.P[:,k], ':', color=color, label='model truth[%d]' % k)
            scale = 1/scales[k]
            ax8.plot(qvector, w*scale, color=color, label='decomposed[%d]' % k)

        for ax in [ax7, ax8]:
            ax.legend(fontsize=16)

        plot_text(ax2, 'u0, u1')
        plot_text(ax3, 's0..s4')
        plot_text(ax4, 'v0, v1')
        plot_text(ax5, 'SA')
        plot_text(ax6, 'PMF')
        plot_text(ax7, 'SA')
        plot_text(ax8, 'PMF')

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.show()

def solve_scale(y, h_list):
    h_array = np.array(h_list)

    def obj_func(scale):
        debug = False
        if debug:
            plt.push()
            fig = plt.figure()
            ax = fig.gca()
            ax.set_title('scale=%.3g'% scale[0])
            ax.plot(y)

        y_ = np.zeros(len(y))
        for k in range(h_array.shape[0]):
            h = h_array[k,:]*scale[k]
            if debug:
                ax.plot(h)
            y_ += h

        if debug:
            plt.show()
            plt.pop()

        return np.sum((y - y_)**2)

    init_x = np.ones((2,))
    result = minimize(obj_func, init_x)
    return result.x

def plot_text(ax, text, alpha=0.05, fontsize=100):
    x = ax.get_xlim()
    y = ax.get_ylim()
    tx = np.average(x)
    ty = np.average(y)
    ax.text(tx, ty, text, alpha=alpha, fontsize=fontsize, ha='center', va='center')
