# coding: utf-8
"""
    Optimizer.Demo.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
from time import sleep
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from molass_legacy._MOLASS.SerialSettings import set_setting
from CorrectedData import CorrectedXray
import molass_legacy.KekLib.DebugPlot as plt
from .OptimalElution import compute_optimal_elution
from MatrixData import simple_plot_3d
from ModeledData import ModeledData
from Tutorials.SvdPrecision import plot_text
from .GlobalInspector import FIG_SIZE, GlobalInspector

def demo(in_folder, sp=None, rank=None, debug=False, log_fh=None, fig_file=None, data_no=None, logger=None):
    print(in_folder)
    set_setting('in_folder', in_folder)

    if sp is None:
        xr = CorrectedXray(in_folder)
        D = xr.data
        E = xr.error
        qvector = xr.vector
        ecurve = xr.ecurve
        legacy_info = None
    else:
        from DecompUtils import CorrectedBaseline
        sp.load(in_folder)
        sd = sp.get_corrected_sd()
        D, E, qvector, ecurve = sd.get_xr_data_separate_ly()
        corbase_info = CorrectedBaseline(sd, sp.mapper)
        legacy_info = [sd, sp.mapper, corbase_info]

    fig = plt.figure(figsize=FIG_SIZE)

    # inspector_impl(fig, in_folder, D, E, qvector, ecurve, log_fh=log_fh, data_no=data_no, legacy_info=legacy_info, logger=logger)
    gi = GlobalInspector(D, E, ecurve, legacy_info, logger)

    if log_fh is not None:
        log_fh.write(','.join([str(data_no), in_folder] + ["%.3g" % r for r in gi.norm_ratios]) + "\n")
        log_fh.flush()

    gi.plot_results(fig, D, qvector)

    if debug:
        plt.show()
    else:
        if fig_file is not None:
            fig.savefig( fig_file )

        plt.show(block=False)
        sleep(0.5)

A_SIZE = 600
E_SIZE = 300
MATRIX_SYM_FONTSIZE = 200

def method_proof():
    q = np.linspace(0.01, 0.6, A_SIZE)
    rg_list = [50, 24]
    d_list = [1, 3]
    md = ModeledData(q, E_SIZE, rg_list=rg_list, d_list=d_list)
    M = md.get_data()

    fig = plt.figure(figsize=(21,12))
    gs = GridSpec(3,3)
    ax00 = fig.add_subplot(gs[0,0], projection='3d')
    ax01 = fig.add_subplot(gs[0,1])
    ax02 = fig.add_subplot(gs[0,2])
    ax10 = fig.add_subplot(gs[1,0], projection='3d')
    ax11 = fig.add_subplot(gs[1,1])
    ax12 = fig.add_subplot(gs[1,2])
    ax22 = fig.add_subplot(gs[2,2])

    fig.suptitle("Proof of the Factored Weights Method", fontsize=30)

    simple_plot_3d(ax00, M, x=q)

    for c in md.C:
        ax01.plot(c)
    plot_text(ax01, "$C$", fontsize=MATRIX_SYM_FONTSIZE)

    for k, p in enumerate(md.P.T):
        ax02.plot(q, p, color='C%d'%k)

    P_ = M @ np.linalg.pinv(md.C)
    for k, p in enumerate(P_.T):
        ax02.plot(q, p, ':', color='C%d'%k)
    plot_text(ax02, "$P$", fontsize=MATRIX_SYM_FONTSIZE)

    aw = 1/np.sum(M, axis=1)[:,np.newaxis]
    ew = 1/np.sum(M, axis=0)[np.newaxis,:]
    W = aw @ ew
    print('W.shape=', W.shape)
    WM = W*M
    simple_plot_3d(ax10, WM, x=q)

    WC = md.C*ew
    WP = WM @ np.linalg.pinv(WC)

    for c in WC:
        ax11.plot(c)
    plot_text(ax11, "$C_w$", fontsize=MATRIX_SYM_FONTSIZE)

    for p in WP.T:
        ax12.plot(q, p)
    plot_text(ax12, "$P_w$", fontsize=MATRIX_SYM_FONTSIZE)

    WP_ = WP/aw
    for p in WP_.T:
        ax22.plot(q, p)
    plot_text(ax22, "$P$", fontsize=MATRIX_SYM_FONTSIZE)

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.show()
