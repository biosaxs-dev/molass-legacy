# coding: utf-8
"""
    PhaseRetrievalDemo.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from .PhaseRetrieval import plot_3d_image, plot_3d_scatter, fienup_phase_retrieval
from .ReciprocalData import ReciprocalData
from .SaxsSamples import BallVoxels

def ball_demo(**kwargs):
    from .SaxsSamples import BallVoxels
    n = kwargs.get('n')
    radius = kwargs.pop('radius', 0.3)
    ball = BallVoxels(shape=(n,n,n), radius=radius)
    object_demo(ball, **kwargs)

def disc_demo(**kwargs):
    from .SaxsSamples import DiscVoxels
    n = kwargs.get('n')
    radius = kwargs.pop('radius', 0.5)
    height = kwargs.pop('height', 0.2)
    disc = DiscVoxels(shape=(n,n,n), radius=radius, height=height)
    object_demo(disc, **kwargs)

def torus_demo(**kwargs):
    from .SaxsSamples import TorusVoxels
    n = kwargs.get('n')
    R = kwargs.pop('R', 0.3)
    r = kwargs.pop('r', 0.1)
    torus = TorusVoxels(shape=(n,n,n), R=R, r=r)
    object_demo(torus, **kwargs)

def object_demo(obj_voxels, fig=None, n=32, gpu=None, exact_mag=False, use_mask=False):
    if fig is None:
        fig_ = plt.figure(figsize=(24,15))
    else:
        fig_ = fig

    if gpu is None:
        from Env.EnvInfo import get_global_env_info
        env_info = get_global_env_info()
        gpu = env_info.nvidiagpu_is_available

    t0 = time.time()

    density = 0.1
    data = obj_voxels.get_data(density=density)
    q = np.linspace(0, 0.5, 600)
    rdata = ReciprocalData(data.shape)
    F = rdata.get_reciprocal(data)
    y = rdata.get_scattering_curve(q, F)
    x = rdata.qbinsc
    # print('rdata.qbin_labels=', rdata.qbin_labels)
    print('rdata.qbin_labels.dtype=', rdata.qbin_labels.dtype)
    qv = rdata.interp(x)
    absF = np.abs(F)
    if exact_mag:
        magnitudes = absF
    else:
        magnitudes = rdata.get_3d_magnitudes(qv)
    min_value = density/2
    if use_mask:
        ball = BallVoxels(shape=(n,n,n), radius=n)
        mask = ball.get_data()
    else:
        mask = None
    result = fienup_phase_retrieval(magnitudes, mask=mask,
                            steps=10000, gpu=gpu,
                            min_value=min_value,
                            verbose=True)

    print('It took %.3g seconds.' % (time.time() - t0))

    gs = gridspec.GridSpec(3,4)
    fig_.suptitle("Phase Retrieval for 3D in SAXS with (%d,%d,%d) voxels" % (n,n,n), fontsize=20)
    ax0 = fig_.add_subplot(gs[0,0], projection='3d')
    ax1 = fig_.add_subplot(gs[0,1], projection='3d')
    ax2 = fig_.add_subplot(gs[0,2], projection='3d')
    ax3 = fig_.add_subplot(gs[0,3])
    ax4 = fig_.add_subplot(gs[1,0], projection='3d')
    ax5 = fig_.add_subplot(gs[1,1], projection='3d')
    ax6 = fig_.add_subplot(gs[1,2], projection='3d')
    ax7 = fig_.add_subplot(gs[1,3])
    ax8 = fig_.add_subplot(gs[2,0], projection='3d')
    ax9 = fig_.add_subplot(gs[2,1], projection='3d')
    axa = fig_.add_subplot(gs[2,2], projection='3d')
    axb = fig_.add_subplot(gs[2,3])

    fontsize = 16
    ax0.set_title('Time Domain Object', y=1.1, fontsize=fontsize)
    ax1.set_title('Frequency Domain abs(F) (from Object)', y=1.1, fontsize=fontsize)
    ax2.set_title('qbin_labels for 3D → 1D spherical average', y=1.1, fontsize=fontsize)
    ax3.set_title('Scattering Curve', fontsize=fontsize)
    ax4.set_title('Time Domain Retrieved Object', y=1.1, fontsize=fontsize)
    ax5.set_title('Frequency Domain abs(F) (from Curve)', y=1.1, fontsize=fontsize)
    ax6.set_title('abs(F)(from Curve) - abs(F)(from Ball)', y=1.1, fontsize=fontsize)
    ax7.set_title('qbin values (adopted points for phase retrieval)', fontsize=fontsize)
    ax8.set_title('Time Domain Retrieved Object', y=1.1, fontsize=fontsize)
    ax9.set_title('Frequency Domain abs(F) (from Retrieved Object)', y=1.1, fontsize=fontsize)
    axa.set_title('qbin_labels for 3D → 1D spherical average', y=1.1, fontsize=fontsize)
    axb.set_title('Scattering Curve from Retrieved Object', fontsize=fontsize)

    imean = rdata.Imean[rdata.Imean > 0]
    ymax = np.log10(imean.max())
    ymin = np.log10(imean.min())
    ymax_ = ymin*(-0.1) + ymax*1.1
    ymin_ = ymin*1.1 + ymax*(-0.1)
    for ax in [ax3, ax7, axb]:
        ax.set_ylim(ymin_, ymax_)
        ax.set_xlabel('Q')
        ax.set_ylabel('$Log_{10}$(Intensity)')

    plot_3d_scatter(ax0, data, shape_limits=True)
    logy = np.log10(y)
    ax3.plot(q, logy, color='orange')
    log_imean = np.log10(rdata.Imean)
    ax3.scatter(x, log_imean, cmap=cm.cool, c=x)

    plot_3d_scatter(ax2, rdata.qbin_labels, cmap=cm.cool)
    plot_3d_scatter(ax1, absF, cmap=cm.bwr)

    ax7.plot(q, logy, color='orange', label='simulated curve')
    ax7.scatter(x, np.log10(qv), cmap=cm.cool, c=x)
    yr = rdata.get_scattering_curve(q, magnitudes, absolute=True)
    ax7.plot(q, np.log10(yr), ':', color='green', label='curve for verification')
    ax7.legend()

    diff = magnitudes - absF
    plot_3d_scatter(ax6, diff, cmap=cm.bwr)
    plot_3d_scatter(ax5, magnitudes, cmap=cm.bwr)
    plot_3d_scatter(ax4, result, min_value=min_value, shape_limits=True)

    data_verif = copy.deepcopy(result)
    data_verif[data_verif < min_value] = 0
    plot_3d_scatter(ax8, data_verif, shape_limits=True)
    Fr = rdata.get_reciprocal(data_verif)
    absFr = np.abs(Fr)
    plot_3d_scatter(ax9, absFr, cmap=cm.bwr)
    plot_3d_scatter(axa, rdata.qbin_labels, cmap=cm.cool)

    yv = rdata.get_scattering_curve(q, Fr)
    axb.plot(q, logy, color='orange', label='simulated curve')
    axb.plot(q, np.log10(yv), ':', color='blue', label='curve from retrieved object')
    axb.set_xlim(ax7.get_xlim())
    axb.set_ylim(ax7.get_ylim())
    axb.legend()

    fig_.tight_layout()
    fig_.subplots_adjust(top=0.9)

    if fig is None:
        plt.show()
