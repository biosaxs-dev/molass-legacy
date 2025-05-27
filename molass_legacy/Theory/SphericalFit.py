"""
    Theory.SphericalFit.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from matplotlib.patches import Rectangle
import molass_legacy.KekLib.DebugPlot as plt
from PlotUtils import align_yaxis_np
from MatrixData import simple_plot_3d
from SvdDenoise import get_denoised_data
from .Rg import compute_corrected_Rg, compute_corrected_Rg_impl
from .SolidSphere import phi
from Rank.Synthesized import synthesized_data
from .SolidSphere import get_boundary_params, get_boundary_params_simple
from DataUtils import get_in_folder

CONC_PICK_RATIO = 0.8

def conc_pick(i):
    return int(i*CONC_PICK_RATIO)

def estimate_L(qv, M, Rg):
    R = np.sqrt(5/3)*Rg
    M_ = get_denoised_data(M, rank=2)
    Aq = phi(qv,R)**2
    Bq = -Aq*phi(qv,2*R)
    P_ = np.array([Aq, Bq]).T
    C_ = np.linalg.pinv(P_) @ M_

    c = C_[0,:]
    lc2 = C_[1,:]
    L_ = (lc2[np.newaxis,:]) @ np.linalg.pinv(c[np.newaxis,:]**2)

    return L_[0,0], C_

def debug_plot(c, qv, Aq, Bq, bq_):
    plt.push()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,7))
    ax1.plot(c)
    ax2.set_yscale('log')
    ax2.plot(qv, Aq)
    ax2t = ax2.twinx()
    ax2t.plot(qv, Bq)
    qv_ = qv[0:len(bq_)]
    ax2t.plot(qv_, bq_, color='pink')
    fig.tight_layout()
    plt.show()
    plt.pop()

def estimate_LR(arg_qv, arg_M, Rg, b1, debug=False):
    i = bisect_right(arg_qv, b1)
    qv = arg_qv[0:i]
    M = arg_M[0:i,:]

    R = np.sqrt(5/3)*Rg
    M_ = get_denoised_data(M, rank=2)
    Aq = phi(qv,R)**2

    c = arg_M[conc_pick(i),:]

    Pw = np.zeros((len(qv),2))
    Pw[:,0] = Aq

    def objective(x):
        rscale, bscale = x
        bq = -Aq*phi(qv, rscale*R)
        Cw = np.array([c, c**2])
        Pw = M_ @ np.linalg.pinv(Cw)
        return np.linalg.norm(bq - bscale*Pw[:,1])

    res = minimize(objective, (2, 1), bounds=((2, 10), (1e-3, None)))
    rscale, bscale = res.x
    score = res.fun/np.linalg.norm(-Aq*phi(qv, rscale*R))
    print('res.x=', res.x, 'res.fun=', res.fun, 'score=', score)

    Bq = -Aq*phi(qv, rscale*R)

    if debug:
        Cw = np.array([c, c**2])
        Pw = M_ @ np.linalg.pinv(Cw)
        bq_ = bscale*Pw[:,1]
        debug_plot(c, qv, Aq, Bq, bq_)

    P_ = np.array([Aq, Bq]).T
    C_ = np.linalg.pinv(P_) @ M_
    c = C_[0,:]
    lc2 = C_[1,:]
    L_ = (lc2[np.newaxis,:]) @ np.linalg.pinv(c[np.newaxis,:]**2)

    return L_[0,0], C_, rscale, bscale, score

def demo(parent, sd, pno=0, use_estimated_c=False, use_better_estimate=True, debug=False):
    M, E, qv, ecurve = sd.get_xr_data_separate_ly()

    range_ = ecurve.get_ranges_by_ratio(0.5)[pno]
    f_ = range_[0]
    p_ = range_[1]
    t_ = range_[2]

    if True:
        f, t = f_, t_
    else:
        if p_ < M.shape[1]*0.7:
            f, t = p_, t_
        else:
            f, t = f_, p_

    eslice = slice(f,t)

    Rg, gf, gt, guinier_truncated = compute_corrected_Rg(sd, ecurve, pno, qv, M[:,eslice], E[:,eslice])
    aslice = slice(0, gt)

    fig = plt.figure(figsize=(21,7))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    fig.suptitle("Spherical Fitting Demo for " + get_in_folder(), fontsize=30)
    ax1.set_title("Guinier Region" + guinier_truncated, fontsize=20)
    ax2.set_title("Estimated Concentration Curve", fontsize=20)
    ax3.set_title("Scattering Curves and Interparticle Effects", fontsize=20)

    simple_plot_3d(ax1, M, x=qv, alpha=0.3)
    if False:
        gslice = slice(gf, gt)
        simple_plot_3d(ax1, M[gslice,:], x=qv[gslice], color='green', edgecolor='green', alpha=0.5)

    y = sd.jvector

    for i in [gf, gt]:
        x = np.ones(len(y))*qv[i]
        z = M[i,:]
        ax1.plot(x, y, z, color='green')

    if use_better_estimate:
        b1, b2, k = get_boundary_params_simple(Rg)
        L, C_, rscale, bscale, score = estimate_LR(qv, M[:,eslice], Rg, b1, debug=debug)
        print('L=', L, 'bscale=', bscale)
        c = C_[0,:]
        kc2 = C_[1,:]
    else:
        L, C_ = estimate_L(qv[aslice], M[aslice,eslice], Rg)
        print('L=', L)
        c = C_[0,:]
        kc2 = C_[1,:]
        rscale = 2
        bscale = 1
        b1, b2, k = get_boundary_params(Rg)

    R = np.sqrt(5/3)*Rg
    aq_ = phi(qv,R)**2
    # rq_ = -L * phi(qv,rscale*R)
    rq_ = -phi(qv,rscale*R)
    bq_ = aq_ * rq_

    print('boundary=', b1)
    M_, _ = synthesized_data(qv, M[:,eslice], Rg, cd=2, boundary=b1, k=k)

    i = bisect_right(qv, b1)
    if use_estimated_c:
        c_ = c
    else:
        # c_ = M_[i,:]    # this makes a spike at the boundary
        c_ = M[conc_pick(i),eslice]

    print('max c =', np.max(c))

    qlim = len(qv)
    C = np.array([c_, c_**2])
    P = M_[0:qlim,:] @ np.linalg.pinv(C)
    aq = P[:,0]
    scale = aq[0]
    print('aq scale=', scale)

    aq = P[:,0]/scale
    if use_better_estimate:
        bq = P[:,1]*bscale
    else:
        bq = P[:,1]/(scale**2)      # not sure if this is correct

    if debug:
        debug_plot(c_, qv, aq_, bq_, bq)

    j = np.argmax(c)
    ey = ecurve.y*c[j]/ecurve.y[f+j]

    ax2.plot(y, ey, ':', color='orange')
    y_ = y[eslice]
    ax2.plot(y_, c, color='orange', label='$ c_{1,*} $')
    ax2t = ax2.twinx()
    ax2t.grid(False)
    ax2t.plot(y_, L*c**2, alpha=0.3, label='$ L(=%.3g) \cdot {c_{1,*}}^2 $' % L)
    ax2t.plot(y_, kc2, alpha=0.3, label='$ c_{2,*} $')
    align_yaxis_np(ax2, ax2t)

    leg = ax2.legend(loc='upper left', fontsize=16)
    fig.canvas.draw()
    bounds = leg.get_frame().get_bbox().bounds
    x, y = ax2.transAxes.inverted().transform(bounds[0:2])
    print('(x,y)=', (x,y))
    if x < 0.5:
        pos = (0,y)
        side = 'left'
    else:
        pos = (1,y)
        side = 'right'
    ax2t.legend(bbox_to_anchor=pos, loc='upper ' + side, fontsize=16)

    ymin, ymax = ax2.get_ylim()
    p = Rectangle(
            (f, ymin),      # (x,y)
            t - f,          # width
            ymax - ymin,    # height
            facecolor   = 'cyan',
            alpha       = 0.2,
        )
    ax2.add_patch(p)

    ax3.set_yscale('log')

    ax3.plot(qv, aq_, color='C0', label='sphere A(q)')
    ax3.plot(qv[0:qlim], aq, color='C1', label='sample A(q)')

    ymin, ymax = ax3.get_ylim()
    ax3.set_ylim(ymin, 1e2)

    ax3t = ax3.twinx()
    ax3t.grid(False)
    ax3t.plot(qv[0:qlim], bq, color='pink', label='sample B(q)')
    ax3t.plot(qv, bq_, color='cyan', label='shpere B(q)')
    ax3t.plot(qv, rq_, color='yellow', label='sphere B(q)/A(q)')

    ymin, ymax = ax3t.get_ylim()
    ymax_ = -0.2*ymin + 1.2*ymax
    ax3t.set_ylim(ymin, ymax_)

    ax3t.plot([b1, b1], [ymin, ymax_], ':', color='red', label='boundary')
    ax3t.plot([b2, b2], [ymin, ymax_], ':', color='gray', label='extinction')

    ax3.legend(loc='upper right', fontsize=16)
    ax3t.legend(loc='lower left', fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()

def compute_bq_score(M, E, qv, ecurve, gslice, eslice, preRg):
    gf = gslice.start
    gt = gslice.stop
    aslice = slice(0, gt)
    Rg = compute_corrected_Rg_impl(qv, M[:,eslice], E[:,eslice], preRg, gf, gt)
    b1, b2, k = get_boundary_params_simple(Rg)
    L, C_, rscale, bscale, score = estimate_LR(qv, M[:,eslice], Rg, b1)
    return Rg, L, rscale, bscale, score
