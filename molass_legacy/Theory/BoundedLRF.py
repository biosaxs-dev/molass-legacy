"""
    Theory.BoundedLRF.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from scipy.optimize import minimize
from matplotlib.patches import Rectangle
from time import time
from molass_legacy.KekLib.SciPyCookbook import smooth
import molass_legacy.KekLib.DebugPlot as plt
from SvdDenoise import get_denoised_data
from .SolidSphere import phi
from .Rg import compute_corrected_Rg
from .SphericalFit import estimate_LR
from .SolidSphere import get_boundary_params_simple
from Rank.Synthesized import synthesized_data

def fit_exp_dist(p, py):
    apy = np.abs(py)

    def obj_func(x):
        K, L, M = x
        v = K*L*np.exp(-L*M*p)
        return np.sum((v - apy)**2)

    res = minimize(obj_func, (1,1,1))

    return res.x

def spike():

    qv = np.linspace(0.005, 0.5, 100)
    R = np.sqrt(5/3)*30
    K = 5

    n = int(R*K*qv[-1]/(2*np.pi))
    print('n=', n)
    pp = np.arange(1, n)*2*np.pi/(R*K)
    pn = (np.arange(1, n)*2-1)*np.pi/(R*K)

    Aq = phi(qv, R)**2
    Sf_1 = -phi(qv, K*R)
    Bq = Aq*Sf_1

    ppy = -phi(pp, K*R)
    pny = -phi(pn, K*R)

    show_fn_bound = False

    if show_fn_bound:
        K, L, M = fit_exp_dist(pn, pny)
        print('K,L,M=', K, L, M)
        ey = K*L*np.exp(-L*M*qv)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16,7))

    for i, ax in enumerate([ax1, ax2]):

        ax.set_title("B(q)/A(q) Upper and Lower Bounds" + ("" if i==0 else " (Zoomed)"), fontsize=16)
        ax.set_yscale('log')

        if i == 0:
            ax.plot(qv, Aq)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
        else:
            ax.set_xlim(xmin, xmin*(1-0.25) + xmax*0.25)
            ax.set_ylim(np.exp(np.log10(ymin)*(1-0.75) + np.log10(ymax)*0.75), ymax)
            # ax.set_axis_off()

        axt = ax.twinx()
        axt.grid(False)
        axt.plot(qv, Sf_1, color='yellow', label='B(q)/A(q)')
        axt.plot(qv, Bq, color='cyan', label='B(q)')

        axt.plot(pp, ppy, 'o:', color='red', label='upper bound')
        axt.plot(pn, pny, 'o:', color='blue', label='lower bound')

        if show_fn_bound:
            for j, y in enumerate([ey, -ey]):
                axt.plot(qv, y, ':', color='green', label='functional bound' if j==0 else None)

        ymint, ymaxt = axt.get_ylim()
        axt.set_ylim(ymint, ymaxt)

        b = pn[1]
        axt.plot([b, b], [ymint, ymaxt], ':', color='gray', label='rank boundary')

        axt.legend(loc='lower center' if i==0 else 'lower right')

    fig.tight_layout()
    plt.show()

def plot_P(ax, qv, pt, P):
    aq = P[:,0]
    bq = P[:,1]

    aq_ = aq/np.max(aq)

    ax.set_yscale('log')
    axt = ax.twinx()

    pt_ = pt/np.max(pt)

    ax.plot(qv, pt_, label='peak top')
    ax.plot(qv, aq_, label='A(q)')
    axt.plot(qv, bq, label='B(q)', color='pink')

    ax.legend(fontsize=16)
    axt.legend(bbox_to_anchor=(1, 0.82), loc='upper right')

MINBOUND = 1e-8

def bounds(qv, aq, starv, px, py):

    def minbound(v):
        return max(MINBOUND, v) if v > 0 else min(-MINBOUND, v)

    i = 0
    j = 0
    bv = np.zeros(len(qv))
    for k, x in enumerate(px):
        while qv[j] < x:
            j +=1

        # print([k], x, (i, j))
        if i == 0:
            bv[i:j] = starv
        else:
            n = j - i
            w = np.arange(n)/n
            bv[i:j] = minbound(aq[i] * py[k-1]) * (1-w) + minbound(aq[j] * py[k]) * w

        i = j
    if j < len(bv):
        bv[j:] = minbound(aq[i] * py[k])

    return bv

def construct_bounds(qv, R, P, rscale, bscale, debug=False):
    n = int(R*rscale*qv[-1]/(2*np.pi))
    print('n=', n)

    pp = np.arange(1, n)*2*np.pi/(R*rscale)
    pn = (np.arange(1, n)*2-1)*np.pi/(R*rscale)

    ppy = -phi(pp, rscale*R)
    pny = -phi(pn, rscale*R)

    print(np.min(ppy), np.max(pny))

    # aq = P[:,0]
    aq = phi(qv, R)**2
    lowerv = bounds(qv, aq, -np.inf, pn, pny)
    upperv = bounds(qv, aq, +np.inf, pp, ppy)

    if debug:
        Aq = phi(qv, R)**2
        Sf_1 = -phi(qv, rscale*R)
        aq = P[:,0]
        bq = P[:,1]
        minaq = np.percentile(aq, 10)
        aq_ = np.max([aq, np.ones(len(aq))*minaq], axis=0)
        sf_1 = bscale*bq/aq_

        print('bscale=', bscale)
        plt.push()
        fig, axes = plt.subplots(ncols=2, figsize=(16,7))

        for i, ax in enumerate(axes):
            ax.plot(qv, bq, color='pink')
            ax.plot(qv, Sf_1, ':', color='green')
            # ax.plot(qv, sf_1, color='yellow')
            if i == 0:
                ax.plot(pp, ppy, 'o:', color='red', label='upper bound')
                ax.plot(pn, pny, 'o:', color='blue', label='lower bound')
            else:
                ax.plot(qv, upperv, ':', color='red', label='upper bound')
                ax.plot(qv, lowerv, ':', color='blue', label='lower bound')

        fig.tight_layout()
        plt.show()
        plt.pop()

    return lowerv, upperv

def optimize_impl(P, C, M, lbound, ubound, boundary_info=None):

    # method_opts = ['Nelder-Mead', False]  # it takes too long
    # method_opts = ['Powell',      True]   # it takes too long
    method_opts = ['CG', False]             #
    # method_opts = ['BFGS', False]         # Method BFGS cannot handle constraints nor bounds.

    method, bounds_ok = method_opts

    print('method=', method, bounds_ok)

    t0 = time()

    xinit = P.flatten()
    isize = P.shape[0]

    if bounds_ok:
        def obj_func(x):
            P_ = x.reshape(P.shape)
            return np.linalg.norm(P_@C - M)
        lbound_ = np.hstack([-np.ones(isize)*np.inf, lbound])
        ubound_ = np.hstack([+np.ones(isize)*np.inf, ubound])
        res = minimize(obj_func, xinit, bounds=np.vstack([lbound_, ubound_]).T, method=method)

    else:
        assert boundary_info is not None

        if False:
            qv, k, b = boundary_info

            boundary_i = bisect_right(qv, b)
            print('b=', b, 'boundary_i=', boundary_i)
            bslice = slice(boundary_i, None)
            zeros = np.zeros(isize-boundary_i)
            lbound_ = lbound[bslice]
            ubound_ = ubound[bslice]

            def obj_func_regular1(x):
                P_ = x.reshape(P.shape)
                norm = np.linalg.norm(P_@C - M)
                bq = P_[bslice,1]
                penalty = ( np.sum(np.min([bq - lbound_, zeros], axis=0)**2)
                          + np.sum(np.max([bq - ubound_, zeros], axis=0)**2)
                          )
                return norm + penalty*10

            res = minimize(obj_func_regular1, xinit, method=method)
        else:
            qv, k, b = boundary_info

            zeros = np.zeros(isize)
            penalty_weights_logistic = 1/(1+np.exp(-k*(qv-b)))
            if False:
                denom = 1/np.max([np.abs(lbound), np.abs(ubound)], axis=0)
                normal_denom = denom/np.sum(denom)
                penalty_weights_total = penalty_weights_logistic*normal_denom
            penalty_weights_total = penalty_weights_logistic

            if True:
                plt.push()
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16,7))
                ax1.plot(qv, lbound)
                ax1.plot(qv, ubound)
                ax2.plot(qv, penalty_weights_logistic)
                # ax2.plot(qv, normal_denom)
                # ax2.plot(qv, penalty_weights_total)
                fig.tight_layout()
                plt.show()
                plt.pop()

            def obj_func_regular2(x):
                P_ = x.reshape(P.shape)
                norm = np.linalg.norm(P_@C - M)
                bq = P_[:,1]
                penalty = ( np.sum(np.min([bq - lbound, zeros]*penalty_weights_total, axis=0)**2)
                          + np.sum(np.max([bq - ubound, zeros]*penalty_weights_total, axis=0)**2)
                          )
                return norm + penalty*100

            res = minimize(obj_func_regular2, xinit, method=method)

    Popt = res.x.reshape(P.shape)

    print('res.success=', res.success)
    print('res.status=', res.status)
    print('res.nit=', res.nit)

    print("It took", time()-t0)

    return Popt

def demo(root, sd, pno=0, synthesized_lrf=None, bounded_lrf=True, debug=True):
    M, E, qv, ecurve = sd.get_xr_data_separate_ly()

    range_ = ecurve.get_ranges_by_ratio(0.5)[pno]
    f = range_[0]
    p = range_[1]
    t = range_[2]

    pt = M[:,p]

    x = ecurve.x
    y = ecurve.y

    i = bisect_right(qv, 0.02)

    eslice = slice(f,t)
    M0 = M[:,eslice]
    Rg, gf, gt, _ = compute_corrected_Rg(sd, ecurve, pno, qv, M0, E[:,eslice])

    norm_list = []

    c_ = y[eslice]

    M2 = get_denoised_data(M0, rank=2)
    C2 = np.array([c_, c_**2])
    P2 = M2 @ np.linalg.pinv(C2)

    norm_list.append(np.linalg.norm(P2@C2 - M0))

    b1, b2, k = get_boundary_params_simple(Rg)
    L, _, rscale, bscale, score = estimate_LR(qv, M[:,eslice], Rg, b1, debug=False)

    M_, _ = synthesized_data(qv, M[:,eslice], Rg, rank=2, cd=2, boundary=b1, k=k)
    P_ = M_ @ np.linalg.pinv(C2)

    norm_list.append(np.linalg.norm(P_@C2 - M0))

    if bounded_lrf:
        R = np.sqrt(5/3)*Rg
        lbound, ubound = construct_bounds(qv, R, P_, rscale, bscale, debug=True)

        Popt_ = optimize_impl(P_, C2, M2, lbound, ubound, boundary_info=(qv, k, b1))
        norm_list.append(np.linalg.norm(Popt_@C2 - M0))
    else:
        Popt_ = None

    if synthesized_lrf is None:
        Popt = Popt_

        ax3_tilte = "Bounded LRF"
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(21,11))

        ax1, ax2, ax3 = axes[0,:]
        ax4, ax5, ax6 = axes[1,:]
    else:
        Popt = synthesized_lrf(qv, M[:,eslice], c_, M2, P2, boundary=b1, k=k)
        norm_list.append(np.linalg.norm(Popt@C2 - M0))

        ax3_tilte = "Result Synthesized LRF"
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(22,10))

        ax1, ax2, ax3 = axes[0,0:3]
        ax4, ax5, ax6 = axes[1,0:3]
        axu, axd = axes[:,3]
        axu.set_title("Bounded LRF", fontsize=20)

    ax1.set_title("Rank(2,2) LRF", fontsize=20)
    ax2.set_title("Data Synthesized LRF", fontsize=20)
    ax3.set_title(ax3_tilte, fontsize=20)

    if False:
        ax1.plot(x, y, color='orange')
        ymin, ymax = ax1.get_ylim()
        p = Rectangle(
                (f, ymin),      # (x,y)
                t - f,          # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        ax1.add_patch(p)

    plot_P(ax1, qv, pt, P2)
    plot_P(ax2, qv, pt, P_)
    plot_P(ax3, qv, pt, Popt)
    axes_ = [ax2, ax3]
    if bounded_lrf:
        plot_P(axu, qv, pt, Popt_)
        axes_.append(axu)

    for ax in axes_:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
        ax.plot([b1, b1], [ymin, ymax], ':', label='rank boundary', color='gray')
        ax.legend(fontsize=16)

    ax4.plot(qv, P2[:,1]/P2[:,0], label='B(q)/A(q)')
    ax5.plot(qv, P_[:,1]/P_[:,0], label='B(q)/A(q)')
    ax6.plot(qv, Popt[:,1]/Popt[:,0], label='B(q)/A(q)')
    axes_ = [ax4, ax5, ax6]
    if bounded_lrf:
        axd.plot(qv, Popt_[:,1]/Popt_[:,0], label='B(q)/A(q)')
        axes_.append(axd) 

    for ax in axes_:
        ax.set_ylim(-2, 2)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax)
        ax.plot([xmin, xmax], [0, 0], ':', color='red')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
        ax.plot([b1, b1], [ymin, ymax], ':', label='rank boundary', color='gray')
        ax.legend(loc='upper left', fontsize=16)

    fig.tight_layout()

    print('norm_list=', norm_list)
    plt.show()
