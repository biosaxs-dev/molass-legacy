"""
    ErrorMonteCarlo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
import numpy as np
from bisect import bisect_right
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
from matplotlib.patches import Rectangle
import molass_legacy.KekLib.DebugPlot as plt
from Theory.SolidSphere import phi
from time import time

def demo_impl(sd, f, t):
    from DataUtils import get_in_folder
    from .NaiiveLrfSolver import NaiiveLrfSolver
    from molass_legacy.DENSS.DenssUtils import fit_data
    if False:
        from importlib import reload
        import BoundedLRF.BoundedLrfSolver
        reload(BoundedLRF.BoundedLrfSolver)
    from .BoundedLrfSolver import BoundedLrfSolver

    print("demo_impl")

    D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
    x = xr_curve.x
    y = xr_curve.y
    data_title = get_in_folder()

    range_ = slice(f, t+1)

    D_, E_ = D[:,range_], E[:,range_]

    i = bisect_right(qv, 0.02)
    cy = D_[i,:]
    j = np.argmax(cy)

    print("len(cy)=", len(cy), "j=", j)
    pv1 = D_[:,j]
    # pv1 = pv1/np.max(pv1)

    solver = NaiiveLrfSolver(qv, D_, E_)
    P, C = solver.solve()
    c1, c2 = C[:,j]
    aq, bq = P.T
    aq1 = aq*c1
    bq1 = bq

    naive_simulation = False
    if naive_simulation:
        P_ = P
        bq_bounds_ = None
    else:
        solver = BoundedLrfSolver(qv, D_, E_)
        P_, C_, Rg, R_, L_, hK, hL, bq_bounds_, coerced_bq_ = solver.solve(debug=False)
        aq_, bq_ = P_.T

    Dinv = np.linalg.pinv(D_)

    def simulate_Pe():
        t0 = time()
        P_results = []
        for n in range(1000):
            if n % 10 == 0:
                print([n], "simulating")
            Dp = D_ + E_ * np.random.normal(0, 1, E_.shape)
            if naive_simulation:
                solver = NaiiveLrfSolver(qv, Dp, E_)
                P_, C_ = solver.solve()
            else:
                solver = BoundedLrfSolver(qv, Dp, E_)
                P_, C_, Rg, R_, L_, hK, hL, bq_bounds_, coerced_bq_ = solver.solve(debug=False)
            P_results.append(P_)
        Pe = np.std(P_results, axis=0)
        print("It took", time() - t0)
        return Pe

    def plot_diff(ax1, ax2, ax3, P, Pe=None, bounds=None):
        a = P[:,0]
        b = P[:,1]

        if Pe is None:
            W = Dinv @ P
            Pe = np.sqrt(np.dot(E_**2, W**2))

        ae = Pe[:,0]
        qc, Ic, Icerr, Dc = fit_data(qv, a, ae)
        spline = UnivariateSpline(qc, Ic, s=0, ext=3)
        delta = (a - spline(qv))/ae

        ax1.plot(qv, a, label="LRF A(q)")
        ax1.plot(qc, Ic, label="IFT fit")
        ax1.legend()
        ax2.plot(qv, b, color="pink", label="LRF B(q)")
        if bounds is not None:
            for k, bound in enumerate(bounds):
                label = "B(q) Bounds" if k == 0 else None
                ax2.plot(qv, bound, ":", color="red", label=label)
        ax2.legend()

        ax3.plot(qv, delta, "o", markersize=3)

    bounded_lef_simulation = False

    with plt.Dp():
        from matplotlib.gridspec import GridSpec
        from DataUtils import get_in_folder

        gs = GridSpec(4,2)

        fig = plt.figure(figsize=(12,8))
        fig.suptitle("$\Delta I/\sigma$ Comparison for %s" % get_in_folder(), fontsize=20)
        ax1 = fig.add_subplot(gs[0:2,0])
        ax2 = fig.add_subplot(gs[2,0])
        ax3 = fig.add_subplot(gs[3,0])
        ax1.set_yscale("log")
        ax1.set_title("Naive LRF (propagated error)", fontsize=16)

        ax4 = fig.add_subplot(gs[0:2,1])
        ax5 = fig.add_subplot(gs[2,1])
        ax6 = fig.add_subplot(gs[3,1])
        ax4.set_yscale("log")
        if naive_simulation:
            title = "Naive LRF (simulated error)"
        else:
            if bounded_lef_simulation:
                title = "Bounded LRF (simulated error)"
            else:
                title = "Bounded LRF (propagated error)"
        ax4.set_title(title, fontsize=16)

        plot_diff(ax1, ax2, ax3, P)
        if bounded_lef_simulation:
            Pe = simulate_Pe()
        else:
            Pe = None
        plot_diff(ax4, ax5, ax6, P_, Pe=Pe, bounds=bq_bounds_)

        ylim1 = ax1.get_ylim()
        ylim4 = ax4.get_ylim()
        ymin = min(ylim1[0], ylim4[0])
        ymax = max(ylim1[1], ylim4[1])
        for ax in ax1, ax4:
            ax.set_ylim(ymin, ymax)

        ax5.set_ylim(ax2.get_ylim())
        ax6.set_ylim(ax3.get_ylim())

        fig.tight_layout()
        plt.show()

def demo(caller):
    dialog = caller.dialog
    pdata, popts = caller.get_preview_data(with_update=False)

    print("pdata.cnv_ranges=", pdata.cnv_ranges)
    paired_range = pdata.cnv_ranges[0]
    ranges = paired_range.get_fromto_list()
    print(ranges)
    f, t = ranges[0]
    sd = dialog.sd
    demo_impl(sd, f, t)
