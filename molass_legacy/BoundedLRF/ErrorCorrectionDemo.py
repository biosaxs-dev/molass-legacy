"""
    ErrorCorrectionDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
import numpy as np
from scipy.optimize import minimize
from bisect import bisect_right
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
from matplotlib.patches import Rectangle
import molass_legacy.KekLib.DebugPlot as plt
from Theory.SolidSphere import phi
from time import time

# SAVE_FOLDER = r"D:\TODO\20230710\temp"
SAVE_FOLDER = None

def save_aq(file, qv, P, Pe):
    if SAVE_FOLDER is not None:
        path = os.path.join(SAVE_FOLDER, file)
        np.savetxt(path, np.array([qv, P[:,0], Pe[:,0]]).T)

def demo_impl(sd, f, t):
    from DataUtils import get_in_folder
    from molass_legacy.DENSS.DenssUtils import fit_data
    from .NaiiveLrfSolver import NaiiveLrfSolver
    from .BoundedLrfSolver import BoundedLrfSolver
    from importlib import reload
    import BoundedLRF.ErrorCorrection
    reload(BoundedLRF.ErrorCorrection)
    from .ErrorCorrection import compute_corrected_error

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

    Dinv = np.linalg.pinv(D_)
    W = Dinv @ P
    Pe = np.sqrt(np.dot(E_**2, W**2))

    save_aq("aq-naive.dat", qv, P, Pe)

    solver = BoundedLrfSolver(qv, D_, E_)
    P_, C_, Rg, R_, L_, hK, hL, bq_bounds_, coerced_bq_ = solver.solve(debug=False)
    aq_, bq_ = P_.T

    W = Dinv @ P_
    Peb = np.sqrt(np.dot(E_**2, W**2))

    save_aq("aq-bounded-linear-error.dat", qv, P_, Peb)

    Peb_, adjust_scale = compute_corrected_error(qv, i, P, c1, c2, P_, Pe, bq_bounds_, coerced_bq_, debug=True)

    save_aq("aq-bounded-corrected-error.dat", qv, P_, Peb_)

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

    with plt.Dp():
        from matplotlib.gridspec import GridSpec
        from DataUtils import get_in_folder

        gs = GridSpec(4,3)

        fig = plt.figure(figsize=(18,8))
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
        title = "Bounded LRF (propagated error)"
        ax4.set_title(title, fontsize=16)

        ax7 = fig.add_subplot(gs[0:2,2])
        ax8 = fig.add_subplot(gs[2,2])
        ax9 = fig.add_subplot(gs[3,2])
        ax7.set_yscale("log")
        title = "Bounded LRF (corrected error: adjust_scale=%.3g)" % adjust_scale
        ax7.set_title(title, fontsize=16)

        plot_diff(ax1, ax2, ax3, P, Pe=Pe)
        plot_diff(ax4, ax5, ax6, P_, Pe=Peb, bounds=bq_bounds_)
        plot_diff(ax7, ax8, ax9, P_, Pe=Peb_, bounds=bq_bounds_)

        ylim1 = ax1.get_ylim()
        ylim4 = ax4.get_ylim()
        ymin = min(ylim1[0], ylim4[0])
        ymax = max(ylim1[1], ylim4[1])
        for ax in ax1, ax4, ax7:
            ax.set_ylim(ymin, ymax)

        for ax in ax5, ax8:
            ax.set_ylim(ax2.get_ylim())

        for ax in ax6, ax9:
            ax.set_ylim(ax3.get_ylim())

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
