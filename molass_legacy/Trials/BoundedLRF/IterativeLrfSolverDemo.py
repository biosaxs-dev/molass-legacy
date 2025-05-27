"""
    IterativeLrfSolverDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from matplotlib.patches import Rectangle
import molass_legacy.KekLib.DebugPlot as plt

def demo_impl(D, E, qv, x, y, f, t, data_title):
    from SvdDenoise import get_denoised_data
    from SimpleGuinier import SimpleGuinier
    from importlib import reload
    import BoundedLRF.IterativeLrfSolver
    reload(BoundedLRF.IterativeLrfSolver)
    from .IterativeLrfSolver import IterativeLrfSolver

    i = bisect_right(qv, 0.02)
    cy = D[i,:]
    j = np.argmax(cy)
    pv1 = D[:,j]
    pv1 = pv1/np.max(pv1)

    range_ = slice(f, t+1)

    solver = IterativeLrfSolver(qv, D[:,range_], E[:,range_])
    P, C, Rg, R, L, bq_bounds, coerced_bq, changes = solver.solve(maxiter=100000, L=0.4)

    with plt.Dp():
        cx = np.arange(len(changes))*100
        fig, ax = plt.subplots()
        ax.plot(cx, changes)
        fig.tight_layout()
        plt.show()

    aq, bq = P.T
    c1, c2 = C[:,j-f]
    aq_ = aq*c1
    bq_ = bq*c2
    coerced_bq *= c2

    with plt.Dp():
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18,10))
        ax1, ax2, ax3 = axes[0,:]
        fig.suptitle("Bounnded LRF Solver Demo for %s" % data_title, fontsize=20)
        ax1.set_title("Elution and Range", fontsize=16)
        ax2.set_title("LRF Result (Linear)", fontsize=16)
        ax3.set_title("LRF Result (Log)", fontsize=16)
        ax1.plot(x, y)
        ymin, ymax = ax1.get_ylim()
        p = Rectangle(
                (f, ymin),  # (x,y)
                t - f,   # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        ax1.add_patch(p)
        ax3.set_yscale('log')
        for ax in [ax2, ax3]:
            for k, pv in enumerate([pv1] + list(P.T)):
                if k < 2:
                    color = None
                    alpha = 1
                else:
                    color = "pink"
                    alpha = 0.5
                ax.plot(qv, pv, color=color, alpha=alpha)

        ax2_ = axes[1,1]
        ax2_.set_title("Bound-modified B(q)", fontsize=16)

        ax2_.plot(qv, aq_)
        ax2_.plot(qv, coerced_bq, color="pink", alpha=0.5)

        for ax in [ax2, ax2_]:
            for k, bv in enumerate(bq_bounds):
                label = "Bound L=%g" % L if k == 0 else None
                ax.plot(qv, bv, color="red", label=label, alpha=0.5)

            ax.set_ylim(-5, 5)
            ax.legend(fontsize=16)

        for ax in axes[1,0],axes[1,2]:
            ax.grid(False)
            ax.set_axis_off()

        fig.tight_layout()
        plt.show()

def demo(dialog, pdata, popts):
    from DataUtils import get_in_folder

    sd = dialog.sd
    D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
    x = xr_curve.x
    y = xr_curve.y
    print("pdata.cnv_ranges=", pdata.cnv_ranges)
    paired_range = pdata.cnv_ranges[1]
    ranges = paired_range.get_fromto_list()
    print(ranges)
    f, t = ranges[1]
    data_title = get_in_folder()

    demo_impl(D, E, qv, x, y, f, t, data_title)

def demo2(caller):
    from importlib import reload

    import Tools.EmbedCushionUtils
    reload(Tools.EmbedCushionUtils)
    from molass_legacy.Tools.EmbedCushionUtils import get_caller_attr

    gi_in_folder = r"D:\PyTools\Data\20180526\GI"
    cushion_dict = get_caller_attr(caller, "cushion_dict", {})
    gi_sd = cushion_dict.get("gi_sd")
    if gi_sd is None:
        from molass_legacy.Tools.SdUtils import get_sd
        cushion_dict["gi_sd"] = gi_sd = get_sd(gi_in_folder)

    from DataUtils import get_in_folder

    sd = gi_sd
    D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
    x = xr_curve.x
    y = xr_curve.y

    data_title = get_in_folder(gi_in_folder)
    f = 130
    t = 200
    demo_impl(D, E, qv, x, y, f, t, data_title)

def demo3(dialog, pdata, popts):
    from DataUtils import get_in_folder
    from importlib import reload
    import Trials.BoundedLRF.IterativeLrfSolver
    reload(Trials.BoundedLRF.IterativeLrfSolver)
    from .IterativeLrfSolver import IterativeLrfSolver

    sd = dialog.sd
    D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
    x = xr_curve.x
    y = xr_curve.y
    print("pdata.cnv_ranges=", pdata.cnv_ranges)
    paired_range = pdata.cnv_ranges[1]
    ranges = paired_range.get_fromto_list()
    print(ranges)
    f, t = ranges[1]
    data_title = get_in_folder()

    i = bisect_right(qv, 0.02)
    cy = D[i,:]
    j = np.argmax(cy)
    pv1 = D[:,j]
    # pv1 = pv1/np.max(pv1)

    range_ = slice(f, t+1)

    # L = 0.4
    L = 1
    bq_changes = []

    solver = IterativeLrfSolver(qv, D[:,range_], E[:,range_])
    P, C, Rg, R, L, bq_bounds, coerced_bq, w, changes = solver.solve(maxiter=1, L=L)
    c1, c2 = C[:,j-f]
    aq, bq = P.T
    aq1 = aq*c1
    bq1 = bq
    bq_changes.append((C, bq1, coerced_bq, bq_bounds, w))

    P, C, Rg, R, L, bq_bounds, coerced_bq, w, changes = solver.solve(maxiter=100, L=L)
    c1, c2 = C[:,j-f]
    aq, bq = P.T
    aq_ = aq*c1
    bq_ = bq
    bq_changes.append((C, bq_, coerced_bq, bq_bounds, w))

    P, C, Rg, R, L, bq_bounds, coerced_bq, w, changes = solver.solve(maxiter=1000, L=L)
    c1, c2 = C[:,j-f]
    aq, bq = P.T
    aq_ = aq*c1
    bq_ = bq
    bq_changes.append((C, bq_, coerced_bq, bq_bounds, w))

    P, C, Rg, R, L, bq_bounds, coerced_bq, w, changes = solver.solve(maxiter=100000, L=L)
    c1, c2 = C[:,j-f]
    aq, bq = P.T
    aq_ = aq*c1
    bq_ = bq
    bq_changes.append((C, bq_, coerced_bq, bq_bounds, w))

    with plt.Dp():
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20,10))
        ax1, ax2, ax3, ax4 = axes[0,:]
        fig.suptitle("Bounnded LRF Solver Demo for %s with L=%g" % (data_title, L), fontsize=20)
        ax1.set_title("Elution Data and Range", fontsize=16)
        ax2.set_title("LRF Result (Linear)", fontsize=16)
        ax3.set_title("LRF Result (Log)", fontsize=16)
        ax4.set_title("LRF Result (Guinier Plot)", fontsize=16)
        ax1.plot(x, y)
        ymin, ymax = ax1.get_ylim()
        p = Rectangle(
                (f, ymin),  # (x,y)
                t - f,   # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        ax1.add_patch(p)

        pv1 = pv1/pv1[i]
        aq1 = aq1/aq1[i]
        aqf = aq_/aq_[i]

        ax3.set_yscale("log")
        for ax in ax2, ax3:
            ax.plot(qv, pv1, label="data")
            ax.plot(qv, aq1, label="LRF first A(q)", alpha=0.5)
            ax.plot(qv, aqf, label="LRF final A(q)", alpha=0.5)
            ax.legend()
            axt = ax.twinx()
            axt.grid(False)
            axt.plot(qv, w, ":", color="cyan")

        glim_i = bisect_right(qv, 1.5/Rg)
        gslice = slice(0,glim_i)
        qv2 = qv[gslice]**2

        ax4.plot(qv2, np.log(pv1[gslice]), label="data")
        ax4.plot(qv2, np.log(aq1[gslice]), label="LRF first A(q)", alpha=0.5)
        ax4.plot(qv2, np.log(aqf[gslice]), label="LRF final A(q)", alpha=0.5)
        ax4.legend()

        axc_ylims = []
        ax_ylims = []
        x_ = x[range_]
        for axc, ax, (C, bq_, coerced_bq, bq_bounds, w) in zip(axes[1,:], axes[2,:], bq_changes):
            axc.plot(x, y)
            for k, cv in enumerate(C):
                color = "pink" if k == 1 else None
                axc.plot(x_, cv, color=color)
            axc_ylims.append(axc.get_ylim())
            ax.plot(qv, bq_, color="pink", label="B(q)")
            ax.plot(qv, coerced_bq, color="yellow", alpha=0.5, label="coerced B(q)")
            ax.plot(qv, bq_bounds[0], ":", color="red", alpha=0.5, label="lower bound")
            ax.plot(qv, bq_bounds[1], ":", color="red", alpha=0.5, label="upper bound")
            ax.set_ylim(-2, 2)
            ax.legend()
            ax_ylims.append(ax.get_ylim())

        for axes_, ylims in [(axes[1,:], axc_ylims), (axes[2,:], ax_ylims)]:
            ylims = np.array(ylims)
            ymin = np.min(ylims[:,0])
            ymax = np.max(ylims[:,1])
            for ax in axes_:
                ax.set_ylim(ymin, ymax)

        fig.tight_layout()
        plt.show()
