"""
    SecTheory.SecParamsPlot.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
# import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.ElutionCurveModels import egh
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

def compute_sec_peak_params(rp, tI, t0, P, m, rgs, Npc):
    rhos = rgs/rp
    rhos[rhos > 1] = 1
    trs = t0 + P*(1 - rhos)**m
    sigmas = (trs - tI)/np.sqrt(Npc)
    xlim = trs[0] - 8*sigmas[0], trs[-1] + 8*sigmas[-1]
    x = np.linspace(tI, t0+P, 2000)
    return trs, sigmas, xlim, x

def draw_sec_curve(ax, rp, tI, t0, P, m, rgs, Npc, plot_tI=False):
    ax.set_ylabel("$R_g$")

    trs, sigmas, xlim, x = compute_sec_peak_params(rp, tI, t0, P, m, rgs, Npc)

    r = np.flip(np.arange(0, rp*0.9, 0.1))
    t = t0 + P*np.power(1 - r/rp, m)
    ax.plot(t, r, color="cyan", markersize=1, lw=3, label=r"$t_R = t_0 + P \cdot (1-\rho)^m; \;\; r_p=%.3g, \; m=%.3g$" % (rp, m))

    t_ = t0 + P*np.power(1 - rgs/rp, m)
    ax.plot(t_, rgs, "o", color="red", label="elution points")

    m = np.argmax(sigmas)
    smax = sigmas[m]
    if plot_tI:
        xmin = min(tI - smax*2, xlim[0])
    else:
        xmin = min(t0 - smax*2, xlim[0])
    xmax = max(t0 + P + smax*2, xlim[1])

    ax.set_xlim(xmin, xmax)
    dx = (xmax - xmin)*0.05
    ymin, ymax = ax.get_ylim()
    ymin = min(0, ymin)
    ax.set_ylim(ymin, ymax)
    ty = ymin*0.7 + ymax*0.3
    label_fontsize = 14

    tL = t0 + P
    if plot_tI:
        point_list = [(tI, "$t_I$")]
    else:
        point_list = []

    k = 0
    for px, label in point_list + [(t0, "$t_0$"), (t0 + P, "$t_0 + P$")]:
        ax.plot([px, px], [ymin, ymax], ":")
        dx_ = dx if k < 1 and px < trs[0] - smax*2 else -dx
        ax.annotate(label, xy=(px, 0), xytext=(px + dx_, ty), ha='center',
            arrowprops=dict(arrowstyle="-", color='k'), fontsize=label_fontsize )
        k += 1

    ax.legend(bbox_to_anchor=(0.7, 1), loc="upper center")

def plot_sec_params_impl(fig, ax, optimizer, new_params, lrf_info):
    from Experiment.ColumnTypes import get_columnname

    in_folder = get_in_folder()
    ax.set_title("Simple Exclusion Curve on %s with %s" % (in_folder, get_columnname()), fontsize=20)
    ax.set_ylabel("Intensity")

    if optimizer.rg_curve.__class__.__name__.find("Dummy") >= 0:
        # 'DummyRgCurve' object has no attribute 'segments'
        axt = None
    else:
        axt = ax.twinx()
        axt.grid(False)
    axis_info = [fig, (None, ax, None, axt)]    # fig, (ax1, ax2, ax3, axt)
    optimizer.objective_func(new_params, plot=True, axis_info=axis_info)

    rg_params = optimizer.separate_params[2]
    print("rg_params=", rg_params)

    seccol_params = optimizer.separate_params[7]
    print("seccol_params=", seccol_params)

    Npc, rp, tI, t0, P, m = seccol_params

    if axt is None:
        axt = ax.twinx()
        axt.grid(False)

    draw_sec_curve(axt, rp, tI, t0, P, m, rg_params, Npc)


def plot_sec_params(optimizer, params, lrf_info, devel=True):
    if devel:
        from importlib import reload
        import SecTheory.TriporeExclCurve
        reload(SecTheory.TriporeExclCurve)
        import SecTheory.MonoporeExclCurve
        reload(SecTheory.MonoporeExclCurve)
        import SecTheory.OligoporeExclCurve
        reload(SecTheory.OligoporeExclCurve)
    from SecTheory.TriporeExclCurve import estimate_tripore_exclcurve
    from SecTheory.MonoporeExclCurve import estimate_monopore_exclcurve
    from SecTheory.OligoporeExclCurve import estimate_polypore_exclcurve

    new_params = lrf_info.update_optimizer(optimizer, params)

    with plt.Dp():
        fig = plt.figure(figsize=(20,12))
        gs = GridSpec(3,4)
        ax1 = fig.add_subplot(gs[0,0:3])

        ax2 = fig.add_subplot(gs[1,0:3])
        ax2psd = fig.add_subplot(gs[1,3])

        ax3 = fig.add_subplot(gs[2,0:3])
        ax3psd = fig.add_subplot(gs[2,3])

        plot_sec_params_impl(fig, ax1, optimizer, new_params, lrf_info)
        estimate_monopore_exclcurve(fig, ax2, ax2psd, optimizer, lrf_info, new_params)
        # estimate_tripore_exclcurve(fig, ax3, ax3psd, optimizer, lrf_info, new_params)
        estimate_polypore_exclcurve(fig, ax3, ax3psd, optimizer, lrf_info, new_params)

        xlims = []
        for ax in ax1, ax2, ax3:
            xlims.append(ax.get_xlim())
        xlims = np.array(xlims)
        xmin = np.min(xlims[:,0])
        xmax = np.max(xlims[:,1])
        for ax in ax1, ax2, ax3:
            ax.set_xlim(xmin, xmax)

        fig.tight_layout()
        plt.show()
