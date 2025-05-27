"""
    SecTheory.TriporeSimulation.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
import sys
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.interpolate import UnivariateSpline

def demo():
    import molass_legacy.KekLib.DebugPlot as plt
    from SecTheory.SecCF import gec_monopore_phi, gec_dipore_phi, gec_tripore_phi
    from SecTheory.SecPDF import FftInvPdf
    from SecTheory.ColumnIllust import plot_column_illust, plot_column_legend

    monopore_pdf = FftInvPdf(gec_monopore_phi)
    dipore_pdf = FftInvPdf(gec_dipore_phi)
    tripore_pdf = FftInvPdf(gec_tripore_phi)

    x = np.arange(200)

    me = 0.1
    mp = 1.5
    rp0   =  76
    rp2_1 =  90
    rp2_2 =  40
    rp3_1 = 120
    rp3_2 =  76
    rp3_3 =  40

    rg = 35
    N = 171
    T = 0.63
    t0 = 20

    rg_array = np.array([70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 3, 1])
    rho0 = rg_array/rp0
    rho0[rho0 > 1] = 1
    np0 = N*(1 - rho0)**me
    tp0 = T*(1 - rho0)**mp

    rho2_1 = rg_array/rp2_1
    rho2_1[rho2_1 > 1] = 1
    np2_1 = N*(1 - rho2_1)**me
    tp2_1 = T*(1 - rho2_1)**mp

    rho2_2 = rg_array/rp2_2
    rho2_2[rho2_2 > 1] = 1
    np2_2 = N*(1 - rho2_2)**me
    tp2_2 = T*(1 - rho2_2)**mp

    rho3_1 = rg_array/rp3_1
    rho3_1[rho3_1 > 1] = 1
    np3_1 = N*(1 - rho3_1)**me
    tp3_1 = T*(1 - rho3_1)**mp

    rho3_2 = rg_array/rp3_2
    rho3_2[rho3_2 > 1] = 1
    np3_2 = N*(1 - rho3_2)**me
    tp3_2 = T*(1 - rho3_2)**mp

    rho3_3 = rg_array/rp3_3
    rho3_3[rho3_3 > 1] = 1
    np3_3 = N*(1 - rho3_3)**me
    tp3_3 = T*(1 - rho3_3)**mp

    y1_list = []
    confpoints1 = []
    y2_list = []
    confpoints2 = []
    y3_list = []
    confpoints3 = []

    k = 0
    for np0_, tp0_, np1_, tp1_, np2_, tp2_, np3_1_, tp3_1_, np3_2_, tp3_2_, np3_3_, tp3_3_ in zip(np0, tp0, np2_1, tp2_1, np2_2, tp2_2, np3_1, tp3_1, np3_2, tp3_2, np3_3, tp3_3):
        y1 = monopore_pdf(x,  np0_, tp0_, t0)
        y1_list.append(y1)
        j = np.argmax(y1)
        confpoints1.append((x[j], rg_array[k]))

        y2 = dipore_pdf(x, 0.5*np1_, tp1_, 0.5*np2_, tp2_, t0)
        y2_list.append(y2)
        j = np.argmax(y2)
        confpoints2.append((x[j], rg_array[k]))

        y3 = tripore_pdf(x, 1/3*np3_1_, tp3_1_, 1/3*np3_2_, tp3_2_, 1/3*np3_3_, tp3_3_, t0)
        y3_list.append(y3)
        j = np.argmax(y3)
        confpoints3.append((x[j], rg_array[k]))

        k += 1

    confpoints1 = np.array(confpoints1)
    confpoints2 = np.array(confpoints2)
    confpoints3 = np.array(confpoints3)

    def create_spline(confpoints):
        return UnivariateSpline(confpoints[:,0], confpoints[:,1], s=0)

    spline1 = create_spline(confpoints1)
    spline2 = create_spline(confpoints2)
    spline3 = create_spline(confpoints3)

    x_ = x[t0+8:130]

    with plt.Dp():
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9))
        ax1, ax2, ax3, ax4 = axes[0,:]

        fig.suptitle("Comparison of Conformance Curves among Monopore, Dipore and Tripore Columns", fontsize=20)
        ax1.set_title("Monopore Column with Poresize=%g" % rp0, fontsize=16)
        ax2.set_title("Dipore Column with Poresizes=%g, %g" % (rp2_1, rp2_2), fontsize=16)
        ax3.set_title("Tripore Column with Poresizes=%g, %g, %g" % (rp3_1, rp3_2, rp3_3), fontsize=16)
        ax4.set_title("Overlay of the Conformance Curves", fontsize=16)

        ylimits = []
        for ax, y_list, spline, confpoints, color1, color2, name in [
                    (ax1, y1_list, spline1, confpoints1, "yellow", "red", "Monopore"),
                    (ax2, y2_list, spline2, confpoints2, "cyan", "green", "Dipore"),
                    (ax3, y3_list, spline3, confpoints3, "orange", "blue", "Tripore")]:

            axt = ax.twinx()
            axt.grid(False)

            for y, rg_ in zip(y_list, rg_array):
                ax.plot(x, y, alpha=0.5, label="$R_g=%.3g$" % rg_)

            for k, ax_ in enumerate([axt, ax4]):
                label = None if k == 0 else name
                ax_.plot(x_, spline(x_), color=color1, lw=3, alpha=0.5, label=name)
                ax_.plot(*confpoints.T, "o", color=color2, alpha=0.5)

            # ax.legend()
            ylimits.append(ax.get_ylim())

        ax4.legend()
        ax4.set_xlim(0, 200)

        ylimits = np.array(ylimits)
        ymin = np.min(ylimits[:,0])
        ymax = np.max(ylimits[:,1])
        for ax in ax1, ax2, ax3:
            ax.set_ylim(ymin, ymax)

        plot_column_illust(axes[1,0], [rp0], ["white"])
        plot_column_illust(axes[1,1], [rp2_1, rp2_2], ["orange", "yellow"])
        plot_column_illust(axes[1,2], [rp3_1, rp3_2, rp3_3], ["pink", "white", "yellow"])
        plot_column_legend(axes[1,3], [rp3_1, rp2_1, rp3_2, rp3_3], ["pink", "orange", "white", "yellow"])

        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.dirname(this_dir)
    sys.path.append(lib_dir)

    import seaborn
    seaborn.set()
    import molass_legacy.KekLib

    demo()
