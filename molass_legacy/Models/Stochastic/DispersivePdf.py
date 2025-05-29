"""
    Models.Stochastic.DispersivePdf.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.SecTheory.SecCF import dispersive_monopore
from molass_legacy.SecTheory.SecPDF import FftInvPdf
dispersive_monopore_pdf_impl = FftInvPdf(dispersive_monopore)

DEFUALT_TIMESCALE = 0.25    # 0.1 for FER_OA
N0 = 14400.0    # 48000*0.3 (30cm) or (t0/σ0)**2, see meeting document 20221104/index.html 

def dispersive_monopore_pdf(x, npi, tpi, N0, t0, timescale=DEFUALT_TIMESCALE):
    return timescale*dispersive_monopore_pdf_impl(timescale*x, npi, timescale*tpi, N0, timescale*t0)

def timescale_proof():
    import molass_legacy.KekLib.DebugPlot as plt
    N = 2000
    T = 0.5
    K = N*T
    me = 1.5
    mp = 1.5
    poresize = 75
    rgs = np.array([50, 40, 30, 20])
    rhov = rgs/poresize
    rhov[rhov > 1] = 1
    t0 = 500
    N0 = 14400
    t = np.arange(0, 1000)

    def compute_curves(x, N, T, x0, tI, N0, rhov, timescale=1):
        cy_list = []
        for rho in rhov:
            ni = N*(1 - rho)**me
            ti = T*(1 - rho)**mp
            cy = dispersive_monopore_pdf(x - tI, ni, ti, N0, x0 - tI, timescale=timescale)
            cy_list.append(cy)
        return cy_list

    rv = np.linspace(1, 70, 100)
    def compute_trv(N, T, poresize):
        rhov_ = rv/poresize
        rhov_[rhov_ > 1] = 1
        niv = N*(1 - rhov_)**me
        tiv = T*(1 - rhov_)**mp
        trv = t0 + niv*tiv
        return trv
    trv = compute_trv(N, T, poresize)

    def plot_curves(ax, x, x0, tI, title, with_arrow=True, timescale=1):
        ax.set_title(title + " with $t_{I}$=%g" % tI, fontsize=16)
        cy_list = compute_curves(x, N, T, x0, tI, N0, rhov, timescale=timescale)
        curves = []
        for k, (cy, rg) in enumerate(zip(cy_list, rgs)):
            curve, = ax.plot(x, cy, ':', label='$R_g$=%g' % rg)
            curves.append(curve)
        lines = []
        line = ax.axvline(x=x0, color='red', label='$t_0$')
        lines.append(line)
        line = ax.axvline(x=tI, color='gray', label='$t_I$')
        lines.append(line)
        if with_arrow:
            hy = np.average(ax.get_ylim())
            dx = x0 - tI
            ax.arrow(x=tI, y=hy, dx=dx, dy=0, width=0.005, head_width=0.015,
                    head_length=0.2*dx, length_includes_head=True, color='pink')
        ax.legend()

        axt = ax.twinx()
        axt.grid(False)
        excl_curve, = axt.plot(trv + tI, rv, color='yellow', label='exclusion curve')
        axt.legend(loc="center right")

        return curves, lines, excl_curve

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        fig.suptitle("SDM PDF Study in Experimental Time Axis with Poresize=%g, N*T=%g" % (poresize, K), fontsize=20)

        plot_curves(ax1, t, t0, 0, "Theoretical Time Axis", timescale=1)

        tI = -500
        x = t + tI
        x0 = t0 + tI
        plot_curves(ax2, x, x0, tI, "Experimental Time Axis", timescale=0.5)

        fig.tight_layout()

        ret = plt.show()
    return ret