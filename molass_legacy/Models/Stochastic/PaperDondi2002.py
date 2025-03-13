"""
    Models.Stochastic.PaperDondi2002.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def demo_monopore():
    from SecTheory.BasicModels import robust_single_pore_pdf_scaled
    from SecTheory.SecCF import gec_monopore_phi, dispersive_monopore
    from SecTheory.SecPDF import FftInvPdf
    from HdcTheory.ElutionCurve import compute_elution_curve, NUM_PLATES_PC, C, retention_time
    monopore_pdf = FftInvPdf(gec_monopore_phi)
    dispersive_monopore_pdf = FftInvPdf(dispersive_monopore)

    x = np.arange(300, 800)
    t0 = 400
    N = 1000
    T = 0.5
    poresize = 75
    rgs = [150, 90, 50, 35, 20]
    rv = np.linspace(10, poresize, 100)
    rhov = rv/poresize
    rhov[rhov > 1] = 1
    m = 3
    trv = t0 + N*T*(1 - rhov)**m
    r0 = 5000
    rv_hdc = np.linspace(poresize, 200, 50)
    sigma0 = 10
    tp = np.sqrt(NUM_PLATES_PC)*sigma0
    tI = t0 - tp
    trv_hdc = tI + retention_time(tp, rv_hdc, r0, C)
    fig, axes = plt.subplots(nrows=len(rgs), ncols=2,figsize=(18,10))
    for axrow, rg in zip(axes, rgs):
        for k, ax in enumerate(axrow):
            ax.set_xlabel("Time (frames)")
            ax.set_ylabel("Density")
            axt = ax.twinx()
            axt.grid(False)
            axt.set_ylabel("$R_g$")
            axt.plot(trv, rv, color="yellow", label=r"Exclusion Curve: $t_0 + N T (1 - \rho)^m$")
            if rg > 100:
                axt.plot(trv_hdc, rv_hdc, color="cyan", label=r"Calibration Curve: $T_0/(1 + 2 \lambda - C \lambda^2)$")
            axt.legend(loc="upper center", fontsize=9)
            if rg <= 100:
                rg_str = "$R_g$=%g" % rg if rg < poresize else "$R_g$ > %g" % poresize
                ax.set_title("Monopore Dispersive Model Elution Curves with varying $N_0$ with N=%g, T=%g, Poresize=%g, %s" % (N, T, poresize, rg_str))
                
            if k == 0:
                ax.set_ylim(-0.01, 0.21)
            if rg < poresize:
                rho = rg/poresize
                ni_ = N*(1 - rho)**1.5
                ti_ = T*(1 - rho)**1.5

                y = robust_single_pore_pdf_scaled(x - t0, ni_, ti_)
                y_ = y/np.sum(y)

                # y2 = monopore_pdf(x, ni_, ti_, t0)
                # y3 = monopore_pdf(x - t0, ni_, ti_, 0)
            
                ax.plot(x, y_, label='Non-Dispersive (i.e., $N_0=\infty$)')
                # ax.plot(x, y2, ":", label='CF inversion')
                # ax.plot(x, y3, label='CF inversion')
            else:
                ni_ = 0
                ti_ = 0

            if rg > 100:
                sigma = 10
                ax.set_title(r"Hydrodynamic Model Elution Curves with $R_g$ > Poresize=%g $r_0$=%g, $\sigma_0$=%g" % (poresize, r0, sigma))
                for j, rg_ in enumerate([200, 150, 100]):
                    y5 = compute_elution_curve(x, t0, rg_, r0, sigma)
                    y5_ = y5/np.sum(y5)
                    ax.plot(x, y5_, color="C%d" % (4 + j), label='HDC $R_g$=%g' % rg_)
            else:
                for k, N0 in enumerate([1600, 2500, 3600]):
                    y4 = dispersive_monopore_pdf(x, ni_, ti_, N0, t0)
                    ax.plot(x, y4, color="C%d" % (k+1), label='Dispersive $N_0$=%g' % N0)
            ax.axvline(x=t0, color='red', label='t0')

            ax.legend(fontsize=9)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    # import seaborn as sns
    # sns.set_theme()
    sys.path.append("../lib")

    demo_monopore()