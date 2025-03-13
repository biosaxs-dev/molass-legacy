"""
    Models.Stochastic.SdmTimeTutorial.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt

def illust_and_proof():
    from molass_legacy.Models.Stochastic.DispersivePdf import dispersive_monopore_pdf
    from molass_legacy.Models.Stochastic.Monopore import monopore_func
    print("tutorial")

    x = np.arange(-1000, 800)
    poresize = 100
    rgs = np.array([100, 50, 40, 30, 20])
    rhov = rgs/poresize
    N = 1000
    T = 1
    me = 1.5
    mp = 1.5
    t0 = 100
    tinj = x[0]
    N0 = 3600
    scale = 1
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
    fig.suptitle("SDM Elution Illustration in Different Time Axes with Poresize=%g, $N_0=%d$" % (poresize, N0), fontsize=20)
    for k, (rg, rho) in enumerate(zip(rgs, rhov)):
        ni = N*(1 - rho)**me
        ti = T*(1 - rho)**mp
        for ax, x_ in zip((ax1, ax2), (x, x-tinj)):
            cy = dispersive_monopore_pdf(x - tinj, ni, ti, N0, t0 - tinj, timescale=0.25)
            bop = ">=" if k == 0 else "="
            ax.plot(x_, cy, label="$R_g$%s%g" % (bop, rg))
            if k > 0:
                cy = monopore_func(x, scale, N, T, t0, me, mp, poresize, rg)
                ax.plot(x_, cy,":", color="C%s" % k)
    for ax, tinj_, t0_ in [(ax1, tinj, t0), (ax2, 0, t0 - tinj)]:
        ax.set_title("$t_{inj}=%g$, $t_0=%g$" % (tinj_, t0_), fontsize=16)
        ax.axvline(x=tinj_, color="gray", label="$t_{inj}$")
        ax.axvline(x=t0_, color="red", label="$t_0$")
        hy = np.percentile(ax.get_ylim(), 60)
        dx = t0_ - tinj_
        ax.arrow(x=tinj_, y=hy, dx=dx, dy=0, width=0.003, head_width=0.008,
                    head_length=0.1*dx, length_includes_head=True, color='pink', alpha=0.5)
        ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    import seaborn as sns
    sns.set_theme()
    sys.path.append("../lib")

    illust_and_proof()