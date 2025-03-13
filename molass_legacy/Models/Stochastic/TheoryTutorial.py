"""
    Models.Stochastic.TheoryTutorial.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt

def tutorial_illust():
    from molass_legacy.Models.Stochastic.Monopore import monopore_func
    print("tutorial_illust")

    N = 1000
    T = 0.5
    t0 = 50
    me = 1.5
    mp = 1.5
    poresize = 100

    x = np.arange(600)
    cy_list = []
    rgs = [44, 40, 36, 20]
    scales = [0.1, 0.5, 0.2, 0.1]
    for (rg, scale) in zip(rgs,scales):
        cy = monopore_func(x, scale, N, T, t0, me, mp, poresize, rg)
        cy_list.append(cy)
    y = np.sum(cy_list, axis=0)

    def compute_monopore_tv(rv):
        rhov = np.asarray(rv)/poresize
        rhov[rhov > 1] = 1
        return t0 + N * T * (1 - rhov)**(me + mp)

    rv = np.linspace(1, 75, 100)
    tv = compute_monopore_tv(rv)
    pv = compute_monopore_tv(rgs)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title("Stochastic Theory Illustration where N=%g, T=%g, $t_0$=%g, $m_e$=%g, $m_p$=%g, poresize=%g"
                 % (N, T, t0, me, mp, poresize), y=1.05, fontsize=14)
    ax.set_xlabel("Time (Frames)")
    ax.set_ylabel("Density")
    for k, (rg, cy) in enumerate(zip(rgs, cy_list), start=1):
        ax.plot(x, cy, ":", lw=2, label="component-%d ($R_g$=%g)" %(k, rg))

    ax.plot(x, y, color="orange", alpha=0.5, label="component total")
    ax.axvline(x=t0, color="red", label="$t_0$ = %g" % t0, alpha=0.5)
    ax.axvline(x=t0 + N*T, color="green", label="$t_0$ + N*T = %g" % (t0+N*T), alpha=0.5)
    ax.legend(loc="upper center")

    axt = ax.twinx()
    axt.grid(False)
    axt.set_ylabel(r"$R_g (\AA_)$")

    axt.plot(tv, rv, color="yellow", label="Exclusion Curve")
    axt.plot(pv, rgs, "o", markersize=3, color="red", label="Component Peak Positions")
    dx, dy = 5, 2
    for tx, ty in zip(pv, rgs):
        axt.text(tx+dx, ty+dy, "%g" % ty, alpha=0.5)
    axt.legend()

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, 0.06)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    import seaborn as sns
    sns.set_theme()
    sys.path.append("../lib")

    tutorial_illust()