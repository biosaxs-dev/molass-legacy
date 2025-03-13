"""
    HdcTheory.SpikeDemo.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def demo():
    poresize = 75
    rv = np.linspace(10, poresize, 100)
    rhov = rv/poresize
    rhov[rhov > 1] = 1
    m = 3
    N = 1000
    T = 1
    t0 = 50
    trv = t0 + N*T*(1 - rhov)**m

    lrv = np.linspace(poresize-5, 800, 100)
    lamv = lrv/5000
    C = 2.698
    xv = t0/(1 + 2*lamv - C*lamv**2)

    fig, ax = plt.subplots()
    ax.plot(trv, rv, label=r"$t_0 + N T (1 - \rho)^m$")
    ax.axvline(x=t0, color='red', label=r"$t_0$")
    ax.plot(xv, lrv, label=r"$t_0/(1 + 2 \lambda - C \lambda^2)$")
    ax.set_xlim(0, 300)
    ax.legend()
    fig.tight_layout()
    plt.show()

def valid_range_demo():
    from HdcTheory.ElutionCurve import C
    rv = np.linspace(-200, 1000, 100)
    r0 = 1000
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
    fig.suptitle("Valid $R_{eff}$ Range of $K_{HDC}$ when $R_0$=%g and C=%g" % (r0, C), fontsize=20)
    lamb = rv/r0
    y = 1 + 2*lamb - C*lamb**2
    ax1.set_xlabel(r"$R_{eff}$")
    # Rh/R0 * C= 1
    # Rh = R0/C
    ax1.set_ylabel(r"$1 + 2 \lambda - C \lambda^2$")
    boundary = r0/C
    negative = rv <= 0
    positive = rv > 0
    invalid = rv > boundary
    valid = np.logical_and(positive, rv <= boundary)
    ax1.plot(rv[valid], y[valid])
    ax1.plot(rv[negative], y[negative], ":", color='gray', alpha=0.5)
    ax1.plot(rv[invalid], y[invalid], ":", color='gray', alpha=0.5)
    boundary_label = r"$\lambda=\frac{R_{eff}}{R_0}=\frac{1}{C}$"
    ax1.axvline(x=r0/C, color='pink', label=boundary_label)
    ax1.legend()

    ax2.set_xlabel(r"$K_{HDC}: (1 + 2 \lambda - C \lambda^2)^{-1}$")
    ax2.set_ylabel(r"$R_{eff}$")
    ax2.plot(1/y[valid], rv[valid], label=r"Calibration Curve", color="cyan")
    ax2.plot(1/y[negative], rv[negative], ":", color='gray', alpha=0.5)
    ax2.plot(1/y[invalid], rv[invalid], ":", color='gray', alpha=0.5)
    ax2.axhline(y=r0/C, color='pink', label=boundary_label)
    ax2.axvline(x=1, color='red', label=r"$t_0$")
    # xmin, xmax = ax2.get_xlim()
    ax2.set_xlim(0, 2)
    ax2.legend(loc="upper left")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    sys.path.append("../lib")    
    # demo()
    valid_range_demo()
