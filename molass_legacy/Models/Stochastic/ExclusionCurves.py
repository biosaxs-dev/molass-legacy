"""
    Models.Stochastic.ExclusionCurves.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from molass_legacy.Models.Stochastic.ParamLimits import USE_K

RESOL_C = 100
RESOL_F = 1000
def get_resolusin(fine=False):
    return RESOL_F if fine else RESOL_C

def simple_curve(rv, N, T, t0, m, poresize):
    rhov = rv/poresize
    tv = t0 + N*T*(1 - rhov)**m
    return tv

def compute_simply_conformant_rgs(N, KT, t0, poresize, m, trs, debug=False, rgs=None, plot_info=None):
    from importlib import reload
    import Models.Stochastic.MonoporeUtils as module
    reload(module)
    from molass_legacy.Models.Stochastic.MonoporeUtils import plot_monopore_conformance

    if USE_K:
        T = KT/N
    else:
        T = KT
    # trs = t0 + N*T*(1 - rhov)**m
    # (trs - t0)/(N*T) = (1 - rhov)**m
    small_values = np.ones(len(trs))
    tr_diff = np.max([trs - t0, small_values], axis=0)
    rhov = 1 - np.power(tr_diff/(N*T), 1/m)
    conformant_rgs = rhov * poresize

    if debug and plot_info is not None:
        plot_monopore_conformance(KT, t0, poresize, plot_info, rg_info=(rgs, trs), confrg_info=(conformant_rgs, trs))

    return conformant_rgs

def plot_simple_curves(ax, N, T, t0, m, poresize_list):
    ax.set_title("Simple Exclusion Curves with N=%d, T=%.1f, t0=%.1f, m=%.1f" % (N, T, t0, m))
    for poresize in poresize_list:
        rv = np.linspace(poresize, 10, 100)
        tv = simple_curve(rv, N, T, t0, m, poresize)
        ax.plot(tv, rv, label="poresize=%d" % poresize)
    ax.axvline(x=t0, color="red", label="$t_0$=%d" % t0)
    ax.legend()

def monopore_curve(rv, N, T, t0, me, mp, poresize, fine=False):
    from SecTheory.BasicModels import robust_single_pore_pdf
    rhov = rv/poresize
    niv = N*(1 - rhov)**me
    tiv = T*(1 - rhov)**mp
    x = np.linspace(t0, N*T, get_resolusin(fine))
    tv = []
    for ni, ti in zip(niv, tiv):
        y = robust_single_pore_pdf(x - t0, ni, ti)
        m = np.argmax(y)
        tv.append(x[m])
    return np.asarray(tv)

def plot_monopore_curves(ax, N, T, t0, m, poresize, fine=False):
    ax.set_title("Monopore Exclusion Curves with N=%d, T=%.3g, poresize=%g" % (N, T, poresize))
    rv = np.linspace(poresize, 10, get_resolusin(fine))
    tv = simple_curve(rv, N, T, t0, m, poresize)
    ax.plot(tv, rv, label="simple, m=%g" % m)
    for me in [1, 1.5, 2]:
        mp = m - me
        tv = monopore_curve(rv, N, T, t0, me, mp, poresize, fine=fine)
        ax.plot(tv, rv, label="me, mp=(%g, %g)" % (me, mp))
    ax.axvline(x=t0, color="red", label="$t_0$=%d" % t0)
    ax.legend()

def lnpore_curve(rv, N, T, t0, me, mp, mu, sigma, fine=False):
    from molass_legacy.Models.Stochastic.LognormalPoreFunc import lognormal_pore_func
    x = np.linspace(t0, N*T, get_resolusin(fine))
    tv = []
    for rg in rv:
        y = lognormal_pore_func(x, 1, N, T, me, mp, mu, sigma, rg, t0)
        m = np.argmax(y)
        tv.append(x[m])
    return np.asarray(tv)

def plot_lnpore_curves(ax, N, T, t0, m, poresize, fine=False):
    from molass_legacy.Models.Stochastic.LognormalUtils import compute_mu_sigma
    ax.set_title("Lnpore Exclusion Curves with N=%d, T=%.3g, poresize=%g" % (N, T, poresize))
    rv = np.linspace(poresize, 10, get_resolusin(fine))
    tv = simple_curve(rv, N, T, t0, m, poresize)
    ax.plot(tv, rv, label="simple, m=%g" % m)
    me = 1.5
    mp = 1.5
    for stdev_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        stdev = stdev_ratio*poresize
        mu, sigma = compute_mu_sigma(poresize, stdev)
        tv = lnpore_curve(rv, N, T, t0, me, mp, mu, sigma, fine=fine)
        ax.plot(tv, rv, label=r"$\mu$, $\sigma$=(%.3g, %.3g)" % (mu, sigma))
    ax.axvline(x=t0, color="red", label="$t_0$=%d" % t0)
    ax.legend()

def demo(N, T, t0, m, fine=False):
    from time import time
    start_t = time()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
    fig.suptitle("Exclusion Curves: Monopore vs. Lognormalpore", fontsize=20)
    plot_simple_curves(axes[0,0], N, T, t0, m, [30, 50, 100, 200, 300])
    plot_monopore_curves(axes[0,1], N, T, t0, m, 100, fine=fine)
    plot_lnpore_curves(axes[1,0], 4000, 0.25, t0, m, 100, fine=fine)
    plot_lnpore_curves(axes[1,1], N, T, t0, m, 100, fine=fine)
    fig.tight_layout()
    print("Elapsed time: %.3f sec" % (time() - start_t))
    plt.show()

if __name__ == "__main__":
    import sys
    import seaborn as sns
    sns.set_theme()
    sys.path.append("../lib")

    demo(2000, 0.5, 100, 3, fine=False)     # it takes 10 min when fine=True