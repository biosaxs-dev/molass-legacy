"""
    Peaks.EghTauLimits.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .ElutionModels import egh
from SecTheory.BasicModels import robust_single_pore_pdf

# t = k*(x - t0)
# Ksec = (1 - rho)**

def get_stochastic_model(x, y):

    def objective(p):
        t0, h, k, np_, tp_ = p
        return np.sum((h*robust_single_pore_pdf(k*(x - t0), np_, tp_) - y)**2)

    ret = minimize(objective, (0, 30, 1, 150, 1))
    return ret

def demo(taus=np.arange(-10, 20).reshape((6,5))):
    nrows, ncols = taus.shape
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, nrows*2))
    fig = plt.figure(figsize=(3*ncols + nrows*2, nrows*2))
    gs = GridSpec(nrows, ncols + (ncols-1))

    fig.suptitle("Single Pore Model Correspondence to EGH Model where sigma=20", fontsize=30)

    x = np.arange(300)
    fv_s = []
    for i in range(nrows):
        for j in range(ncols):
            tau = taus[i,j]
            ax = fig.add_subplot(gs[i,j])
            ax.set_title("tau=%g" % tau, fontsize=16)
            y = egh(x, 1, 150, 20, tau)
            ax.plot(x, y, label="egh")
            ret = get_stochastic_model(x, y)
            print("ret.x=", ret.x)
            fv_s.append(ret.fun)
            t0, h, k, np_, tp_ = ret.x
            y_ = h*robust_single_pore_pdf(k*(x - t0), np_, tp_)
            ax.plot(x, y_, label="stochastic")
            ax.legend()

    ax = fig.add_subplot(gs[:,ncols:])
    ax.set_title("EGH Deviations from Stochastic Model", fontsize=20)
    ax.set_xlabel("Tau", fontsize=16)
    ax.set_ylabel("Deviation", fontsize=16)
    ax.plot(taus.flatten(), fv_s, "o-")

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()
