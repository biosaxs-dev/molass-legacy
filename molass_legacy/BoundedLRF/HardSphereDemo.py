"""
    HardSphereDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt

def demo1():
    from importlib import reload
    import SimTools.HardSphere
    reload(SimTools.HardSphere)
    from SimTools.HardSphere import get_model_data
    import SimTools.EoiiPlotUtils
    reload(SimTools.EoiiPlotUtils)
    from SimTools.EoiiPlotUtils import plot_eoii

    qv = np.linspace(0.005, 0.4, 200)
    jv = np.arange(300)
    rg = 35
    h = 1
    mu = 150
    sigma = 20
    tau = 0
    K = 0.5
    D, P, C = get_model_data(qv, jv, rg, h, mu, sigma, tau, K=K)

    title = "Effect of Interparticle Interactions Simulation with K=%g" % K

    De, Pe, Ce = get_model_data(qv, jv, rg, h, mu, sigma, tau, K=K, error=0.008)

    with plt.Dp():
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18,10))
        fig.suptitle(title, fontsize=20)
        ax1, ax2, ax3 = axes[0,:]
        ax1.set_title("Elution View", fontsize=16)
        ax2.set_title("Scattering View (Linear)", fontsize=16)
        ax3.set_title("Scattering View (Log)", fontsize=16)
        plot_eoii(qv, jv, D, P, C, axes[0,:])

        ax1, ax2, ax3 = axes[1,:]
        ax1.set_title("Noisy Data Elution View", fontsize=16)
        ax2.set_title("Noisy Data Scattering View (Linear)", fontsize=16)
        ax3.set_title("Noisy Data Scattering View (Log)", fontsize=16)
        plot_eoii(qv, jv, De, Pe, Ce, axes[1,:])

        fig.tight_layout()
        plt.show()

def demo():
    from importlib import reload
    import SimTools.HardSphere
    reload(SimTools.HardSphere)
    from SimTools.HardSphere import get_model_data
    import SimTools.EoiiPlotUtils
    reload(SimTools.EoiiPlotUtils)
    from SimTools.EoiiPlotUtils import plot_eoii

    qv = np.linspace(0.005, 0.4, 200)
    jv = np.arange(300)
    rg = 35
    h = 1
    mu = 150
    sigma = 20
    tau = 0
    K = 0.5
    D, P, C = get_model_data(qv, jv, rg, h, mu, sigma, tau, K=K)

    title = "Eoii Algorithm using Aparent Data"

    i = bisect_right(qv, 0.02)

    with plt.Dp():
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18,10))
        fig.suptitle(title, fontsize=20)
        ax1, ax2, ax3 = axes[0,:]
        ax1.set_title("Elution View", fontsize=16)
        ax2.set_title("Scattering View (Linear)", fontsize=16)
        ax3.set_title("Scattering View (Log)", fontsize=16)
        plot_eoii(qv, jv, D, P, C, axes[0,:])

        A = D.copy()
        cy_init = A[i,:]
        cy = cy_init
        cy2 = cy**2
        C_ = np.array([cy, cy2])
        C_inv = np.linalg.pinv(C_)
        P_ = D @ C_inv
        aq, bq = P_.T
        ax4 = axes[1,0]
        ax4.plot(qv, aq)
        ax4.plot(qv, bq)
        B = bq[:,np.newaxis] @ cy2[np.newaxis,:]
        A_ = D - B

        ax5 = axes[1,1]
        ax5.plot(jv, cy_init)
        ax5.plot(jv, A_[i,:], ":")

        fig.tight_layout()
        plt.show()
