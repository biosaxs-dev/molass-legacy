"""
    UnknownConc.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from bisect import bisect_right
from molass_legacy.Peaks.ElutionModels import egh
import seaborn
seaborn.set()
import molass_legacy.KekLib.DebugPlot as plt

def demo1_():
    qv = np.linspace(0.005, 0.4, 400)
    jv = np.arange(300)
    rg = 35
    h = 1
    mu = 150
    sigma = 20
    tau = 0
    a = 0
    K = 0.5
    error = 0.05
    demo_impl(qv, jv, rg, h, mu, sigma, tau, a, K, error)

def demo1():
    qv = np.linspace(0.005, 0.4, 400)
    jv = np.arange(300)
    rg = 30
    h = 1
    mu = 211
    sigma = 23.6
    tau = -29.5
    a = -0.49
    K = 0.5
    error = 0.02
    demo_impl(qv, jv, rg, h, mu, sigma, tau, a, K, error)

def demo_impl(qv, jv, rg, h, mu, sigma, tau, a, K, error):
    from importlib import reload
    import SimTools.HardSphere
    reload(SimTools.HardSphere)
    from SimTools.HardSphere import get_model_data

    D, P, C = get_model_data(qv, jv, rg, h, mu, sigma, tau, a=a, K=K, error=0)

    aq, bq = P.T

    D, P, C = get_model_data(qv, jv, rg, h, mu, sigma, tau, a=a, K=K, error=error)

    i = bisect_right(qv, 0.02)

    cy = D[i,:]
    C_ = np.array([cy, cy**2])
    C_inv = np.linalg.pinv(C_)
    P_ = D @ C_inv

    """
    Iqc1 = aq*c1 + bq*c1**2
    Iqc2 = aq*c2 + bq*c2**2
        ...
        ...
    Iqcj = aq*cj + bq*cj**2
        ...
    """

    aq_, bq_ = P_.T

    def objective(p):
        h, m, s, t, a, b = p
        cy_ = egh(jv, h, m, s, t)
        return np.linalg.norm(a*cy_ + b*cy_**2 - cy)

    j = np.argmax(cy)
    h = cy[j]
    s = 10
    t = 0

    init_params = h, j, s, t, aq_[i], bq_[i]
    ret = minimize(objective, init_params)
    print("ret.fun=", ret.fun)

    h, m, s, t, a, b = ret.x
    cy_ = egh(jv, h, m, s, t)

    Cs = np.array([cy_, cy_**2])
    Csinv = np.linalg.pinv(Cs)
    Ps = D @ Csinv
    aqs, bqs = Ps.T

    plot_demo_result(qv, jv, cy, cy_, aq, bq, aq_, bq_, aqs, bqs, rg, error, D, j)

def plot_demo_result(qv, jv, cy, cy_, aq, bq, aq_, bq_, aqs, bqs, rg, error, D, j):
    scale = np.max(aq)/np.max(aqs)
    print("scale=", scale)
    aqs_ = scale*aqs
    bqs_ = scale**2*bqs

    ig = bisect_right(qv, 1.8/rg)
    gslice = slice(0,ig)
    gqv = qv[gslice]
    gqv2 = gqv**2

    gy = D[gslice,j]

    with plt.Dp():
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20,5))
        fig.suptitle("True A(q), B(q) retrieval in Hard Sphere Model with Error=%g" % error, fontsize=20)
        ax1.set_title("Apparent Estimation", fontsize=16)
        ax2.set_title("Concetration Correction", fontsize=16)
        ax3.set_title("Corrected Estimation", fontsize=16)
        ax4.set_title("Guinier Comparison", fontsize=16)

        ax1.plot(qv, aq, label="True A(q)")
        ax1.plot(qv, bq, label="True B(q)", color="pink")
        ax1.plot(qv, aq_, ":", label="A(q) directly from measured data", color="C1", lw=2)
        ax1.plot(qv, bq_, ":", label="B(q) directly from measured data", color="cyan", lw=2)
        ax2.plot(jv, cy, label="apparent concentration", color="gray", alpha=0.5)
        ax2.plot(jv, cy_, ":", label="corrected concentration", color="C2", lw=2)
        ax3.plot(qv, aq, label="True A(q)")
        ax3.plot(qv, bq, label="True B(q)", color="pink")
        ax3.plot(qv, aqs_, ":", label="retrieved A(q)", color="C2", lw=2)
        ax3.plot(qv, bqs_, ":", label="retrieved B(q)", color="C3", lw=2)

        ax4.plot(gqv2, np.log(gy), label="apparent data", color="gray", alpha=0.5)
        ax4.plot(gqv2, np.log(aq[gslice]), label="true")
        ax4.plot(gqv2, np.log(aq_[gslice]), ":", label="naiive A(q)", color="C1", lw=2)
        ax4.plot(gqv2, np.log(aqs_[gslice]), ":", label="retrieved A(q)", color="C2", lw=2)

        q = 1.3/rg
        q2 = q**2
        ymin, ymax = ax4.get_ylim()
        ax4.set_ylim(ymin,ymax)
        ax4.plot([q2, q2], [ymin, ymax], color="yellow", label="$qR_g=1.3$")

        for ax in ax1, ax2, ax3, ax4:
            ax.legend()

        fig.tight_layout()
        plt.show()

def realistic_demo_impl(qv, jv, rg, h, mu, sigma, tau, a, K, error):
    from importlib import reload
    import SimTools.HardSphere
    reload(SimTools.HardSphere)
    from SimTools.HardSphere import get_model_data

    D, P, C = get_model_data(qv, jv, rg, h, mu, sigma, tau, a=a, K=K, error=0)

    aq, bq = P.T

    D, E, P, C = get_model_data(qv, jv, rg, h, mu, sigma, tau, a=a, K=K, error=error, return_error_matrix=True)

    i = bisect_right(qv, 0.02)

    cy = D[i,:]
    C_ = np.array([cy, cy**2])
    C_inv = np.linalg.pinv(C_)
    P_ = D @ C_inv

    """
    Iqc1 = aq*c1 + bq*c1**2
    Iqc2 = aq*c2 + bq*c2**2
        ...
        ...
    Iqcj = aq*cj + bq*cj**2
        ...
    """

    aq_, bq_ = P_.T

    def objective(p):
        h, m, s, t, a, b = p
        cy_ = egh(jv, h, m, s, t)
        return np.linalg.norm(a*cy_ + b*cy_**2 - cy)

    j = np.argmax(cy)
    h = cy[j]
    s = 10
    t = 0

    init_params = h, j, s, t, aq_[i], bq_[i]
    ret = minimize(objective, init_params)
    print("ret.fun=", ret.fun)

    h, m, s, t, a, b = ret.x
    cy_ = egh(jv, h, m, s, t)

    Cs = np.array([cy_, cy_**2])
    Csinv = np.linalg.pinv(Cs)
    Ps = D @ Csinv
    aqs, bqs = Ps.T

    plot_demo_result(qv, jv, cy, cy_, aq, bq, aq_, bq_, aqs, bqs, rg, error, D, j)
