"""
    SecTheory.MonoporeSecSimulation.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
import sys
import numpy as np
from scipy.optimize import minimize, basinhopping
from matplotlib.gridspec import GridSpec

def demo(use_cf=True):
    import molass_legacy.KekLib.DebugPlot as plt
    if use_cf:
        from SecTheory.SecCF import sdm_monopore
        from SecTheory.SecPDF import FftInvPdf
        monopore_pdf = FftInvPdf(sdm_monopore)
    else:
        # 
        assert False

    x = np.arange(300)

    me = 0.1
    mp = 1.5
    rp = 76
    rg = 35
    rho = rg/rp
    N = 171
    T = 0.63
    t0 = 100
    sigma0 = 10
    N0 = (t0/sigma0)**2
    x0 = -20

    rg_array = np.array([60, 50, 40, 30, 20, 10, 5])

    y_list = []
    for rg_ in rg_array:
        rho_ = rg_/rp
        np_ = N*(1 - rho_)**me
        tp_ = T*(1 - rho_)**mp
        if use_cf:
            y = monopore_pdf(x, np_, tp_, N0, t0, x0)
        else:
            assert False
        y_list.append(y)

    with plt.Dp():
        fig, ax = plt.subplots()

        for y, rg_ in zip(y_list, rg_array):
            ax.plot(x, y, label="$R_g=%.3g$" % rg_)

        ax.legend()
        fig.tight_layout()
        plt.show()

def compare_stochastic_dispersive():
    import molass_legacy.KekLib.DebugPlot as plt
    from SecTheory.SecCF import shifted_phi, sdm_monopore
    from SecTheory.SecPDF import FftInvPdf
    gec_monopore_pdf = FftInvPdf(shifted_phi)
    sdm_monopore_pdf = FftInvPdf(sdm_monopore)

    x = np.arange(300)

    me = 0.1
    mp = 1.5
    rp = 76
    rg = 35
    rho = rg/rp
    N = 171
    T = 0.63
    t0 = 100
    sigma0 = 10
    N0 = (t0/sigma0)**2
    x0 = 50

    rg_array = np.array([60, 50, 40, 30, 20, 10, 5])

    y1_list = []
    y2_list = []
    for rg_ in rg_array:
        rho_ = rg_/rp
        np_ = N*(1 - rho_)**me
        tp_ = T*(1 - rho_)**mp
        y1 = gec_monopore_pdf(x, np_, tp_, t0 - x0)
        y1_list.append(y1)
        y2 = sdm_monopore_pdf(x, np_, tp_, N0, t0, -x0)
        y2_list.append(y2)

    with plt.Dp():
        fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(8,10))

        ax1.set_title("Simple Monopore Model", fontsize=16)
        ax2.set_title("Dispeasive Monopore Model", fontsize=16)

        for y1, y2, rg_ in zip(y1_list, y2_list, rg_array):
            ax1.plot(x, y1, label="$R_g=%.3g$" % rg_)
            ax2.plot(x, y2, label="$R_g=%.3g$" % rg_)

        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        plt.show()

def compare_simple_curve():
    import molass_legacy.KekLib.DebugPlot as plt
    from SecTheory.SecCF import sdm_monopore
    from SecTheory.SecPDF import FftInvPdf
    monopore_pdf = FftInvPdf(sdm_monopore)

    x = np.arange(300)

    me = 0.1
    mp = 1.5
    rp = 200
    rg = 35
    rho = rg/rp
    N = 171
    T = 0.63
    t0 = 100
    sigma0 = 10
    N0 = (t0/sigma0)**2
    x0 = -20

    rg_array = np.array([180, 150, 120, 90, 80, 70, 60, 50, 40, 30, 20, 10])

    y_list = []
    p_list = []
    for rg_ in rg_array:
        rho_ = rg_/rp
        np_ = N*(1 - rho_)**me
        tp_ = T*(1 - rho_)**mp
        y = monopore_pdf(x, np_, tp_, N0, t0, x0)
        y_list.append(y)
        j = np.argmax(y)
        p_list.append((x[j], y[j]))

    r = np.linspace(0, rp, 100)
    rho = r/rp
    P = 300
    rho_ = rg_array/rp
    rho_[rho_ > 1] = 1
    p_array = np.array(p_list)
    x_array = p_array[:,0]

    m = me + mp

    def objective(p):
        t0_, P_ = p
        tr_ = t0_ + P_*(1 - rho_)**m
        return np.sum((tr_ - x_array)**2)

    # method="Nelder-Mead"
    bounds = [(None, None), (0, None)]
    ret = basinhopping(objective, (t0, P), minimizer_kwargs=dict(method='Nelder-Mead', bounds=bounds)
    )

    t0opt, Popt = ret.x
    tr = t0opt + Popt*(1 - rho)**m
    tr_ = t0opt + Popt*(1 - rho_)**m

    print("t0opt=", t0opt)
    print("Popt=", Popt)

    with plt.Dp():
        fig, ax = plt.subplots()

        ax.set_title(r"$ t_R = t_0 + P(1 - \rho)^m $ Fitting to Monopore Model", fontsize=16)

        axt = ax.twinx()
        axt.grid(False)

        for y, rg_ in zip(y_list, rg_array):
            ax.plot(x, y, label="$R_g=%.3g$" % rg_)

        ax.plot(*p_array.T, "o", color="yellow")

        axt.plot(tr, r, color="yellow")
        axt.plot(tr_, rg_array, "o", color="red")

        ymin, ymax = axt.get_ylim()
        axt.set_ylim(-ymax, ymax)

        ax.legend()
        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.dirname(this_dir)
    sys.path.append(lib_dir)

    import seaborn
    seaborn.set()
    import molass_legacy.KekLib

    # demo()
    compare_stochastic_dispersive()
    # compare_simple_curve()
