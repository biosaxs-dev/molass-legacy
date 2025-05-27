"""
    SecTheory.TriporeSecSimulation.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
import sys
import numpy as np
from matplotlib.gridspec import GridSpec

def demo(use_cf=True):
    import molass_legacy.KekLib.DebugPlot as plt
    if use_cf:
        from SecTheory.SecCF import sdm_tripore
        from SecTheory.SecPDF import FftInvPdf
        tripore_pdf = FftInvPdf(sdm_tripore)
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
            y = tripore_pdf(x, 1/3*np_, tp_, 1/3*np_, tp_, 1/3*np_, tp_, N0, t0, x0)
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

if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.dirname(this_dir)
    sys.path.append(lib_dir)

    import seaborn
    seaborn.set()
    import molass_legacy.KekLib

    demo()
