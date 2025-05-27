"""
    SecTheory.ExclusionCurveSpike.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
import sys
import numpy as np
from matplotlib.gridspec import GridSpec

def compute_tr_impl(rgv, proportions, poresizes, Ps, x0, m):
    proportions_ = proportions/np.sum(proportions)

    PKsec = 1
    for p, rp, P in zip(proportions_, poresizes, Ps):
        rho = rgv/rp
        rho[rho > 1] = 1
        PKsec *= (P*(1 - rho)**m)**p

    return x0 + PKsec

class ExclusionCurvepy:
    def __init__(self, poresizes, proportions, P, x0, m):
        self.poresizes = poresizes
        self.proportions = proportions
        self.x0 = x0
        self.P = P
        self.m = m

    def compute_deviation(self, rg_points):
        pass

    def compute_tr(self, rgv):
        pass

    def fit(self, rg_points):
        pass

def demo():
    import molass_legacy.KekLib.DebugPlot as plt

    rp = 150

    rgv = np.flip(np.linspace(0, rp, 100))

    x0 = 50
    P = 200
    m = 3

    proportions1 = np.ones(1)
    poresizes1 = np.array([rp])
    Ps1 =  np.array([P])

    proportions2 = np.ones(2)/2
    poresizes2 = np.array([rp]*2)
    Ps2 =  np.array([P]*2)

    proportions3 = np.ones(3)/3
    poresizes3 = np.array([rp]*3)
    Ps3 =  np.array([P]*3)

    proportions4 = np.ones(2)/2
    poresizes4 = np.array([100, 100])
    Ps4 =  np.array([P, P])

    proportions5 = np.ones(2)/2
    poresizes5 = np.array([150, 50])
    Ps5 =  np.array([P/2, P*2])

    proportions6 = np.ones(2)/2
    poresizes6 = np.array([150, 50])
    Ps6 =  np.array([P, P])

    tr1 = compute_tr_impl(rgv, proportions1, poresizes1, Ps1, x0, m)
    tr2 = compute_tr_impl(rgv, proportions2, poresizes2, Ps2, x0, m)
    tr3 = compute_tr_impl(rgv, proportions3, poresizes3, Ps3, x0, m)
    tr4 = compute_tr_impl(rgv, proportions4, poresizes4, Ps4, x0, m)
    tr5 = compute_tr_impl(rgv, proportions5, poresizes5, Ps5, x0, m)
    tr6 = compute_tr_impl(rgv, proportions6, poresizes6, Ps6, x0, m)

    with plt.Dp():
        fig, ax = plt.subplots()

        ax.plot(tr1, rgv, label="x0:%g, poresizes: %s" % (x0, str(poresizes1)))
        ax.plot(tr2, rgv, label="x0:%g, poresizes: %s" % (x0, str(poresizes2)))
        ax.plot(tr3, rgv, label="x0:%g, poresizes: %s" % (x0, str(poresizes3)))
        ax.plot(tr4, rgv, label="x0:%g, poresizes: %s" % (x0, str(poresizes4)))
        ax.plot(tr5, rgv, label="x0:%g, poresizes: %s" % (x0, str(poresizes5)))
        ax.plot(tr6, rgv, label="x0:%g, poresizes: %s" % (x0, str(poresizes6)))

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
