"""
    Theory.Anisotropy.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from .JsPedersen1997 import F1, P6
from molass_legacy.Saxs.SaxsSamples import EllipsoidVoxels
from molass_legacy.Saxs.ReciprocalData import ReciprocalData
import molass_legacy.KekLib.DebugPlot as plt

class Anisotropy:
    def __init__(self):
        pass

    def fit_curve(self, q, y, Rg):
        def get_curve(p):
            a, b, c = p
            ellipsoid = EllipsoidVoxels(radii=(a, b, c))
            data = ellipsoid.get_data(density=0.1)
            rdata = ReciprocalData(data.shape)
            F = rdata.get_reciprocal(data)
            ey = rdata.get_scattering_curve(q, F)
            ey /= ey[0]
            return ey

        def plot_curve(ey):
            plt.push()
            fig, ax = plt.subplots()
            ax.set_yscale('log')
            ax.plot(q, y)
            ax.plot(q, ey)
            plt.show()
            plt.pop()

        def f(p):
            ey = get_curve(p)
            if False:
                plot_curve(ey)
            return np.sum((ey - y)**2)

        # R = np.sqrt(5/3)*Rg
        R = 0.2
        cons = (
            {'type': 'ineq', 'fun': lambda x:  x[1] - x[0]},
            {'type': 'ineq', 'fun': lambda x:  x[2] - x[1]},
        )
        bnds = ((0, None), (0, None), (0, None))

        dmin = None
        pmin = None
        for p in np.random.uniform(0, 1, (200,3)):
            sp = sorted(p)
            d = f(sp)
            if dmin is None or d < dmin:
                dmin = d
                pmin = sp

        print("pmin=", pmin)
        ey = get_curve(pmin)
        plot_curve(ey)

        if False:
            res = minimize(f, pmin, bounds=bnds, constraints=cons)
            self.params = res.x
            print('params=', self.params)

    def get_params(self):
        return self.params

    def get_curve(self):
        pass

def ellipsoid_scale(Rg, a, b, c):
    r = np.sqrt((a**2 + b**2 + c**2)/5)
    scale = Rg/r
    return scale

def demo():

    Rg = 35
    qv = np.linspace(0.005, 0.4, 100)

    R = np.sqrt(5/3)*Rg
    b1 = np.pi/R
    b2 = b1*1.5
    # a, b, c = np.random.uniform(0, 1, 3)
    a, b, c = 40, 35, 30
    print((a, b, c))
    scale = ellipsoid_scale(Rg, a, b, c)
    a_ = a*scale
    b_ = b*scale
    c_ = c*scale
    y_ = P6(qv, a_, b_, c_)

    ai = Anisotropy()
    ai.fit_curve(qv, y_, Rg)

    fig, ax = plt.subplots()

    ax.set_title("Anisotroy Estimation", fontsize=20)
    ax.set_yscale('log')
    ax.plot(qv, F1(qv, R)**2, label='Sphere')
    ax.plot(qv, y_, label='Ellipsoid(%.3g, %.3g, %.3g)' % (a_, b_, c_))

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    for b, pc in zip([b1, b2], [50, 1]):
        ax.plot([b, b], [ymin, ymax], ':', color='red', label='rank2 ratio %d%%' % pc)

    ax.legend()
    fig.tight_layout()
    plt.show()
