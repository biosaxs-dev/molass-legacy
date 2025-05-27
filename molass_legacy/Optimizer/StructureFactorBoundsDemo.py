"""
    StructureFactorBoundsDemo.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import sys
import os
import numpy as np
import seaborn
seaborn.set()

# execute this file with python this-path demo
if len(sys.argv) > 1 and sys.argv[1].find("demo") >= 0:
    this_dir = os.path.dirname(os.path.abspath( __file__ ))
    sys.path.append(this_dir + '/..')
    import molass_legacy.KekLib, SerialAnalyzer

import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Optimizer.StructureFactorBounds import StructureFactorBounds
from Theory.SolidSphere import phi

def demo(qv=None, lrf_info=None, gk_info=None):
    from importlib import reload
    import Optimizer.StructureFactorBounds
    reload(Optimizer.StructureFactorBounds)
    from .StructureFactorBounds import StructureFactorBounds

    if qv is None:
        qv = np.linspace(0.005, 0.4, 200)

    Rg = 35
    R = np.sqrt(5/3)*Rg

    pv = phi(qv, R)**2
    sv = 1 - phi(qv, 2*R)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
    ax11 = axes[0,0]
    ax12 = axes[0,1]
    ax21 = axes[1,0]
    ax22 = axes[1,1]

    ax11.set_title("Hard Sphere")

    ax11.set_yscale('log')
    ax11.plot(qv, pv, label="P(q)")
    ax11.plot(qv, pv*sv, label="P(q)*S(q)")
    ax11.legend()

    bound = 1/(qv*R)**2
    ax12.plot(qv, sv - 1, color="C1", label="S(q) - 1")
    ax12.plot(qv, bound, ":", color="red", label=r"$\frac{1}{(qR)^2}$")
    ax12.plot(qv, -bound, ":", color="red", label=r"$\frac{1}{(qR)^2}$")
    ax12.set_ylim(-1, 1)

    ax12.legend()

    if lrf_info is not None:
        assert gk_info is not None
        Pxr, Cxr, Puv, Cuv, mapped_UvD = lrf_info.matrices

        n = 1
        aq = Pxr[:,n]
        bq = Pxr[:,-1]
        cv = Cxr[n,:]
        k = np.argmax(cv)
        c = cv[k]
        min_aq = np.ones(len(aq)) * 1e-6
        sf1v = bq/np.max([min_aq, aq], axis=0)
        sf_bounds = StructureFactorBounds(qv, lrf_info, gk_info)
        bounds = sf_bounds.get_bounds()
        penalty = sf_bounds.compute_penalty(Pxr)
        print("penalty=", penalty)

        ax21.set_yscale('log')
        ax21.plot(qv, aq*c + bq*c**2)
        ax21.plot(qv, aq*c)
        ax22.plot(qv, sf1v, color="C1")
        ax22.plot(qv, bounds[0], ":", color="red")
        ax22.plot(qv, bounds[1], ":", color="red")
        ax22.set_ylim(-10, 10)

    fig.tight_layout()
    plt.show()

def demo_for_debugger(dialog, canvas):
    from Kratky.GuinierKratkyInfo import GuinierKratkyInfo

    print("demo_for_debugger")

    sd = dialog.optinit_info.sd
    qv = sd.qvector
    params = canvas.get_current_params()
    optimizer = canvas.fullopt
    lrf_info = optimizer.objective_func(params, return_lrf_info=True)
    gk_info = GuinierKratkyInfo(optimizer, params, lrf_info)

    with plt.Dp():
        demo(qv=qv, lrf_info=lrf_info, gk_info=gk_info)

def proof1():
    R = np.sqrt(5/3)*100
    Qmax = 0.5
    qR_max = 0.5*R
    qR_lim = 6
    x1 = np.linspace(0, qR_lim, 100)
    x2 = np.linspace(0, qR_max, 200)

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        fig.suptitle("Proof of sin(x) - x cos(x) < 1.1x where x > 0", fontsize=20)

        ax1.set_title("Proof of sin(x) - x cos(x) < 1.1x where 0 < x < %.2g" % qR_lim, fontsize=16)
        ax2.set_title("Proof of sin(x) - x cos(x) < 1.1x where 0 < x < %.2g" % qR_max, fontsize=16)

        for ax, x in [(ax1, x1), (ax2, x2)]:
            y = np.sin(x) - x*np.cos(x)
            ax.plot(x, y, label=r"$y=sin(x) - x cos(x)$")
            ax.plot(x, 1.1*x, label=r"$y=1.1x$")
            ax.plot(x, -1.1*x, label=r"$y=-1.1x$")
            ax.legend()

        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    # demo()
    proof1()
