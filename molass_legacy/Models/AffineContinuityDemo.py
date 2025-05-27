"""
    AffineContinuityDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""

import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.Affine import Affine
from .ElutionCurveModels import egha, emga

def demo():
    x = np.arange(300)
    mu = 150
    sigma = 30
    tau = 5
    a = 0

    try:
        np.linalg.inv(np.array([[1, 1], [1, 1]]))
    except ValueError:
        print("caught ValueError")
    except Exception:
        from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
        etb = ExceptionTracebacker()
        print(etb)

    with plt.Dp():
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))

        ax1, ax2 = axes[0,:]
        for a in np.linspace(-5, 5, 10):
            ax1.plot(x, egha(x, 1, mu, sigma, tau, a))
            ax2.plot(x, emga(x, 1, mu, sigma, tau, a))

        ax3, ax4 = axes[1,:]
        for a in np.linspace(-1e-7, 1e-7, 10):
            ax3.plot(x, egha(x, 1, mu, sigma, tau, a))
            ax4.plot(x, emga(x, 1, mu, sigma, tau, a))

        fig.tight_layout()
        plt.show()
