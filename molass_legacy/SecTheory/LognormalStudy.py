"""
    SecTheory.LognormalStudy.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.stats import lognorm 

R2PI = np.sqrt(2*np.pi)

def lognormal_pdf(x, mu, sigma):
    return 1/(x*sigma*R2PI) * np.exp(-(np.log(x) - mu)**2/(2*sigma**2))

def demo():
    import molass_legacy.KekLib.DebugPlot as plt

    x = np.linspace(0.01, 10, 200)

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        ax1.set_title("Numpy")
        ax2.set_title("Scipy")

        for mu, style in [(0, "-"), (1, ":")]:
            for sigma in [1, 0.5, 0.25]:
                ax1.plot(x, lognormal_pdf(x, mu, sigma), style, label=r"$\mu=%g, \sigma=%g$" % (mu, sigma))
                ax2.plot(x, lognorm.pdf(x, sigma, mu, np.exp(mu)), style, label=r"$\mu=%g, \sigma=%g$" % (mu, sigma))
                mode = np.exp(mu - sigma**2)
                ax1.plot(mode, lognormal_pdf(mode, mu, sigma), "o", color="red")
                mx = mode + mu
                ax2.plot(mx, lognorm.pdf(mx, sigma, mu, np.exp(mu)), "o", color="red")
                for p, color in [(0.01, "yellow"), (0.99, "cyan")]:
                    px = lognorm.ppf(p, sigma, mu, np.exp(mu))
                    if px < 10:
                        ax2.plot(px, lognorm.pdf(px, sigma, mu, np.exp(mu)), "o", color=color)

        ax1.legend()
        ax2.legend()

        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    import os
    import sys
    this_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.dirname(this_dir)
    sys.path.append(lib_dir)

    import seaborn
    seaborn.set()
    import molass_legacy.KekLib

    demo()
