"""
    Models/Stochastic/LognormalStudy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.stats import lognorm, norm
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # import seaborn as sns
    # sns.set()
    N = 1000
    """
    mode = exp(mu - sigma**2)
    log(mode) = mu - sigma**2
    """
    k = 0
    for mu, sigma, xmax in [(2, 0.5, 25), (4.5, 0.1, 300), (4.5, 0.2, 300)]:
        fig, ax = plt.subplots()
        ax.set_title("Comparison among Normal, Exp(Normal), Lognormal with %d samples" % N)
        ax.set_xlabel("Pore Size")
        ax.set_ylabel("Density")
        X = np.random.normal(mu, sigma, N)
        if k == 0:
            ax_ = ax
        else:
            ax_ = ax.twinx()
        ax.hist(X, bins=50, density=True, alpha=0.5, label="normal", color="C0")
        ax_.hist(np.exp(X), bins=100, density=True, alpha=0.5, label="exp(normal)", color="C1")
        ymin, ymax = ax.get_ylim()
        if k == 0:
            tx = -1*0.9 + xmax*0.1
            xmin = -1
        else:
            tx = -1*0.95 + xmax*0.05
            xmin = -20
        ty = ymin*0.7 + ymax*0.3
        dx = (xmax - xmin)*0.1
        dy = (ymax - ymin)*0.2
        ax.annotate( "$Normal(\mu=%g,\sigma=%g)$" % (mu, sigma), fontsize=16, xy=(tx, ty), xytext=(tx+dx, ty+dy),
                    ha='center', va='center',
                    arrowprops=dict( headwidth=5, width=0.5, color='black', shrink=0.05) )
        ax.annotate( "$Lognormal(\mu=%g,\sigma=%g)$" % (mu, sigma), fontsize=16, xy=(tx+4*dx, ty-dy), xytext=(tx+5*dx, ty),
                    ha='center', va='center',
                    arrowprops=dict( headwidth=5, width=0.5, color='black', shrink=0.05) )
        Y = np.random.lognormal(mu, sigma, N)
        ax_.hist(Y, bins=100, density=True, alpha=0.5, label="lognormal", color="C2")
        ax.set_xlim(xmin, xmax)
        x = np.linspace(xmin, xmax, 1000)
        ax.plot(x, norm.pdf(x, mu, sigma), label="normal pdf", color="C3")

        ax_.plot(x, lognorm.pdf(x, sigma, scale=np.exp(mu)), label="lognormal pdf", color="C4")
        mode = np.exp(mu - sigma**2)
        ax.axvline(mode, color='red', alpha=0.5, linestyle='--', label="$mode=exp(\mu-\sigma^2)$")
        ax.legend()
        if k > 0:
            ax_.legend(loc='center right')
        fig.tight_layout()
        plt.show()
        k += 1


