"""
    Models.Stochastic.PsdIllust.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set_theme()

def draw_psd_illust():
    from molass_legacy.Models.Stochastic.LognormalPoreFunc import distr_func
    from molass_legacy.Models.Stochastic.LognormalUtils import compute_mu_sigma

    pv = np.linspace(0, 400, 100)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    mode, stdev = 200, 20
    mu, sigma = compute_mu_sigma(mode, stdev)
    ax = axes[1,1]
    ax.set_title("Lognormalpore")
    ax.fill_between(pv, np.zeros(len(pv)), distr_func(pv, mu, sigma))
    xmin, xmax = ax.get_xlim()

    ax = axes[0,0]
    ax.set_title("Monopore")
    ax.bar(200, 1, width=20)
    ax.set_xlim(xmin, xmax)

    ax = axes[0,1]
    ax.set_title("Dipore")
    ax.bar([70, 200], [0.3, 0.7], width=20)
    ax.set_xlim(xmin, xmax)

    ax = axes[1,0]
    ax.set_title("Tripore")
    ax.bar([70, 200, 300], [0.2, 0.5, 0.3], width=20)
    ax.set_xlim(xmin, xmax)

    for ax in axes.flatten():
        ax.set_xlabel(r"Poresize ($\AA$)")
        ax.set_ylabel("Density")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    import os
    this_dir = os.path.dirname( os.path.abspath( __file__ ) )
    sys.path.append( this_dir + '/../..' )
    draw_psd_illust()