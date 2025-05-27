"""
    SecTheory.ColumnIllust.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
import sys
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.interpolate import UnivariateSpline
from matplotlib.patches import Rectangle, Circle

R = 1/12
dr = R*0.3
r = R*0.7
width = 2*r
height = 2*r
poresize_max = 200

def plot_column_illust(ax, poresizes, colors):
    ax.set_axis_off()

    num_iters = 36//len(poresizes)
    poreradii = np.array(poresizes * num_iters)
    porecolors = np.array(colors * num_iters)
    print(len(poreradii))
    indeces = np.arange(36)
    np.random.shuffle(indeces)
    print(poreradii[indeces])

    patches = []
    k = 0
    for i in range(6):
        for j in range(6):
            rectangle = Rectangle((dr+i/6, dr+j/6), width, height, facecolor="black")
            ax.add_patch(rectangle)
            k_ = indeces[k]
            pore = Circle((R+i/6, R+j/6), r*poreradii[k_]/poresize_max, color=porecolors[k_])
            ax.add_patch(pore)
            k += 1

def plot_column_legend(ax, poresizes, colors):
    ax.set_axis_off()

    dy = 0.2
    x = 0.2

    for k, (size, color) in enumerate(zip(poresizes, colors)):
        y = 0.7 - k*R*2
        rectangle = Rectangle((dr+x, dr+y), width, height, facecolor="black")
        ax.add_patch(rectangle)
        pore = Circle((R+x, R+y), r*size/poresize_max, color=color)
        ax.add_patch(pore)
        ax.text(R+x+0.2, R+y, "Poresize: %3.0f" % size, va="center", fontsize=16)

def demo():
    import molass_legacy.KekLib.DebugPlot as plt

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(18,4.5))

    plot_column_illust(ax1, [76], ["white"])
    plot_column_illust(ax2, [90, 40], ["orange", "yellow"])
    plot_column_illust(ax3, [100, 76, 40], ["pink", "white", "yellow"])
    plot_column_legend(ax4, [100, 90, 76, 40], ["pink", "orange", "white", "yellow"])

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
