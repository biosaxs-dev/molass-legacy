# coding: utf-8
"""
    LinearEquations.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.ticker as ticker
from OurManim import latex_init
import molass_legacy.KekLib.DebugPlot as plt

def show_figs():
    latex_init()

    fig, axes = plt.subplots(ncols=3, figsize=(21,7))
    a, b, c = 2, 1, 3

    fontsize = 30
    ax1, ax2, ax3 = axes
    ax1.set_title("Under-determined System", fontsize=fontsize)
    ax2.set_title("Exactly-determined System", fontsize=fontsize)
    ax3.set_title("Over-determined System", fontsize=fontsize)

    x = np.linspace(-0.5, 3, 10)
    y = a - x
    for ax in axes:
        ax.plot(x, y, label='$x+y=%g$' % a)
    for ax in axes[1:]:
        ax.plot(x, x-b, label='$x-y=%g$' % b)
    for ax in axes[2:]:
        ax.plot(x, (c-x)/2, label='$x+2y=%g$' % c)

    v = np.linalg.pinv(np.array([[1,1]]))@np.array([[a]])
    ax1.plot(*v, 'o', color='red', label=r'$ {\begin{bmatrix} 1 & 1 \end{bmatrix}}^+ \begin{bmatrix} %g \end{bmatrix} $' % a)


    v = np.linalg.pinv(np.array([[1,1], [1,-1]])) @ np.array([[a],[b]])
    ax2.plot(*v, 'o', color='red', label=r'$ {\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}}^+ \begin{bmatrix} %g \\ %g \end{bmatrix} $' % (a, b))

    v = np.linalg.pinv(np.array([[1,1], [1,-1], [1,2]])) @ np.array([[a],[b],[c]])
    ax3.plot(*v, 'o', color='red', label=r'$ {\begin{bmatrix} 1 & 1 \\ 1 & -1 \\ 1 & 2 \end{bmatrix}}^+ \begin{bmatrix} %g \\ %g \\ %g \end{bmatrix} $' % (a, b, c))

    for ax in axes:
        ax.set_xlim(0,2.5)
        ax.set_ylim(0,2.5)
        ax.set_aspect('equal')
        ax.legend(loc="upper center", fontsize=20)
        ax.xaxis.set_major_locator(ticker.FixedLocator([0,1,2]))
        ax.yaxis.set_major_locator(ticker.FixedLocator([0,1,2]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.1)
    plt.show()
