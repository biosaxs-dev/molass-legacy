# coding: utf-8
"""
    RankControl.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import os
import numpy as np
from bisect import bisect_right
from matplotlib.patches import Rectangle
from molass_legacy.Models.ElutionCurveModels import emg
import molass_legacy.KekLib.DebugPlot as plt

def demo():

    x = np.arange(300)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21,5))

    fig.suptitle("Rank Control Demo", fontsize=30)

    color_pairs = [('green', 'green'), ('red', 'green'), ('red', 'red')]
    for ax, cpair in zip(axes, color_pairs):
        ys = []
        tops = []
        ty = np.zeros(len(x))
        for h, mu in [(1, 130), (0.5, 170)]:
            y = emg(x, h, mu, 30, 0)
            ty += y
            ys.append(y)
            tops.append((mu, h))
        ax.plot(x, ty, color='orange', label='total elution')

        f = np.argmax(ty)
        t = f + bisect_right(-ty[f:], -ty[f]/2)

        ymin, ymax = ax.get_ylim()
        patch = Rectangle(
                (f, ymin),      # (x,y)
                t - f,          # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )

        ax.add_patch(patch)

        ranks = 0
        rankw = 0
        for k, y in enumerate(ys):
            ax.plot(x, y, ':', label='component %d' % k)
            px, py = tops[k]
            color = cpair[k]
            rankw += 1
            if color == "green":
                ranks += 1
            else:
                ranks += 2
            ax.plot(px, py, 'o', color=color)
        ax.set_title("Rank %d → Rank %d" % (ranks, rankw), fontsize=20)
        ax.legend(fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    plt.show()
