# coding: utf-8
"""
    GammaVisualizer.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
# from matplotlib.patches import Polygon
from matplotlib import colors as mcolors
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.OurMatplotlib import get_facecolors

class GammaVisualizer:
    def __init__(self):
        pass

    def draw_gamma_3d(self, ax, gamma):
        num_components = gamma.shape[1]
        zs = np.arange(num_components)
        xs = np.arange(gamma.shape[0])
        verts = []
        for k in range(num_components):
            ys = gamma[:,k]
            # ys[0], ys[-1] = 0, 0
            verts.append([(0,0)] + list(zip(xs, ys)) + [(xs[-1], 0)])
        poly = PolyCollection(verts, facecolors=get_facecolors(num_components, 2))
        poly.set_alpha(0.7)
        ax.add_collection3d(poly, zs=zs, zdir='y')
        ax.set_xlim3d(0, xs[-1])
        ax.set_ylim3d(-0.5, num_components-0.5)
        ax.set_zlim3d(0, 1)
        return poly, verts

    def show(self, gamma):
        plt.push()
        fig = plt.figure(figsize=(18,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(r'$\gamma$ Visualization', fontsize=20)
        self.draw_gamma_3d(ax, gamma)
        fig.tight_layout()
        plt.show()
        plt.pop()
