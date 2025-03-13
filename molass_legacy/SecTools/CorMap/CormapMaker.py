# coding: utf-8
"""
    CormapMaker.py

    Copyright (c) 2021-2022, SAXS Team, KEK-PF
"""

import numpy as np
from matplotlib import colors
from MplUtils import BoundaryNorm

class CormapMaker:
    def __init__(self, M, qv, seqno, seqname, transpose=False, from_datcmp=False):
        self.transpose = transpose
        self.from_datcmp = from_datcmp
        if from_datcmp:
            self.cormap = M
        else:
            self.cormap = self.compute_cormap(M.T if transpose else M)
        self.qv = qv
        self.seqno = seqno
        self.seqname = seqname

    def draw(self, ax, cbar_ax):
        if self.transpose:
            label = self.seqname
            f, t = self.seqno[[0,-1]]
        else:
            label = r"$q(\AA^{-1})$"
            f, t = self.qv[[0,-1]]
        ax.set_xlabel(label)
        ax.set_ylabel(label)
        if self.from_datcmp:
            """
            https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib
            """
            cmap = colors.ListedColormap(["green", "palegreen", "tomato"])
            bounds=[-1, -0.1, 0.1, 1]
            norm = BoundaryNorm(bounds, cmap.N)
        else:
            cmap = None
            norm = None
        im = ax.imshow(self.cormap, extent=[f, t, t, f], cmap=cmap, norm=norm)
        fig = ax.get_figure()
        if self.from_datcmp:
            colorbar = fig.colorbar(im, ax=cbar_ax, ticks=[-0.5, 0, 0.5])
            colorbar.ax.set_yticklabels(["good", "ok", "bad"])  # vertically oriented colorbar
        else:
            colorbar = fig.colorbar(im, ax=cbar_ax)
        return colorbar

    def compute_cormap(self, M):
        cormap = np.corrcoef(M)
        return cormap
