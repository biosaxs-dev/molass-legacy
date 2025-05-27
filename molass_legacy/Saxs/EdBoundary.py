# coding: utf-8
"""
    EdBoundary.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import molass_legacy.KekLib.DebugPlot as plt

SAFE_START_RATIO = 0.9
POSITIVE_SMALL_VALUE = 1e-10
BOUNDARY_RATIO = 0.05

def guess_boundary_value(data, debug=False):
    v = np.array(sorted(data.flatten()))

    v[v < POSITIVE_SMALL_VALUE] = POSITIVE_SMALL_VALUE

    n = len(v)
    start = int(n*SAFE_START_RATIO)
    v_ = np.log10(v)
    gv = np.gradient(v_)
    gv_ = gv[start:]
    gv_max = gv_.max()
    gv_boundary = gv_max*BOUNDARY_RATIO
    w = np.where(gv_ > gv_boundary)[0]
    i = start + w[0]

    if debug:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,7))
        ax1, ax2 = axes
        ax1.plot(v_)
        ax1.plot(i, v_[i], 'o', color='red')
        ax2.plot(gv)
        ax2.plot(i, gv[i], 'o', color='red')
        fig.tight_layout()
        plt.show()

    return v[i]
