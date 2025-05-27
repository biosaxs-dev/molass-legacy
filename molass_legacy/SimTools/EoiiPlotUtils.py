"""
    EoiiPlotUtils.py.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt

def plot_eoii(qv, jv, D, P, C, axes):

    i = bisect_right(qv, 0.02)

    cy = C[0,:]
    by = C[1,:]
    mu = np.argmax(cy)
    py = D[:,mu]
    aq, bq = P.T

    P_ = D @ np.linalg.pinv(C)
    aq_, bq_ = P_.T

    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    ax1.plot(jv, D[i,:], label="apparent data")
    ax1.plot(jv, aq[i]*cy, label="concentration")
    ax1.plot(jv, bq[i]*by, color="pink", label="eoii")
    ax1.legend()

    ax3.set_yscale("log")
    for ax in ax2, ax3:

        ax.plot(qv, py, label="apparent data")
        ax.plot(qv, aq, label="true aq")
        ax.plot(qv, bq, color="pink", label="true bq")
        ax.plot(qv, aq_, ":", label="solved aq")
        ax.plot(qv, bq_, ":", label="solved bq")

        ymin, ymax = ax2.get_ylim()
        ax.set_ylim(ymin, ymax)

        q = qv[i]
        ax.plot([q,q], [ymin, ymax], color="yellow", label="q=0.02")

        ax.legend()
