"""
    Theory.WLS.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def npWLS(x, y, w=None):
    pass

def npWLS_spike():
    x = np.array([0, 1, 2, 3])
    y = np.array([-1, 0.2, 0.9, 2.1])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    fig, ax = plt.subplots()
    ax.plot(x, y, 'o', label='Original data', markersize=10)
    ax.plot(x, m*x + c, 'r', label='Fitted line')
    ax.legend()
    plt.show()
