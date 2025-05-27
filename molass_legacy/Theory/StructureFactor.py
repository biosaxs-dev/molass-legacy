"""
    Theory.StructureFactor.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.integrate import quad
import molass_legacy.KekLib.DebugPlot as plt


def demo1():

    Na_pos = np.array([(0,0,0), (0,1/2,1/2), (1/2,0,1/2), (1/2,1/2,0)])
    Cl_pos = np.array([(1/2,0,0), (0,1/2,0), (0,0,1/2), (1/2,1/2,1/2)])

    fig = plt.figure(figsize=(14,7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    ax1.scatter(*Na_pos.T, color='blue')
    ax1.scatter(*Cl_pos.T, color='red')

    fig.tight_layout()
    plt.show()
