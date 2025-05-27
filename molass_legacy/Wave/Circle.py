"""
    Wave.Circle.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

class BeamAnim:
    def __init__(self, ):
        pass


def circle_xy(cx, cy, r, n=100):
    t = np.linspace(0, np.pi*2, n)
    return cx + r*np.cos(t), cy + r*np.sin(t)

def demo1():
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(21, 7))

    ax1.plot(0, 0, 'o')

    z = np.exp(np.pi/4*1j)
    print(z)

    ax1.plot(z.real, z.imag, 'o')
    x, y = circle_xy(0, 0, 1)
    ax1.plot(x, y)

    fig.tight_layout()
    plt.show()
