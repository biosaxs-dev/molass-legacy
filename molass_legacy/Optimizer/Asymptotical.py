"""
    Asymptotical.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt

def asymptotical(x, y, dinit, rate, return_d=False):
    """
    x : log value like guinier deviation
    y : log value like LRF residual

    d : should decrease as x and y impove with smaller dist(x, y)
    (x - (y + d)) : should be evaluated smaller as they improve
    """
    d = np.exp((y + dinit)*rate)
    z = d*(x - (y + d))**2

    if return_d:
        return z, d
    else:
        return z

if __name__ == '__main__':
    from matplotlib import cm

    x = np.linspace(-3, 1, 100)
    y = np.linspace(-3, 1, 100)
    # z = asymptotical(x, y, 0.5, 0.1)
    xx, yy = np.meshgrid(x, y)
    zz, dd = asymptotical(xx, yy, 0.5, 1, return_d=True)

    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)

    ax1.set_title("(x, y) Deviation Evaluation")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.plot_surface(xx, yy, zz, cmap=cm.coolwarm)

    ax2.set_title("Asymptotic Differance")
    ax2.set_xlabel("y")
    ax2.set_ylabel("d")

    ax2.plot(y, dd)

    fig.tight_layout()
    plt.show()
