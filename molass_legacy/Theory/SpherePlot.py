"""
    Theory.SpherePlot.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn


def sphere_xyz(center=(0,0,0), r=1, n=20, u=None, v=None):
    """
    https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    """
    if u is None:
        u = np.linspace(0, 2 * np.pi, n)
    if v is None:
        v = np.linspace(0, np.pi, n)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    return center[0]+x, center[1]+y, center[2]+z


def demo1(seed=1111):
    import molass_legacy.KekLib.DebugPlot as plt

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    R = 0.3
    v0 = 4*np.pi*R**3/3
    ax.set_title(r"$N=16, V=4^3=64, v_1=\frac{V}{N}=4, v_0=\frac{4}{3}\pi R^3=%.3g$" % v0, fontsize=20)

    np.random.seed(seed)
    purtubations = np.random.uniform(-0.5, 0.5, (16,3))
    for k, center in enumerate([
                    (0.5, 0.5, 0.5), (2.5, 0.5, 0.5), (0.5, 0.5, 2.5), (2.5, 0.5, 2.5),
                    (1.5, 1.5, 1.5), (3.5, 1.5, 1.5), (1.5, 1.5, 3.5), (3.5, 1.5, 3.5), 
                    (0.5, 2.5, 0.5), (2.5, 2.5, 0.5), (0.5, 2.5, 2.5), (2.5, 2.5, 2.5),
                    (1.5, 3.5, 1.5), (3.5, 3.5, 1.5), (1.5, 3.5, 3.5), (3.5, 3.5, 3.5), 
                    ]):
        p = purtubations[k]
        s = sphere_xyz(center+p, R)
        ax.plot_surface(*s, color='b')

    x, y, z = np.indices((4, 4, 4))
    cubes = x + y + z  >= 0
    colors = np.array([[['#1f77b430']*4]*4]*4)

    ax.voxels(cubes, facecolors=colors)

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    import sys
    import seaborn as sns
    sns.set_theme()
    sys.path.append("../lib")
    demo1()