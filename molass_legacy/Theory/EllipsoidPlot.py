"""
    Theory.EllisoidPlot.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn
from .RotationMatrix import rand_rotation_matrix, rotate_3d_mesh
import molass_legacy.KekLib.DebugPlot as plt


def ellipsoid_xyz(center=(0,0,0), a=1, b=1, c=1, n=20):
    """
    https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    """
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))
    return center[0]+x, center[1]+y, center[2]+z

def demo1(seed=1111):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    R = 0.3
    Rg = np.sqrt(3/5)*R
    # radii = np.random.uniform(0.5, 1.5, 3)*R
    radii = np.array([2, 3, 4])
    scale = Rg/np.sqrt(np.sum(radii**2)/5)
    radii_ = radii*scale

    v0 = 4*np.pi*np.prod(radii_)/3
    ax.set_title(r"$N=16, V=4^3=64, v_1=\frac{V}{N}=4, v_0=\frac{4}{3}\pi a b c  =%.3g$" % v0, fontsize=20)

    np.random.seed(seed)
    purtubations = np.random.uniform(-0.5, 0.5, (16,3))
    for k, center in enumerate([
                    (0.5, 0.5, 0.5), (2.5, 0.5, 0.5), (0.5, 0.5, 2.5), (2.5, 0.5, 2.5),
                    (1.5, 1.5, 1.5), (3.5, 1.5, 1.5), (1.5, 1.5, 3.5), (3.5, 1.5, 3.5), 
                    (0.5, 2.5, 0.5), (2.5, 2.5, 0.5), (0.5, 2.5, 2.5), (2.5, 2.5, 2.5),
                    (1.5, 3.5, 1.5), (3.5, 3.5, 1.5), (1.5, 3.5, 3.5), (3.5, 3.5, 3.5), 
                    ]):
        p = purtubations[k]
        center_ = center+p
        s = ellipsoid_xyz(center_, *radii_)
        M = rand_rotation_matrix()
        rs = rotate_3d_mesh(M, center_, *s)
        ax.plot_surface(*rs, color='b')

    x, y, z = np.indices((4, 4, 4))
    cubes = x + y + z  >= 0
    colors = np.array([[['#1f77b430']*4]*4]*4)

    ax.voxels(cubes, facecolors=colors)

    fig.tight_layout()
    plt.show()
