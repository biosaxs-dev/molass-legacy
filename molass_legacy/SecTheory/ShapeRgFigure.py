"""
    SecTheory.ShapeRgFiguree.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_ellipsoid(ax, a, b, c, max_radius=None):
    coefs = (a, b, c)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
    # Radii corresponding to the coefficients:
    rx, ry, rz = 1/np.sqrt(coefs)

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    # Plot:
    # ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')
    ax.plot_surface(x, y, z, alpha=0.5)
    # ax.plot_wireframe(x, y, z)

    # Adjustment of the axes, so that they all have the same span:
    if max_radius is None:
        max_radius = max(rx, ry, rz)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

def compute_weights(a, b, c):
    s = np.power(a*b*c, -1/3)
    return a*s, b*s, c*s

def compute_Rg(a, b, c):
    Rg = np.sqrt((a**2 + b**2 + c**2)/5)
    return Rg
        
def plot_ellipsoid_rg_ratio(fig):

    
    axes = []
    for n in [141, 142, 143]:
        ax = fig.add_subplot(n, projection='3d')
        ax.set_axis_off()
        axes.append(ax)
    ax2 = fig.add_subplot(144)
   
    for k, (a, b, c) in enumerate([(0.1,1,1), (1,1,1), (10,1,1)]):
        ax = axes[k]
        a_, b_, c_ = compute_weights(a, b, c)
        Rg = compute_Rg(a_, b_, c_)
        ax.set_title("a=%.2g, b=%.2g, c=%.2g; $R_g$=%.2g" % (a_, b_, c_, Rg))
        plot_ellipsoid(ax, a_, b_, c_, max_radius=2.5)
    av = np.linspace(0.1, 10, 40)
    av_, bv_, cv_ = compute_weights(av, b, c)
    Rg = compute_Rg(av_, bv_, cv_)
    ax2.set_title("$R_g$ vs. a in (0.1, 10)")
    ax2.plot(av, Rg)
    ax2.set_xlabel("a")
    ax2.set_ylabel("$R_g$")

if __name__ == '__main__':
    import seaborn
    seaborn.set()

    fig = plt.figure(figsize=(16,4))
    fig.suptitle("Rg dependence on Shapes with the same Volume as a unity radius Sphere")
    plot_ellipsoid_rg_ratio(fig)
    fig.tight_layout()
    plt.show()
