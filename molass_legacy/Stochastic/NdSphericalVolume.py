"""
    Stochastic.NdSphericalVolume.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle, Circle
from scipy.special import gamma
import seaborn as sns
sns.set_theme()

fig = plt.figure(figsize=(18,4))
ax0 = fig.add_subplot(141)
ax1 = fig.add_subplot(142, projection='3d')
ax2 = fig.add_subplot(143)
ax3 = fig.add_subplot(144)
ax3.set_yscale('log')

ax0.set_title(r"Area Ratio of 2D Circle/Square: $\pi\cdot 2^{-2} \approx %.2g$" % (np.pi/4))
ax0.set_aspect('equal')
square = Rectangle((-1,-1), 2, 2, color='red', alpha=0.1)
ax0.add_patch(square)
circle = Circle(xy=(0,0), radius=1.0, color='blue', alpha=0.1)
ax0.add_patch(circle)
ax0.set_xlim(-1.2, 1.2)
ax0.set_ylim(-1.2, 1.2)

v3 = 4/3*np.pi
ax1.set_title(r"Volume Ratio of 3D Sphere/Cube: $\frac{4}{3}\pi \cdot 2^{-3} \approx %.2g$" % (v3 * 2**(-3)))

# See https://stackoverflow.com/questions/73679251/how-can-i-plot-a-cube-in-matplotlibpython-when-the-8-vertices-are-given-in-num
# Unit cube coordinates
x = np.array([0, 1, 1, 0, 0, 1, 1, 0])*2 - 1
y = np.array([0, 0, 1, 1, 0, 0, 1, 1])*2 - 1
z = np.array([0, 0, 0, 0, 1, 1, 1, 1])*2 - 1

# Face IDs
vertices = [[0,1,2,3],[1,5,6,2],[3,2,6,7],[4,0,3,7],[5,4,7,6],[4,5,1,0]]

tupleList = list(zip(x, y, z))

poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
# ax1.scatter(x,y,z)
ax1.add_collection3d(Poly3DCollection(poly3d, facecolors='r', linewidths=1, alpha=0.1))

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax1.plot_surface(x, y, z)

# Set an equal aspect ratio
ax1.set_aspect('equal')

x_ = np.arange(2,40)
y_ = np.power(np.pi, x_/2)/gamma(x_/2 + 1) * np.power(2.0, -x_)

for ax, scale in [(ax2, 'Linear'), (ax3, 'Log')]:
    ax.set_title(r"Volume Ratio of N-D Sphere/Cube (%s scale)" % scale)
    ax.set_xlabel("Dimension: N")
    ax.set_ylabel("Volume")
    ax.plot(x_, y_, label=r"$VR=\frac{\pi^{n/2}}{\Gamma(n/2 + 1)} \cdot 2^{-n}$")
    ax.plot(3, v3/8, "o", color="red", label="3D Sphere/Cube")
    ax.legend()

fig.tight_layout()
plt.show()