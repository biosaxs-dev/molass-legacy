from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

point = np.array([1, 2, 3])
normal = np.array([1, 1, 2])

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d = -point.dot(normal)

# create x,y
xx, yy = np.meshgrid(range(10), range(10))

# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

# plot the surface
plt3d = plt.figure().add_subplot(111, projection='3d')

Gx, Gy = np.gradient(xx * yy)  # gradients with respect to x and y
G = (Gx ** 2 + Gy ** 2) ** .5  # gradient magnitude
N = G / G.max()  # normalize 0..1

plt3d.plot_surface(xx, yy, z, rstride=1, cstride=1,
                    facecolors=cm.jet(N),
                    linewidth=0, antialiased=False, shade=False,
                    )
plt.show()
