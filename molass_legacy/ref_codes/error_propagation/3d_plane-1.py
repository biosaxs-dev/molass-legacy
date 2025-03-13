""" borrowd from:
    Plot a plane based on a normal vector and a point in Matlab or matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

point  = np.array([1, 2, 3])
normal = np.array([1, 1, 2])

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d = -point.dot(normal)

# create x,y
xx, yy = np.meshgrid(range(10), range(10))

print( 'xx=', xx )
print( 'yy=', yy )

# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

# plot the surface
plt3d = plt.figure().add_subplot(111, projection='3d')
plt3d.plot_surface(xx, yy, z, alpha=0.5 )
plt.show()
