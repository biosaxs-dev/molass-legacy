# coding: utf-8
"""
    SaxsSamples.py

    Copyright (c) 2019-2021, SAXS Team, KEK-PF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def get_plot_axis():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax

def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

class ObjectVoxels:
    def __init__(self, shape):
        self.shape = shape_ = np.array(shape)
        n = np.max(shape_)/2
        x, y, z = np.indices(shape_+1)/n - 1
        self.xyz = (x, y, z)
        mx = midpoints(x)
        my = midpoints(y)
        mz = midpoints(z)
        self.points = (mx, my, mz)

    def get_xyz(self):
        return self.xyz

    def get_data(self, density=1):
        data = np.zeros(self.shape)
        data[self.bool_array] = density
        return data

class BallVoxels(ObjectVoxels):
    def __init__(self, shape=(64,64,64), center=(0,0,0), radius=1):
        ObjectVoxels.__init__(self, shape)
        mx, my, mz = self.points
        cx, cy, cz = center
        self.bool_array = (mx-cx)**2 + (my-cy)**2 + (mz-cz)**2 < radius**2

class EllipsoidVoxels(ObjectVoxels):
    def __init__(self, shape=(64,64,64), center=(0,0,0), radii=(1,1,1)):
        ObjectVoxels.__init__(self, shape)
        mx, my, mz = self.points
        cx, cy, cz = center
        a, b, c = radii
        self.bool_array = ((mx-cx)/a)**2 + ((my-cy)/b)**2 + ((mz-cz)/c)**2 < 1

class DiscVoxels(ObjectVoxels):
    def __init__(self, shape=(64,64,64), center=(0,0,0), radius=1, height=0.05):
        ObjectVoxels.__init__(self, shape)
        mx, my, mz = self.points
        cx, cy, cz = center
        in_circle = (mx-cx)**2 + (my-cy)**2 < radius**2
        in_depth = (mz-cz)**2 < height**2
        self.bool_array = np.logical_and(in_circle, in_depth)

class TorusVoxels(ObjectVoxels):
    def __init__(self, shape=(64,64,64), center=(0,0,0), R=0.8, r=0.2):
        ObjectVoxels.__init__(self, shape)
        mx, my, mz = self.points
        cx, cy, cz = center
        self.bool_array = (np.sqrt((mx-cx)**2 + (my-cy)**2) - R)**2 + (mz-cz)**2 < r**2
