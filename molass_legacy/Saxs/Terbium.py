# coding: utf-8
"""
    Terbium.py

    slightly adapted from
        https://terbium.io/2017/12/matplotlib-3d/
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plot_cube(cube, angle=320):
    IMG_DIM = 50
    fig = plt.figure(figsize=(30/2.54, 30/2.54))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=IMG_DIM*2)
    ax.set_ylim(top=IMG_DIM*2)
    ax.set_zlim(top=IMG_DIM*2)
    draw_voxels(ax, cube)
    plt.show()

def draw_voxels(ax, data):
    data = normalize(data)
    facecolors = cm.viridis(data)
    facecolors[:,:,:,-1] = data
    facecolors = explode(facecolors)
    filled = facecolors[:,:,:,-1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
