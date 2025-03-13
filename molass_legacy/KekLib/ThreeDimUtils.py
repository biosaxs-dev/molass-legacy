# coding: utf-8
"""
    ThreeDimUtils.py

    Copyright (c) 2018-2021, SAXS Tam, KEK-PF
"""
import numpy                as np

def compute_plane( A, B, C, x, y ):
    """
    base_plane = [
        [ A * x[0] + B * y[0], A * x[0] + B * y[1], A * x[0] + B * y[2] ],
        [ A * x[1] + B * y[0], A * x[1] + B * y[1], A * x[1] + B * y[2] ],
        [ A * x[2] + B * y[0], A * x[2] + B * y[1], A * x[2] + B * y[2] ],
        ...
        ] + C
                = np.dot( [
                            [ A * x[0], 1 ],
                            [ A * x[1], 1 ],
                            [ A * x[2], 1 ],
                              ...
                          ],
                          [
                            [ 1,        1,        1,        ... ],
                            [ B * y[0], B * y[1], B * y[2], ... ].
                            ...
                          ] )

    """
    return np.dot( np.vstack( [ A * x, np.ones( len(x) ) ] ).T, np.vstack( [ np.ones( len(y) ), B * y ] ) ) + C

def neaten_axes(axes):
    axis_lims = []
    for ax in axes:
        axis_lims.append((ax.get_xlim(), ax.get_ylim(), ax.get_zlim()))
    axis_lim_array = np.array(axis_lims)
    axis_limits = []
    for j  in range(3):
        vmin = np.min(axis_lim_array[:,j,0])
        vmax = np.max(axis_lim_array[:,j,1])
        axis_limits.append((vmin, vmax))
    for ax in axes:
        ax.set_xlim(*axis_limits[0])
        ax.set_ylim(*axis_limits[1])
        ax.set_zlim(*axis_limits[2])
