# coding: utf-8
"""
    GeometryUtils.py

    Copyright (c) 2017-2022, Masatsuyo Takahashi, KEK-PF
"""
import numpy as np

def rotate( th, wx, wy ):
    cx  = wx[-1]
    cy  = wy[-1]
    wx_ = wx - cx
    wy_ = wy - cy
    c   = np.cos( th )
    s   = np.sin( th )
    return cx + ( wx_*c - wy_* s ), cy + ( wx_*s + wy_* c )

def rotated_argmin(theta, wy, debug=False):
    wy_ = wy/(np.max(wy) - np.min(wy))
    n = len(wy)
    wx_ = np.arange(n)/(n-1)
    rx, ry = rotate(theta, wx_, wy_)
    m = np.argmin(ry)
    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        print('m=', m)
        degrees = 180*theta/np.pi
        plt.push()
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 7))
        fig.suptitle("rotated_argmin debug", fontsize=20)
        ax1.set_title("Normalized Data", fontsize=16)
        ax2.set_title("Rotated %+.4g degrees" % degrees, fontsize=16)
        ax1.plot(wx_, wy_)
        ax1.plot(wx_[m], wy_[m], 'o', color='red')
        ax2.plot(rx, ry)
        ax2.plot(rx[m], ry[m], 'o', color='red')
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.show()
        plt.pop()
    return m

def rotated_argminmax(theta, wy, debug=False):
    wy_ = wy/(np.max(wy) - np.min(wy))
    n = len(wy)
    wx_ = np.arange(n)/(n-1)
    rx, ry = rotate(theta, wx_, wy_)
    m1 = np.argmin(ry)
    m2 = np.argmax(ry)

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        degrees = 180*theta/np.pi
        plt.push()
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 7))
        fig.suptitle("rotated_argmin debug", fontsize=20)
        ax1.set_title("Normalized Data", fontsize=16)
        ax2.set_title("Rotated %+.4g degrees" % degrees, fontsize=16)
        ax1.plot(wx_, wy_)
        ax2.plot(rx, ry)

        for m in [m1, m2]:
            print('m=', m)
            ax1.plot(wx_[m], wy_[m], 'o', color='red')
            ax2.plot(rx[m], ry[m], 'o', color='red')

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.show()
        plt.pop()

    return m1, m2

def polygon_area_centroid(polygon, num_division=10):
    path = polygon.get_path()
    xy = path.vertices
    xmin = np.min(xy[:,0])
    xmax = np.max(xy[:,0])
    ymin = np.min(xy[:,1])
    ymax = np.max(xy[:,1])
    x = np.linspace(xmin, xmax, num_division+1)
    y = np.linspace(ymin, ymax, num_division+1)
    x_ = (x[1]-x[0])/2 + x[:-1]
    y_ = (y[1]-y[0])/2 + y[:-1]
    xx, yy = np.meshgrid(x_, y_)
    points = np.array(list(zip(xx.flatten(), yy.flatten())))
    inside = path.contains_points(points)
    in_points = points[inside]
    return np.average(in_points, axis=0)
