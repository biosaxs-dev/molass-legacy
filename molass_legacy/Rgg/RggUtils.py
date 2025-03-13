# coding: utf-8
"""
    Rgg.RggUtils.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np

def normal_pdf(x, mu, sigma):
    return np.exp(-1/2*((x - mu)/sigma)**2)/(sigma*np.sqrt(np.pi))

def convert_to_probabilitic_data(x, y, rg, max_y=None, num_precision=100):
    data = []
    if max_y is None:
        max_y = np.max(y)
    for x_, y_, rg_ in zip(x, y, rg):
        data += [(x_, rg_)]*int(y_/max_y*num_precision)
    return np.array(data)

"""
c.f.
https://stackoverflow.com/questions/43926473/convert-plot-to-a-surface-plot-matplotlib
"""
def plot_histogram_2d(ax, x_, y_, rg, max_y, num_precision=100):
    z_ = y_/max_y*num_precision
    y_ = rg
    i = np.arange(len(x_))
    ii, jj = np.meshgrid(i, [0,1])
    xx = x_[ii]
    yy = y_[ii]
    zz = z_[jj]
    zz[1,:] = z_
    ax.plot_surface(xx, yy, zz, color='green', alpha=0.3, edgecolor='green')
    ax.plot(x_, y_, z_, color='orange')
