# coding: utf-8
"""

    HyperbolaFit.py

    Copyright (c) 2021, SAXS Team, KEK-PF

"""
import numpy as np
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt

USE_MINIMIZE = True
if USE_MINIMIZE:
    from scipy.optimize import minimize
else:
    from scipy.optimize import curve_fit

"""
    c.f.
    Fit a curve for data made up of two distinct regimes
"""
def hyperbola(x, a, b, c, d, e):
    """ hyperbola(x) with parameters
        a/b = asymptotic slope
         c  = curvature at vertex
         d  = offset to vertex
         e  = vertical offset
    """
    return a*np.sqrt((b*c)**2 + (x-d)**2)/b + e

def rot_hyperbola(x, a, b, c, d, e, th):
    pars = a, b, c, 0, 0 # do the shifting after rotation
    xd = x - d
    hsin = hyperbola(xd, *pars)*np.sin(th)
    xcos = xd*np.cos(th)
    return e + hyperbola(xcos - hsin, *pars)*np.cos(th) + xcos - hsin


class RotatedHyperbola:
    def __init__(self, x, y, pt=None):
        if pt is None:
            h0 = 1, 3, 0, 100, 1.5, 0.5*np.pi
        else:
            ptx, pty = pt
            h0 = pty*10/x[-1], 1, 35, ptx, pty*1.6, 0.5*np.pi
        if USE_MINIMIZE:
            self.debug = True
            def obj_func(p):
                y_ = rot_hyperbola(x, *p)
                if self.debug:
                    plt.push()
                    fig, ax = plt.subplots()
                    ax.plot(x, y)
                    ax.plot(x, y_)
                    ax.plot(ptx, pty, 'o', color='red')
                    self.debug = plt.show()
                    plt.pop()
                return np.sum((y_ - y)**2)
            result = minimize(obj_func, h0)
            h = result.x
        else:
            h, hcov = curve_fit(rot_hyperbola, x, y, h0)
        self.h = h

    def get_peak_top(self):
        ptx = self.h[3]
        pty = rot_hyperbola(ptx, *self.h)
        return ptx, pty

    def __call__(self, x):
        return rot_hyperbola(x, *self.h)

def demo():
    from .ElutionModels import gaussian

    x = np.linspace(0, 200, 201)
    y = gaussian(x, 1, 100, 30)
    slice_ = slice(70, 130)
    x_ = x[slice_]
    y_ = y[slice_]
    rh = RotatedHyperbola(x_, y_)
    print('rh.h=', rh.h)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    if False:
        for rad in [1/2]:
            ax.plot(x_, rot_hyperbola(x_, 1.1, 1, 0, 100, 1.5, rad*np.pi))
    ax.plot(x_, rh(x_))
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax*1.1)
    plt.show()

def demo_real(in_folder):
    from matplotlib.patches import Rectangle
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Peaks.RobustPeaks import RobustPeaks
    sp = StandardProcedure()
    sp.load(in_folder, debug=False)
    sd = sp.get_corrected_sd(proxy=False)
    xr_curve = sd.get_xray_curve()
    x = xr_curve.x
    y = xr_curve.y
    rp = RobustPeaks(x, y)
    ls, pt, rs = rp.get_peaks()[1]
    print((ls, pt, rs))
    hw = (pt-ls + rs-pt)//4
    by = np.min(y[[pt-hw, pt+hw]])
    ls_ = bisect_right(y[0:pt], by)
    rs_ = pt + bisect_right(-y[pt:], -by)
    # slice_ = slice(pt-hw, pt+hw)
    slice_ = slice(ls_, rs_)
    x_ = x[slice_]
    y_ = y[slice_]

    if True:
        plt.push()
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(x_, y_)
        ax.plot(x[pt],y[pt], 'o', color='red')
        plt.show()
        plt.pop()

    try:
        rh = RotatedHyperbola(x_, y_, pt=(x[pt],y[pt]))
        print('rh.h=', rh.h)
        success = True
    except:
        success = False
    fig, ax = plt.subplots()
    ax.plot(x, y)
    if False:
        for rad in [1/2]:
            ax.plot(x_, rot_hyperbola(x_, 1.1, 1, 0, 100, 1.5, rad*np.pi))

    if success:
        ax.plot(x_, rh(x_))
        ptx, pty = rh.get_peak_top()
        ax.plot(ptx, pty, 'o', color='red')

    ymin, ymax = ax.get_ylim()
    ymax_ = ymax*1.1
    ax.set_ylim(ymin, ymax_)

    f, t = x[[ls,rs]]
    p = Rectangle(
            (f, ymin),      # (x,y)
            t - f,          # width
            ymax_ - ymin,    # height
            facecolor   = 'cyan',
            alpha       = 0.2,
        )
    ax.add_patch(p)

    plt.show()
