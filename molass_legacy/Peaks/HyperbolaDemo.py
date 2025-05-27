# coding: utf-8
"""

    HyperbolaDemo.py

    Copyright (c) 2021, SAXS Team, KEK-PF

"""
import numpy as np
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt
from .Hyperbola import RotatedHyperbola

def demo():
    from .ElutionModels import gaussian

    x = np.linspace(0, 200, 201)
    height = 3
    gy = gaussian(x, height, 100, 20)
    peak_slice = slice(80, 120)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(21,11))
    fig.suptitle("Rotated Hyperbola Local Peak Model", fontsize=20)

    for i, angles in enumerate([[0], [0, -3, 3]]):
        for ax, a in zip(axes[i,:], [0.2, 1, 5]):
            ax.set_title("a=%g (related to curvature)" % a, fontsize=16)
            ax.plot(x, gy, lw=3, label="gaussian")
            for r in angles:
                ptx, pty= 100,height
                rh = RotatedHyperbola(ptx, pty, a=a, hbw=30, deg=r)
                # rh.fit(x[peak_slice], gy[peak_slice])
                ax.plot(x, rh(x), label="hyperbola rotation(%g°)" % r)
                ax.plot(ptx, pty, 'o', color='red')
                ax.set_ylim(-2, 4)
            ax.legend(fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()

def demo_real(in_folder, pno=0):
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
    ls, pt, rs = rp.get_peaks()[pno]
    print((ls, pt, rs))
    hw = (pt-ls + rs-pt)//4
    by = np.min(y[[pt-hw, pt+hw]])
    ls_ = bisect_right(y[0:pt], by)
    rs_ = pt + bisect_right(-y[pt:], -by)
    slice_ = slice(pt-hw, pt+hw)
    # slice_ = slice(ls_, rs_)
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

    # fx, fy = x[ls:rs], y[ls:rs]
    fx, fy = x_, y_

    try:
        hbw = ((x[pt] - x[ls]) + (x[rs] - y[pt]))
        rh = RotatedHyperbola(x[pt], y[pt], hbw=hbw)
        rh.fit(fx, fy)
        success = True
    except:
        success = False
    fig, ax = plt.subplots()
    ax.plot(x, y)
    if False:
        for rad in [1/2]:
            ax.plot(x_, rot_hyperbola(x_, 1.1, 1, 0, 100, 1.5, rad*np.pi))

    if success:
        ax.plot(fx, rh(fx))
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
