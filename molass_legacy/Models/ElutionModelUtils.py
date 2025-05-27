"""

    ElutionModelUtils.py

    Copyright (c) 2017-2024, SAXS Team, KEK-PF

"""
from bisect import bisect_right
import numpy as np
# from scipy.interpolate import UnivariateSpline
from .InvertibleSpline import InvertibleSpline as UnivariateSpline

def compute_4moments(x, y):
    """
    note that this is diffrent from the one in Peaks.ElutionModels
    in the sense that this one returns W as well
    """
    W = np.sum(y)
    M1 = np.sum(x*y)/W              # raw moment
    M2 = np.sum(y*(x-M1)**2)/W      # central moment
    M3 = np.sum(y*(x-M1)**3)/W      # central moment
    return W, M1, M2, M3

def get_xies_from_height_ratio(alpha, x, y, max_y=None, debug=False):
    if max_y is None:
        max_y = np.max(y)

    alpha_y = max_y*alpha
    safe_range = y > alpha_y*0.5
    x_ = x[safe_range]
    y_ = y[safe_range]

    if debug and len(y_) == 0:
        import molass_legacy.KekLib.DebugPlot as plt
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.plot(x, y)
            fig.tight_layout()
            plt.show()

    j = np.argmax(y_)       # ValueError: attempt to get argmax of an empty sequence

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        # to plot UnivariateSpline below failure with pH6
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("get_xies_from_height_ratio debug")
            ax.plot(x, y)
            for k, (f, t) in enumerate([(0, j-1), (j+1, len(x))]):
                xs = x_[f:t]
                ys = y_[f:t]
                ax.plot(xs, ys, ":")
            fig.tight_layout()
            plt.show()

    ret_xes = []
    for k, (f, t) in enumerate([(0, j-1), (j+1, len(x))]):
        xs = x_[f:t]
        ys = y_[f:t]
        try:
            if k == 0:
                spline = UnivariateSpline(ys, xs, s=0, ext=3)       # make the inverse function
            else:
                spline = UnivariateSpline(-ys, xs, s=0, ext=3)      # x must be strictly increasing
                alpha_y *= -1
            ret_xes.append(spline(alpha_y).item())    # numpy.ndarray.item: array to scalar
        except:
            # this case may be ignored
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(None, "get_xies_from_height_ratio: ", n=10)

    if debug:
        print("ret_xes=", ret_xes)
        spline = UnivariateSpline(x, y, s=0, ext=3)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("x_from_height_ratio_impl debug")
            ax.plot(x, y)
            ax.plot(x_, y_)
            for px in ret_xes:
                ax.plot(px, spline(px), "o", color="yellow")
            fig.tight_layout()
            plt.show()

    return ret_xes

def x_from_height_ratio_impl(func, ecurve, alpha, *params, needs_ymax=False, full_params=False,
                             x=None,
                             fx=None,   # currently used for Stochastic Models
                             debug=False):
    assert alpha > 0
    if x is None:
        x = ecurve.x
    if fx is None:
        fx = x
    if full_params:
        # for EDM, task: consider always doing it this way
        y = func(fx, *params)
    else:
        y = func(fx, 1, *params)

    max_y = np.max(y) if needs_ymax else 1       # EDM needs_ymax because its max(y) will not neccesarily be near 1

    return get_xies_from_height_ratio(alpha, x, y, max_y=max_y, debug=debug)
