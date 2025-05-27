"""
    CustomSigmoid.py


"""
import numpy as np
from scipy.optimize import curve_fit
import molass_legacy.KekLib.DebugPlot as plt
from .DiniSigmoid import dini_sigmoid, dini_sigmoid_inv

HW = 50

def custom_sigmoid(x, L, x0, k, b):
    return L*dini_sigmoid(x - x0, k) + b

def custom_sigmoid_inv(y, L, x0, k, b):
    """
    y = L*dini_sigmoid(x - x0, k) + b
    dini_sigmoid(x - x0, k) = (y - b)/L
    x - x0 = dini_sigmoid_inv((y - b)/L, k)
    x = dini_sigmoid_inv((y - b)/L, k) + x0
    """
    return dini_sigmoid_inv((y - b)/L, k) + x0

def get_safer_point_impl(x, a_y, ratio, a_popt, debug=False):
    L, x0, k, b, s1, s2 = a_popt

    nx = x - x0
    y = a_y - np.hstack([s1*nx[nx < 0], s2*nx[nx >= 0]])

    limits = [int(round(p)) for p in [x0 - HW, x0 + HW]]
    slice_ = slice(max(0, limits[0]), min(len(x), limits[1]))
    x_ = x[slice_]
    y_ = y[slice_]
    xlength = x_[-1] - x_[0]
    wx_ = (x_ - x0)*2/xlength

    # k0 = -0.95 if L > 0 else 0.95
    p0 = (L/2, 0, -0.95, b + L/2)
    popt, pcov = curve_fit(custom_sigmoid, wx_, y_, p0)
    print("popt=", popt)
    assert abs(popt[2]) < 1     # k < -1 occured in Matsumura, which ended up in abnormal result

    if debug:
        plt.push()
        fig, ax = plt.subplots()
        ax.plot(wx_, y_)
        ax.plot(wx_, custom_sigmoid(wx_, *popt))
        plt.show()
        plt.pop()

    L_, x0_, k_, b_ = popt
    y1 = b_ - L_ + L_ * 2 * ratio   # note that L ～ 2L_, the ratio must be scaled from (0, 1) to (-1, 1)
    wx1 = custom_sigmoid_inv(y1, *popt)

    if debug:
        plt.push()
        fig, ax = plt.subplots()
        ax.plot(wx_, y_)
        ax.plot(wx_, custom_sigmoid(wx_, *popt))
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax)
        ax.plot([xmin, xmax], [y1, y1])
        ax.plot(wx1, y1, "o", color="red")
        plt.show()
        plt.pop()

    x1 = wx1/2*xlength + x0

    if debug:
        from .Sigmoid import sigmoid
        a_popt_ = a_popt[0:4]
        plt.push()
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(x, sigmoid(x, *a_popt_))
        ax.plot(x_, custom_sigmoid(wx_, *popt))
        ax.plot(x1, y1, "o", color="red")
        plt.show()
        plt.pop()

    s = s1 if x1 < x0 else s2
    r_y1 = y1 - s*(x1 - x0)

    return x1, r_y1
