"""
    SigmoidDemo.py

    Copyright (c) 2021-2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import curve_fit
import molass_legacy.KekLib.DebugPlot as plt
from .Sigmoid import *

def basic_demo():
    plt.push()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.set_title("Basic Sigmoids (1, 100, k, 0)", fontsize=16)
    ax2.set_title("Extended Sigmoids (-1, 100, k, 0, -0.001, 0.001)", fontsize=16)

    x = np.arange(200)
    for k in [0.1, 0.2, 0.5, 1]:
        p1 = 1, 100, k, 0
        ax1.plot(x, sigmoid(x, *p1), label="k=%g" % k)
    ax1.legend()

    for k in [0.1, 0.2, 0.5, 1]:
        p2 = -1, 100, k, 0, -0.001, 0.001
        ax2.plot(x, ex_sigmoid(x, *p2), label="k=%g" % k)

    ax2.legend()
    fig.tight_layout()
    plt.show()
    plt.pop()

def real_demo(x_curve, a_curve, a_curve2, fit_chain=False):
    # from molass_legacy.KekLib.SciPyCookbook import smooth
    from molass_legacy.Elution.CurveUtils import simple_plot
    from .PeakRegion import PeakRegion
    from .FlowChangeCandidates import get_largest_gradients

    peak_region = PeakRegion(x_curve, a_curve, a_curve2)

    x = a_curve2.x
    # y = smooth(a_curve2.y)
    y = a_curve2.y

    gy, pp3 = get_largest_gradients(y, 3, peak_region)
    print("pp3=", pp3)
    x0 = x[pp3[0]]

    stop1 = a_curve.peak_info[0][0]
    if abs(stop1 - x0) < 5:
        # as in 20170301
        stop1 = x0 + 5
    sliceL = slice(0, stop1)
    sliceR = slice(a_curve.peak_info[-1][-1], None)

    results = []
    for slice_ in [sliceL, sliceR]:

        x_ = x[slice_]
        y_ = y[slice_]

        try:
            if x0 < x_[0] or x0 > x_[-1]:
                _, pp3_ = get_largest_gradients(y_, 3, peak_region)
                x0 = x_[pp3_[0]]
            params0 = guess_bent_sigmoid(x_, y_, x0)

            if False:
                plt.push()
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
                simple_plot(ax1, a_curve)
                ax2.plot(x, y, color="gray", alpha=0.3)
                ax2.plot(x_, y_)
                ax2.plot(x_, bent_sigmoid(x_, *params0))
                fig.tight_layout()
                plt.show()
                plt.pop()

            popt, pcov = curve_fit(bent_sigmoid, x_, y_, params0)
            L = popt[0]
            error_ratio = abs(np.std(bent_sigmoid(x_, *popt) - y_)/L)
            print("error_ratio=", error_ratio)
            if error_ratio < 0.3:
                results.append((x_, y_, params0, popt))
        except Exception as exc:
            print(exc)

    with plt.Dp():
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 5))
        simple_plot(ax1, a_curve)
        ax2.plot(x, y)
        for x_, y_, params0, popt in results:
            ax2.plot(x_, ex_sigmoid(x_, *params0), label="initial guess")
            ax2.plot(x_, ex_sigmoid(x_, *popt), label="fit result")
        ax2.legend()
        ax3.plot(x, gy)
        for k, p in enumerate(pp3):
            ax3.plot(p, gy[p], "o", label=str(k))
        ax3.legend()

        fig.tight_layout()
        plt.show()

    if fit_chain:
        from .SigmoidChain import SigmoidChain
        from DataUtils import get_in_folder

        chain = SigmoidChain(x, y, pp3)
        in_folder = get_in_folder()
        with plt.Dp():
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 5))
            fig.suptitle("Sigmoid Chain Demo for %s" % in_folder, fontsize=20)
            ax1.set_title("Elution Curve at λ=280", fontsize=16)
            ax2.set_title("Elution Curve at λ=340", fontsize=16)
            ax3.set_title("Gradient Curve at λ=340", fontsize=16)
            simple_plot(ax1, a_curve)
            ax2.plot(x, y, alpha=0.5)
            # ax2.plot(x, chain(x), label="fit result")
            chain.plot_segments(ax2, label="segment-%d", lw=3)
            ax2.legend()
            ax3.plot(x, gy)
            for k, p in enumerate(pp3):
                ax3.plot(p, gy[p], "o", label=str(k))
            ax3.legend()

            fig.tight_layout()
            plt.show()
