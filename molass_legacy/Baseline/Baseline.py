"""
    Baseline.py

    Copyright (c) 2020-2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.Baseline.ScatteringBaseline import ScatteringBaseline
from .Convex import integrative_curve_convex

LPM_PERCENTILE = 10
NEG_RATIO_LIMIT = 0.3   # < 0.41 for 20200624_2 UV, > 0.16 20200624_2 Xray
END_WIDTH = 50

def integrative_curve(y, baseline=None, epsilon=1e-4, p_final=LPM_PERCENTILE, raise_=False, end_slices=None, return_ratio=False, debug=False):
    if end_slices is None:
        end_slices = (slice(0,END_WIDTH), slice(-END_WIDTH, None))
    assert len(end_slices) == 2

    zeros = np.zeros(len(y))
    if baseline is None:
        x = np.arange(len(y))
        sbl = ScatteringBaseline(y)
        A, B = sbl.solve()
        baseline = A*x+B

    # to avoid inappropriately resorting to integral in the wide angle range
    escape_ratio = p_final/100 + 0.1

    start_y = np.average(y[end_slices[0]])
    final_y = np.average(y[end_slices[1]])

    if debug:
        print("len(y)=", len(y), "end_slices=", end_slices)
        # print("start_y=", start_y, y[end_slices[0]])
        # print("final_y=", final_y, y[end_slices[1]])

    min_e = None
    min_baseline = None
    for k in range(5):
        yb  = y - baseline
        y_ = np.max([yb, zeros], axis=0)
        if raise_ and k > 0:
            neg_ratio = len(np.where(yb < 0)[0])/len(y)
            # print([k], 'neg_ratio=', neg_ratio)
            if neg_ratio < NEG_RATIO_LIMIT or neg_ratio < escape_ratio:
                pass
            else:
                if False:
                    plt.push()
                    fig, ax = plt.subplots()
                    ax.plot(y)
                    ax.plot(baseline)
                    fig.tight_layout()
                    plt.show()
                    plt.pop()
                assert False

        cy = np.cumsum(y_)
        height = cy[-1] - cy[0]
        base = start_y
        ratio = (final_y - start_y)/height
        new_baseline = base + cy*ratio
        e = np.max(np.abs(new_baseline - baseline))/abs(height)
        # print([k], 'e=', e)
        if e < epsilon:
            break

        if min_e is None or e < min_e:
            min_e = e
            min_baseline = new_baseline
        else:
            new_baseline = min_baseline
            break

        baseline = new_baseline

    if return_ratio:
        return new_baseline, ratio
    else:
        return new_baseline

def better_integrative_curve(y, baseline=None, p_final=LPM_PERCENTILE, end_slices=None):
    try:
        b = integrative_curve(y, baseline=baseline, p_final=p_final, raise_=True, end_slices=end_slices)
        convex = False
    except AssertionError:
        b = integrative_curve_convex(y)
        convex = True
    return b, convex

def compute_baseline(y, x=None, integral=False, end_slices=None, return_params=False, debug=False):
    if x is None:
        import logging
        from molass_legacy.KekLib.DebugUtils import show_call_stack
        show_call_stack("compute_baseline")
        logging.getLogger(__name__).warning("*************** compute_baseline without x is deprecated.")
        x = np.arange(len(y))

    sbl = ScatteringBaseline(y, x=x)
    A, B = sbl.solve(debug=debug)
    baseline = A*x+B

    if integral:
        baseline_, ratio = integrative_curve(y, baseline, end_slices=end_slices, return_ratio=True)
        if ratio > 0:
            baseline = baseline_
        else:
            # temporary fix due to a possibly buggy result
            ratio = 0
    else:
        ratio = 0

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        print("integral=", integral, "ratio=", ratio)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("Baseline.compute_baseline")
            ax.plot(x, y)
            ax.plot(x, A*x + B, color="red")
            ax.plot(x, baseline, ":")
            fig.tight_layout()
            plt.show()

    if return_params:
        if integral:
            return baseline, (A, B, ratio)
        else:
            return baseline, (A, B)
    else:
        return baseline
