"""
    Sigmoid.py

    Copyright (c) 2021-2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize
from bisect import bisect_right

FC_CERTAINTY_SCALE = 7      # np.exp(-0.1 * 7) == 0.5

def sigmoid(x, L ,x0, k, b):
    return L/(1 + np.exp(-k*(x-x0))) + b

def sigmoid_inv(y, L ,x0, k, b):
    """
        y = L/(1 + np.exp(-k*(x-x0))) + b
        y - b = L/
        (1 + np.exp(-k*(x-x0))) = L/(y - b)
        np.exp(-k*(x-x0)) = L/(y - b) - 1
        -k*(x-x0) = np.log(L/(y - b) - 1)
        x = -np.log(L/(y - b) - 1)/k + x0
    """
    return -np.log(L/(y - b) - 1)/k + x0

def bent_sigmoid(x, L , x0, k, b, s1, s2):
    y = sigmoid(x, L, x0, k, b)
    x_ = x - x0
    return y + np.hstack([s1*x_[x_ < 0], s2*x_[x_ >= 0]])

def bent_sigmoid_inv(y, L ,x0, k, b, s1, s2, debug=False):
    assert np.isscalar(y)

    nx = np.arange(x0-10, x0+11)
    ny = bent_sigmoid(nx, L, x0, k, b, s1, s2)
    x_ = nx - x0
    y_ = ny - np.hstack([s1*x_[x_ < 0], s2*x_[x_ >= 0]])
    if L > 0:
        i = bisect_right(y_, y)
    else:
        i = bisect_right(-y_, -y)
    ret_x = x0 + x_[i]

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("bent_sigmoid_inv debug")
            ax.plot(nx, ny)
            ax.plot(ret_x, y, "o")
            fig.tight_layout()
            plt.show()

    return ret_x

def guess_x0(x, y, debug=False):
    if debug:
        from importlib import reload
        import Trimming.SigmoidGuessX0 as target_module
        import molass_legacy.KekLib.DebugPlot as plt

        def test_guess_x0_impl():
            reload(target_module)
            from molass_legacy.Trimming.SigmoidGuessX0 import guess_x0_impl
            guess_x0_impl(x, y)

        with plt.Dp(extra_button_specs=[("test_guess_x0_impl", test_guess_x0_impl)]):
            fig, ax = plt.subplots()
            ax.set_title("guess_x0 debug")
            ax.plot(x, y)
            fig.tight_layout()
            plt.show()

        reload(target_module)

    from molass_legacy.Trimming.SigmoidGuessX0 import guess_x0_impl
    i = guess_x0_impl(x, y, debug)
    return i

def guess_bent_sigmoid(x, y, x0=None, return_certainty=False, debug=False, save_fig=False):
    if x0 is None:
        x0 = guess_x0(x, y, debug=debug)

    nx = x - x0
    params = np.zeros((2,2))
    for k, part in enumerate([nx < 0, nx >= 0]):
        x_ = nx[part]
        y_ = y[part]
        slope, intercept = linregress(x_, y_)[0:2]
        if np.isnan(slope):
            # empty x_, y_
            params[k,:] = 0, y[0]
        else:
            params[k,:] = slope, intercept

    s1, s2 = params[:,0]
    b = params[0,1]
    L = params[1,1] - params[0,1]
    k = 1

    if return_certainty or debug or save_fig:
        sigm_y = bent_sigmoid(x, L, x0, k, b, s1, s2)
        rmsd = np.sqrt(np.mean((sigm_y - y)**2))
        height = np.max(y) - np.min(y)
        fc_certainty = np.exp(-rmsd/height*FC_CERTAINTY_SCALE)

    if debug or save_fig:
        import molass_legacy.KekLib.DebugPlot as plt
        from molass_legacy._MOLASS.SerialSettings import get_setting
        from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
        from time import sleep

        std = np.std(y)

        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("guess_bent_sigmoid debug for %s" % get_in_folder())
            ax.plot(x, y)
            ax.plot(x, sigm_y)
            tx = np.mean(ax.get_xlim())
            ymin, ymax = ax.get_ylim()
            ty = np.mean([ymin, ymax])
            dy = (ymax - ymin)*0.1
            ax.text(tx, ty+dy, "RMSD=%.3g" % (rmsd), ha="center", va="center", fontsize=20, alpha=0.5)
            ax.text(tx, ty, "RMSD/STD=%.3g" % (rmsd/std), ha="center", va="center", fontsize=20, alpha=0.5)
            ax.text(tx, ty-dy, "RMSD/HEIGHT=%.3g" % (rmsd/height), ha="center", va="center", fontsize=20, alpha=0.5)
            ax.text(tx, ty-2*dy, "CERTAINTY=%.3g" % (fc_certainty), ha="center", va="center", fontsize=20, alpha=0.5)
            fig.tight_layout()
            if debug:
                plt.show()
            else:
                assert save_fig
                # see https://stackoverflow.com/questions/14694408/runtimeerror-main-thread-is-not-in-main-loop
                # see al so DebugPlot.show()
                plt.switch_backend('agg')
                plt.show(block=False)
                debug_path = get_setting("debug_path")
                fig.savefig(debug_path)
                sleep(1)

    if return_certainty:
        return (L, x0, k, b, s1, s2), fc_certainty
    else:
        return L, x0, k, b, s1, s2

def fit_bent_sigmoid(x_, y_, x0):
    params0 = guess_bent_sigmoid(x_, y_, x0)
    popt, pcov = curve_fit(bent_sigmoid, x_, y_, params0)
    return popt, pcov

def ex_sigmoid(x, L ,x0, k, b, s1, s2):
    if s1 == 0 and s2 == 0:
        return sigmoid(x, L ,x0, k, b)
    else:
        return bent_sigmoid(x, L ,x0, k, b, s1, s2)

def ex_sigmoid_inv(y, L ,x0, k, b, s1, s2):
    if s1 == 0 and s2 == 0:
        return sigmoid_inv(y, L ,x0, k, b)
    else:
        return bent_sigmoid_inv(y, L ,x0, k, b, s1, s2)

def adjust_ex_sigmoid(x, y, params, debug=False):
    # introduced to cope with anomaly in 20200123_1 UV baseline

    L ,x0, k, b, s1, s2 = params[0:6]

    # optimize only b, s1, s2
    def objective(p, title=None):
        sy = ex_sigmoid(x, L ,x0, k, *p)
        if title is not None:
            import molass_legacy.KekLib.DebugPlot as plt
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title(title)
                ax.plot(x, y)
                ax.plot(x, sy)
                fig.tight_layout()
                plt.show()
        return np.sum((sy - y)**2)

    init_params = (b, s1, s2)
    if debug:
        objective(init_params, "adjust_ex_sigmoid: before minimize")
    ret = minimize(objective, init_params)
    if debug:
        objective(ret.x, "adjust_ex_sigmoid: after minimize")
    ret_params = params.copy()
    ret_params[3:6] = ret.x
    return ret_params
