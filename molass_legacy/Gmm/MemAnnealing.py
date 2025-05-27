# coding: utf-8
"""
    MemAnnealing.py

    Mapped Elution Model using Simulated Annealing

    Copyright (c) 2020,2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from matplotlib.patches import Polygon
import molass_legacy.KekLib.DebugPlot as plt

USE_NUMBA = False
if USE_NUMBA:
    try:
        # for numba 1.49 or later
        from numba.core.decorators import jit
    except:
        from numba.decorators import jit

def gaussian(x, h, m, s):
    return h*np.exp(-((x-m)/s)**2)

class MemAnnealing:
    def __init__(self):
        pass

    def fit(self, y1, y2, K):

        x1 = np.arange(len(y1))
        x2 = np.arange(len(y2))
        spline1 = UnivariateSpline(x1, y1, s=0, ext=3)
        spline2 = UnivariateSpline(x2, y2, s=0, ext=3)

        if USE_NUMBA:

            @jit(nopython=True)
            def compute_ch2(x, y, mv, sv, spline, ax=None):
                if ax is not None:
                    ax.plot(x, y)

                y_ = np.zeros(len(x))
                chi2 = 0
                for k in range(K):
                    n = 2 + k*2
                    m = mv[k]
                    s = sv[k]
                    h = spline(m)
                    gy = gaussian(x, h, m, s)
                    y_ += gy
                    chi2 += np.sum((y_ - y)**2)
                return chi2

        else:

            def compute_ch2(x, y, mv, sv, spline, ax=None):
                if ax is not None:
                    ax.plot(x, y)

                y_ = np.zeros(len(x))
                chi2 = 0
                for k in range(K):
                    n = 2 + k*2
                    m = mv[k]
                    s = sv[k]
                    h = spline(m)
                    gy = gaussian(x, h, m, s)
                    y_ += gy
                    if ax is not None:
                        ax.plot(x, gy)
                        if k == K-1:
                            ax.plot(x, y_, ':')
                            poly_points = list(zip(x, y)) + list( reversed( list( zip(x, y_) ) ) )
                            diff_poly   = Polygon( poly_points, alpha=0.2 )
                            ax.add_patch(diff_poly)
                    chi2 += np.sum((y_ - y)**2)
                return chi2

        count = 0

        if USE_NUMBA:
            # spline does not seem to be supported.

            @jit(nopython=True)
            def obj_func(p):
                nonlocal count

                a, b = p[0:2]
                m = p[2:2+K]
                s = p[2+K:2+K+K]
                m_ = m*a + b
                s_ = s*a

                chi2 = compute_ch2(x1, y1, m_, s_, spline1) * compute_ch2(x2, y2, m, s, spline2)
                reverse_penality = 1 + max(0, (m[0] - m[1]))
                close_penality = max(1, (len(y2)/min(50, abs(m[0] - m[1])))**2)
                chi2_ = chi2 * reverse_penality*close_penality

                count += 1

                return chi2_
        else:

            def obj_func(p):
                nonlocal count

                a, b = p[0:2]
                m = p[2:2+K]
                s = p[2+K:2+K+K]
                m_ = m*a + b
                s_ = s*a

                # debug_plot = count%100 == 0
                debug_plot = False

                if debug_plot:
                    print([count], m)
                    plt.push()
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,7))
                    ax1, ax2 = axes
                    fig.suptitle(str([count]))
                else:
                    ax1, ax2 = None, None

                chi2 = compute_ch2(x1, y1, m_, s_, spline1, ax1) * compute_ch2(x2, y2, m, s, spline2, ax2)
                close_penality = max(1, (len(y2)/min(50, abs(m[0] - m[1])))**2)
                chi2_ = chi2 * close_penality

                if debug_plot:
                    print('reverse_penality=', reverse_penality)
                    print('close_penality=', close_penality)
                    print('chi2=', chi2)
                    print('chi2_=', chi2_)
                    fig.tight_layout()
                    plt.show()
                    plt.pop()

                count += 1

                return chi2_

        a_init = len(y1)/len(y2)
        b_init = 0
        m_init = [len(y2)*0.3, len(y2)*0.7]
        s_init = [50, 50]
        # range_init = [[a_init*0.3, a_init*3], [-100, 100], [180, 220], [250, 300], [10, 100], [10, 100]]
        range_init = [[a_init*0.8, a_init*1.2], [-200, 200], [180, 220], [230, 300], [10, 60], [10, 60]]
        x_range = np.array(range_init)
        p_init = np.average(x_range, axis=1)

        if True:
            from molass_legacy.KekLib.BasicUtils import Struct
            from SimulatedAnnealing import SimulatedAnnealing
            anneal = SimulatedAnnealing()
            x_min = x_range[:,0]
            x_max = x_range[:,1]
            def xconstaints(x):
                return x[2] < x[3]
            anneal.minimize(obj_func, xrange=x_range, start=p_init, seed=1234,
                        xconstaints=xconstaints, n=3000, m=1000)
            print('anneal.x=', anneal.x)
            result = Struct(x=anneal.x[-1])
        else:
            result = minimize(obj_func, p_init)
        self.spline1 = spline1
        self.spline2 = spline2
        return result
