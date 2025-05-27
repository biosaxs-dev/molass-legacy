# coding: utf-8
"""
    MXD.FastDecomposer.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Peaks.RobustPeaks import RobustPeaks
from molass_legacy.Peaks.ElutionModels import egh, compute_egh_params
from Prob.GaussianMixture import hist_to_source
from Prob.EghMixture import EghMixture
from Prob.EghMixtureUtils import get_components

def compute_egh_params_with_h(y, init_params, moments):
    h = np.max(y)
    tR, sigma, tau = compute_egh_params(init_params[1:], moments)
    return h, tR, sigma, tau

def compute_moments(x, y):
    W = np.sum(y)
    M1 = np.sum(x*y)/W              # row moment
    M2 = np.sum(y*(x-M1)**2)/W      # central moment
    M3 = np.sum(y*(x-M1)**3)/W      # central moment
    return M1, M2, M3

def verify_moments_estimation():
    """
        slightly not accurate for tau < 0
    """
    for params in [
            (1, 200, 50, 10),
            (1, 200, 50, -10)
            ]:

        h, tR, sigma, tau = params
        x = np.arange(500)
        y = egh(x, *params)

        M = compute_moments(x, y)
        init_params = h+0.1, tR-1, sigma+1, tau*1.1
        new_params = compute_egh_params_with_h(y, init_params, M)
        y_ = egh(x, *new_params)

        for p in [params, init_params, new_params]:
            print(p)

        plt.push()
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(x, y_)
        fig.tight_layout()
        plt.show()
        plt.pop()

def get_safe_slice(x, params):
    tR, sigma, tau = params[1:]
    n = 3
    start = bisect_right(x, tR - n*sigma + min(0, tau))
    stop = bisect_right(x, tR + n*sigma + max(0, tau))
    return slice(start, stop)

VERY_SMALL_VALUE = 1e-5

def str_params_list(params_list):
    ret = '[\n'
    for params in params_list:
        ret += '  [' + ','.join(["%.4g" % p for p in params] ) + '],\n'
    ret += ']\n'
    return ret

class FastDecomposer:
    def __init__(self, num_components=None):
        self.num_components = num_components

    def get_initial_params(self, x, y):
        rp = RobustPeaks(x, y, debug=False)
        params_list = []
        h_list = []
        peaks = rp.get_peaks()
        for k, peak in enumerate(peaks):
            print([k], peak)
            ls, pt, rs = peak
            h = y[pt]
            h_list.append(h)
            tR = x[pt]
            sigma = (x[pt] - x[ls] + x[rs] - x[pt])/2
            tau = 0
            params_list.append((h, tR, sigma, tau))
        if self.num_components is None:
            pass
        else:
            num_diff = len(params_list) - self.num_components
            if num_diff == 0:
                pass
            elif num_diff < 0:
                new_params_list = []
                assert len(peaks) == 1
                for peak, h, params in zip(peaks, h_list, params_list):
                    ls, pt, rs = peak
                    lsw = x[pt] - x[ls]
                    rsw = x[rs] - x[pt]
                    if lsw > rsw:
                        side_list = [-1, 0, 1]
                    else:
                        side_list = [0, 1, -1]
                    for k in range(-num_diff+1):
                        side = side_list[k]
                        if side == 0:
                            new_params_list.append(params)
                        elif side < 0:
                            h_ = y[ls]
                            tR_ = x[ls] - lsw/4
                            sigma_ = lsw/2
                            tau_ = 0
                            new_params_list.append((h_, tR_, sigma_, tau_))
                        else:
                            h_ = y[rs]
                            tR_ = x[rs] + rsw/4
                            sigma_ = rsw/2
                            tau_ = 0
                            new_params_list.append((h_, tR_, sigma_, tau_))

                params_list = sorted(new_params_list, key=lambda x: x[1])
                h_list = [p[0] for p in params_list]

        return rp.max_y, h_list, params_list

    def fit(self, x, y):
        max_y, h_list, params_list = self.get_initial_params(x, y)

        primary = np.argmax(h_list)
        print("init_params_list=", str_params_list(params_list))

        my2 = max_y**2
        last_r2 = None
        last_params_list = None
        for i in range(100):
            sigma_boundary = params_list[primary][2]
            cy_list = []
            ty = np.zeros(len(x))
            for params in params_list:
                y_ = egh(x, *params)
                cy_list.append(y_)
                ty += y_

            r2 = np.average((ty - y)**2)/my2
            if last_r2 is not None:
                if r2 > last_r2:
                    params_list = last_params_list
                    break

            last_r2 = r2
            # temp fix
            ty[np.abs(ty)==0] = VERY_SMALL_VALUE

            weights = []
            new_params_list = []
            for h_init, cy, params in zip(h_list, cy_list, params_list):
                h = params[0]
                w = cy/ty
                weights.append(w)
                wy = w*y
                if i < 50:
                    slice_ = get_safe_slice(x, params)
                    M_ = compute_moments(x[slice_], wy[slice_])
                else:
                    M_ = compute_moments(x, wy)
                params = list(compute_egh_params_with_h(wy, params, M_))
                params[2] = min(sigma_boundary, params[2])
                new_params_list.append(params)

            last_params_list = params_list
            params_list = new_params_list

            if False and i%10 == 0:
                print([i], 'r2=', r2)
                print([i], 'params_list=', str_params_list(params_list))
                print([i], 'new_params_list=', str_params_list(new_params_list))

                plt.push()
                fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(21,7))
                ax1.set_title("step %d" % i)
                ax1.plot(x, y)
                for y_ in cy_list:
                    ax1.plot(x, y_, ':')
                ax1.plot(x, ty, ':', color='red')
                ax2.plot(x, y)

                for w in weights:
                    y_ = w*y
                    ax2.plot(x, y_, ':')

                ax3.plot(x, y)
                ty_ = np.zeros(len(x))
                for params in new_params_list:
                    y_ = egh(x, *params)
                    ax3.plot(x, y_, ':')
                    ty_ += y_

                ax3.plot(x, ty_, ':', color='red')

                fig.tight_layout()
                plt.show()
                plt.pop()


        self.x = x
        self.y = y
        self.params_list = params_list

    def get_components(self):
        x = self.x
        ty = np.zeros(len(x))
        cy_list = []
        for params in self.params_list:
            cy = egh(x, *params)
            cy_list.append(cy)
            ty += cy
        return cy_list, ty
