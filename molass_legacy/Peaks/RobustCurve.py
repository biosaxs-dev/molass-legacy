# coding: utf-8
"""

    RobustCurve.py

        evaluation of simularity between curves

    Copyright (c) 2021, SAXS Team, KEK-PF

"""
import numpy as np
from scipy.optimize import minimize
from molass_legacy.KekLib.SciPyCookbook import smooth
from matplotlib.gridspec import GridSpec
import molass_legacy.KekLib.DebugPlot as plt
from .RobustPeaks import RobustPeaks
from .ElutionModels import gaussian, egh

LIFTED_BASE = False
MINIMIZE_METHOD = 'Nelder-Mead'

class PeakModel:
    def get_peaktop(self):
        return self.peaktop

    def __call__(self, x=None):
        if x is None:
            x = self.px
        return x, self.base + self.func(x, *self.params)

class PeakGaussian(PeakModel):
    name = 'Gaussian'
    def __init__(self, x, y, pt, num_hw=10):
        pt_slice = slice(pt-num_hw, pt+num_hw)
        x_ = x[pt_slice]
        if LIFTED_BASE:
            b_ = np.min(y[pt_slice])
        else:
            b_ = 0
        y_ = y[pt_slice] - b_
        # y_ = smooth(sy, window_len=5)

        def obj_func(p):
            h, m, s = p
            return np.sum((gaussian(x_, h, m, s) - y_)**2)

        h_ = y_[num_hw]
        m_ = x[pt]
        s_ = (x_[-1] - x_[0])/2
        ret = minimize(obj_func, (h_, m_, s_), method=MINIMIZE_METHOD)
        h, m, s = ret.x
        self.px = x_ + m - x[pt]
        self.base = b_
        self.params = (h, m, s)
        self.peaktop = (m, b_+h)
        self.pt_slice = pt_slice
        self.func = gaussian

        if False:
            plt.push()
            fig, ax = plt.subplots()
            # ax.plot(x_, sy)
            ax.plot(x_, y_, ':')
            ax.plot(x_, gaussian(x_, h, m, s))
            plt.show()
            plt.pop()

class PeakEGH(PeakModel):
    name = 'EGH'
    def __init__(self, x, y, pt, num_hw=10):
        pt_slice = slice(pt-num_hw, pt+num_hw)
        x_ = x[pt_slice]
        if LIFTED_BASE:
            b_ = np.min(y[pt_slice])
        else:
            b_ = 0
        y_ = y[pt_slice] - b_

        self.debug = False

        def obj_func(p):
            h, m, s, t = p
            fy = egh(x_, h, m, s, t)
            if self.debug:
                plt.push()
                fig, ax = plt.subplots()
                ax.plot(x_, y_)
                ax.plot(x_, fy)
                self.debug = plt.show()
                plt.pop()
            return np.sum((fy - y_)**2)

        h_ = y_[num_hw]
        m_ = x[pt]
        s_ = (x_[-1] - x_[0])/2
        t_ = 0
        ret = minimize(obj_func, (h_, m_, s_, t_), method=MINIMIZE_METHOD)
        h, m, s, t = ret.x
        self.px = x_ + m - x[pt]
        self.base = b_
        self.params = (h, m, s, t)
        self.peaktop = (m, b_+h)
        self.pt_slice = pt_slice
        self.func = egh

class PeakHyperbola(PeakModel):
    name = 'Hyperbola'
    def __init__(self, x, y, pt, num_hw=10):
        from .Hyperbola import RotatedHyperbola, compute_rotated_hyperbola_y

        pt_slice = slice(pt-num_hw, pt+num_hw)
        x_ = x[pt_slice]
        if LIFTED_BASE:
            b_ = np.min(y[pt_slice])
        else:
            b_ = 0
        y_ = y[pt_slice] - b_

        ptx = x[pt]
        pty = y[pt] - b_
        hbw = (x_[-1] - x_[0])/2
        rh = RotatedHyperbola(ptx, pty, hbw=hbw)
        rh.fit(x_, y_, method=MINIMIZE_METHOD)

        self.base = b_
        self.params = rh.fitted_params
        self.peaktop = b_ + rh.get_peak_top()
        self.px = x_ + self.peaktop[0] - x[pt]
        self.pt_slice = pt_slice
        self.func = compute_rotated_hyperbola_y

        if False:
            plt.push()
            fig, ax = plt.subplots()
            # ax.plot(x_, sy)
            ax.plot(x_, y_, ':')
            ax.plot(x_, rh(x_))
            plt.show()
            plt.pop()

class RobustCurve:
    def __init__(self, x, y, peak_model=PeakEGH, debug=False):
        self.x = x
        self.y = y
        self.rp = RobustPeaks(x, y, debug=debug)
        self.peak_model = peak_model
        self.fit_peak_models()

    def fit_peak_models(self):
        pgs = []
        for peak in self.rp.get_peaks():
            ls, pt, rs = peak
            width = int(((pt - ls) + (rs - pt))/4)
            pgs.append(self.peak_model(self.x, self.y, pt, num_hw=width))
        self.peak_models = pgs

    def get_peak_tops(self):
        pts = []
        for pg in self.peak_models:
            pts.append(pg.get_peaktop())
        return pts

    def get_peak_models(self):
        return self.peak_models

def plot_curve(ax, curve, color=None):
    x = curve.x
    y = curve.y
    ax.plot(x, y, color=color)
    labeled = False
    for pg in curve.get_peak_models():
        g_label = None if labeled else 'locally fitted gaussian'
        ax.plot(*pg(), color='green', label=g_label)
        x_ = x[pg.pt_slice]
        y_ = pg.base
        ax.plot(x_[[0,-1]], [y_, y_], ':', color='red')
        pt_label = None if labeled else 'fitted gaussian top'
        ax.plot(*pg.get_peaktop(), 'o', color='red', label=pt_label)
        labeled = True

def demo(sd):
    xr_curve = sd.get_xray_curve()
    uv_curve = sd.get_uv_curve()

    uv_rc = RobustCurve(uv_curve.x, uv_curve.y)
    xr_rc = RobustCurve(xr_curve.x, xr_curve.y)

    plt.push()
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(21,7))
    ax1.set_title("UV Elution Curve", fontsize=16)
    ax2.set_title("Xray Elution Curve", fontsize=16)
    plot_curve(ax1, uv_rc)
    plot_curve(ax2, xr_rc)
    fig.tight_layout()
    plt.show()
    plt.pop()

