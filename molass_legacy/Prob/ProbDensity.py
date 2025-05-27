# coding: utf-8
"""
    ProbDensity.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.SciPyCookbook import smooth
from .ProbDensityUtils import generate_sample_data

N_CLUSTERS_FOR_PRE_RECOG = 10
MAX_ITER_FOR_PRE_RECOG = 50     # 20 is too small, i.e., unstable

class ProbDensity:
    def __init__(self, y, random_state=None):
        self.y = y
        self.N = N = len(y)
        self.x = x = np.arange(N)
        self.sy = y_ = smooth(y)
        self.py = py = y_.copy()
        py[y_ < 0] = 0
        self.M0 = np.sum(self.py)
        self.ny = self.py/self.M0
        self.pdf = UnivariateSpline(x, self.ny, s=0, ext=3)
        self.spline = UnivariateSpline(x, self.py, s=0, ext=3)
        self.data = generate_sample_data(py, 2)
        self.M1 = np.sum(x*py)/self.M0
        self.M2 = np.sum(self.py*(x - self.M1)**2)/self.M0
        self.s = np.sqrt(self.M2)
        k = int(N*0.99)
        pp = np.argpartition(py, k)
        self.mode = np.average(pp[k:])
        print('m, s=', self.M1, self.s)

    def proof_plot(self, ax=None, show_np_params=False):

        if ax is None:
            plt.push()
            fig  = plt.figure()
            ax_ = fig.gca()
        else:
            ax_ = ax

        ax_.plot(self.y, alpha=0.5)
        ax_.plot(self.sy)

        ymin, ymax =ax_.get_ylim()
        ax_.set_ylim(ymin, ymax)

        m = self.M1
        ax_.plot([m, m], [ymin, ymax], color='green')
        s = self.s
        for x in [m-s, m+s]:
            ax_.plot([x, x], [ymin, ymax], color='orange')

        if show_np_params:
            m_ = np.mean(self.data)
            ax_.plot([m_, m_], [ymin, ymax], ':', color='green')

            s_ = np.std(self.data)
            for x in [m-s_, m+s_]:
                ax_.plot([x, x], [ymin, ymax], ':', color='orange')

        if ax is None:
            fig.tight_layout()
            plt.show()
            plt.pop()

    def get_approx_peak_range(self, nsigma=10):
        m = self.M1
        s = self.s
        return max(0, int(m - s*nsigma)), min(self.N, int(m + s*nsigma))
