# coding: utf-8
"""
    ProbData.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from sklearn.mixture import GaussianMixture
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.SciPyCookbook import smooth
from .GaussianMixture import get_sorted_params, get_curves

N_CLUSTERS_FOR_PRE_RECOG = 10
MAX_ITER_FOR_PRE_RECOG = 50     # 20 is too small, i.e., unstable

class ProbData:
    def __init__(self, y, random_state=None):
        self.y = y
        self.sy = y_ = smooth(y)
        self.yp = yp = y_.copy()
        yp[y_ < 0] = 0
        self.M0 = np.sum(self.yp)
        self.bins = N = len(y)
        self.x = np.arange(N)
        self.M1 = np.sum(self.x*yp)/self.M0
        self.M2 = np.sum(self.yp*(self.x - self.M1)**2)/self.M0
        self.s = np.sqrt(self.M2)
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
            from .ProbDataUtils import generate_sample_data
            data = generate_sample_data(self.yp, 2)
            m_ = np.mean(data)
            ax_.plot([m_, m_], [ymin, ymax], ':', color='green')

            s_ = np.std(data)
            for x in [m-s_, m+s_]:
                ax_.plot([x, x], [ymin, ymax], ':', color='orange')

        if ax is None:
            fig.tight_layout()
            plt.show()
            plt.pop()

    def get_approx_peak_range(self, nsigma=10):
        m = self.M1
        s = self.s
        return max(0, int(m - s*nsigma)), min(self.bins, int(m + s*nsigma))
