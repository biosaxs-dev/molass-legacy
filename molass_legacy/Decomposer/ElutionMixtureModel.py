# coding: utf-8
"""
    ElutionMixtureModel.py

    learned at
    Gaussian Mixture Models of an Image's Histogram
    https://stackoverflow.com/questions/45805316/gaussian-mixture-models-of-an-images-histogram

    Copyright (c) 2019-2020, SAXS Team, KEK-PF
"""
import numpy as np
from sklearn.mixture import GaussianMixture
try:
    # for numba 1.49 or later
    from numba.core.decorators import jit
except:
    from numba.decorators import jit
from lmfit import minimize, Parameters
import molass_legacy.KekLib.DebugPlot as plt

@jit(nopython=True)
def generate_samples(x, y, n):
    data = []
    for k, i in enumerate(x):
        for j in range(int(y[k]*n)):
            data.append(i)

    return np.array(data)

def gaussian(x, mu, sigma, scale):
    return scale * np.exp( - (x - mu)**2 / (2 * sigma**2) )

class ElutionMixtureModel:
    def __init__(self, n_components=3):
        self.gmm = GaussianMixture(n_components, max_iter=1000, covariance_type='spherical' )

    def fit_elution(self, curve, slice_=None, precision=3):
        self.curve = curve
        x_ = curve.x
        y_ = curve.y / curve.max_y

        if slice_ is None:
            x = x_
            y = y_
        else:
            x = x_[slice_]
            y = y_[slice_]

        n = 10**precision
        self.data = generate_samples(x, y, n)
        self.gmm.fit(X=np.expand_dims(self.data,1))

        # Evaluate GMM
        # gmm_x = np.linspace(0,253,256)
        self.gmm_x = x
        self.gmm_y = np.exp(self.gmm.score_samples(self.gmm_x.reshape(-1,1)))
        self.y_ = y_

    def get_gaussians(self):
        x = self.gmm_x
        y = self.gmm_y

        gy_list = []
        ty = np.zeros(len(x))
        for k in range(len(self.gmm.weights_)):
            w = self.gmm.weights_[k]
            m = self.gmm.means_[k][0]
            v = self.gmm.covariances_[k]
            print([k], w, m, v)
            gy = gaussian(x, m, np.sqrt(v), w)
            gy_list.append(gy)
            ty += gy

        def obj_func(params):
            S   = params['S']
            return ty*S - y

        params = Parameters()
        S_init = np.max(y)/np.max(ty)
        params.add('S', value=S_init, min=0, max=S_init*100 )
        result = minimize(obj_func, params, args=())

        scale = result.params['S'].value
        return [ scale*gy for gy in gy_list ]

    def plot(self, ax):
        x = self.gmm_x
        y = self.gmm_y
        # Plot histograms and gaussian curves
        r = [x[0], x[-1]+1]
        ny_ = self.y_ / np.sum(self.y_)
        ax.plot(x, ny_, color="orange", label="data")
        ax.hist(self.data, x[-1], r, density=True, alpha=0.5)
        ax.plot(x, y, ':', color="red", lw=3, label="GMM")
        ax.set_ylabel("Intensity")
        ax.set_xlabel("Elution No")

        for k, gy in enumerate(self.get_gaussians()):
            ax.plot(x, gy, label='%d' % k)

        ax.legend()

def gmm_demo(curve):
    num_comp_matrix = np.array([[1, 2, 3], [4, 5, 6]])
    nrows, ncols = num_comp_matrix.shape

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 10))

    for i in range(nrows):
        for j in range(ncols):
            n = num_comp_matrix[i, j]
            ax = axes[i, j]
            ax.set_title("GMM with n_components=%d" % n)
            gmm = ElutionMixtureModel(n_components=n)
            # slice_=slice(150, 400)
            slice_=None
            gmm.fit_elution(curve, slice_=slice_, precision=3)
            gmm.plot(ax)

    fig.tight_layout()
    plt.show()
