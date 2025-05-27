# coding: utf-8
"""
    GmmUtils.py.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from lmfit import minimize, Parameters
import molass_legacy.KekLib.DebugPlot as dplt
from Prob.ProbDataUtils import generate_sample_data, plot_hist_data

def get_gmm(size, data, k, max_iter=100):
    print('get_gmm: max_iter=', max_iter)
    gmm = GaussianMixture(k, max_iter=max_iter, covariance_type='spherical')
    gmm.fit(X=np.expand_dims(data,1))
    return gmm

def get_gmm__(size, data, k):
    from .SingleGmm import SingleGmm
    gmm = SingleGmm(k, max_iter=1000)
    gmm.fit(X=np.expand_dims(data,1))
    for k_ in range(k):
        print('gamma[0:5,%d]=' % k_, gmm.gamma[0:5,k_])
    return gmm

def get_gmm___(size, data, k):
    from .ZhiyaZuoGmm import GmmAdaptor
    gmm = GmmAdaptor(k, max_iter=20)
    gmm.fit(X=np.expand_dims(data,1))
    return gmm

def get_gmm____(size, data, k):
    from .MatsukenGmm import GMM
    gmm = GMM(k, max_iter=20)
    gmm.fit(X=np.expand_dims(data,1))
    return gmm

def gaussian(x, mu, sigma, scale):
    return scale * np.exp( - (x - mu)**2 / (2 * sigma**2) )

def get_gaussians(x, y, gmm):
    gy_list = []
    ty = np.zeros(len(x))
    for k in range(len(gmm.weights_)):
        w = gmm.weights_[k]
        m = gmm.means_[k]
        v = gmm.covariances_[k]
        print([k], w, m, v)
        gy = gaussian(x, m, np.sqrt(v), w)
        gy_list.append([gy, m])
        ty += gy

    sorted_gy_list = sorted(gy_list, key=lambda x:x[1])

    def obj_func(params):
        S   = params['S']
        return ty*S - y

    params = Parameters()
    S_init = np.max(y)/np.max(ty)
    params.add('S', value=S_init, min=0, max=S_init*100 )
    result = minimize(obj_func, params, args=())

    scale = result.params['S'].value
    return [ scale*rec[0] for rec in sorted_gy_list ], ty*scale

def plot_components(ax, x, cgys):
    for k, gy in enumerate(cgys):
        ax.plot(x, gy, ':', label='component-%d' % k)

def plot_gmm_elution(y1, y2, data1, data2, k, max_iter=100, plot_class=plt):
    from DataUtils import get_in_folder

    gmm1 = get_gmm(len(y1), data1, k, max_iter=max_iter)
    gmm2 = get_gmm(len(y2), data2, k, max_iter=max_iter)

    fig, axes = plot_class.subplots(nrows=1, ncols=2, figsize=(14,7))
    ax1, ax2 = axes

    fig.suptitle("Decomposition of %s using Gaussian Mixture Model" % get_in_folder(), fontsize=20)
    ax1.set_title("UV Elution Decomposition", fontsize=16)
    ax2.set_title("X-ray Elution Decomposition", fontsize=16)
    ax1.plot(y1, label='input data')
    ax2.plot(y2, label='data')

    x1 = np.arange(len(y1))
    cgys1, gy1 = get_gaussians(x1, y1, gmm1)
    ax1.plot(gy1, label='gmm-fit')
    plot_components(ax1, x1, cgys1)

    x2 = np.arange(len(y2))
    cgys2, gy2 = get_gaussians(x2, y2, gmm2)
    ax2.plot(gy2, label='gmm-fit')
    plot_components(ax2, x2, cgys2)

    ax1.legend()
    ax2.legend()

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

def spike_demo(in_folder, lpm_correct=False, n_components=2, max_iter=100):
    from RawData import RawXray, RawUv
    ru = RawUv(in_folder)
    i = ru.get_row_index(280)
    y1 = ru.data[i,:]
    if lpm_correct:
        from LPM import get_corrected
        y1 = get_corrected(y1)

    rx = RawXray(in_folder)
    i = rx.get_row_index(0.02)
    y2 = rx.data[i,:]
    if lpm_correct:
        from LPM import get_corrected
        y2 = get_corrected(y2)

    data1 = generate_sample_data(y1, 2)
    data2 = generate_sample_data(y2, 2)

    dplt.push()
    plot_hist_data(y1, y2, data1, data2, plot_class=dplt)
    dplt.show()
    dplt.pop()

    dplt.push()
    plot_gmm_elution(y1, y2, data1, data2, n_components, max_iter=max_iter, plot_class=dplt)
    dplt.show()
    dplt.pop()
