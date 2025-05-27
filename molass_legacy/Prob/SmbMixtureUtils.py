# coding: utf-8
"""
    SmbMixtureUtils.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from lmfit import minimize, Parameters
import molass_legacy.KekLib.DebugPlot as dplt
from .ProbDataUtils import generate_sample_data, plot_hist_data
from .SmbMixture import smb_pdf

def get_gmm(size, data, k, max_iter=100):
    from .SmbMixture import SmbMixture
    gmm = SmbMixture(k, max_iter=max_iter)
    gmm.fit(X=np.expand_dims(data,1), bins=size)
    return gmm

def set_consistent_base(ax1, axt):
    ymin1, ymax1 = ax1.get_ylim()
    ymint, ymaxt = axt.get_ylim()
    y_ = ymin1/ymax1*ymaxt
    axt.set_ylim(y_, ymaxt)

def gaussian(x, mu, sigma, scale):
    return scale * np.exp( - (x - mu)**2 / (2 * sigma**2) )

def get_components(x, y, gmm):
    gy_list = []
    ty = np.zeros(len(x))
    for k in range(len(gmm.pi)):
        w = gmm.pi[k]
        m = gmm.mu[k]
        s = gmm.sigma[k]
        print([k], w, m, s)
        gy = smb_pdf(x, m, s, w)
        gy_list.append([gy, m])
        ty += gy

    sorted_gy_list = sorted(gy_list, key=lambda x:x[1])

    if True:
        dplt.push()
        fig = dplt.figure()
        ax = fig.gca()
        ax.plot(ty)
        for k, rec in enumerate(sorted_gy_list):
            gy = rec[0]
            ax.plot(gy, ':', label=str(k))

        ax.legend()
        fig.tight_layout()
        dplt.show
        dplt.pop()

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

    fig.suptitle("Decomposition of %s using EMG Mixture Model" % get_in_folder(), fontsize=20)
    ax1.set_title("UV Elution Decomposition", fontsize=16)
    ax2.set_title("X-ray Elution Decomposition", fontsize=16)
    ax1.plot(y1, label='input data')
    ax2.plot(y2, label='input data')

    x1 = np.arange(len(y1))
    cgys1, gy1 = get_components(x1, y1, gmm1)
    ax1.plot(gy1, label='eghmm-fit')
    plot_components(ax1, x1, cgys1)

    x2 = np.arange(len(y2))
    cgys2, gy2 = get_components(x2, y2, gmm2)
    ax2.plot(gy2, label='eghmm-fit')
    plot_components(ax2, x2, cgys2)

    ax1.legend()
    ax2.legend()

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

def spike_demo(in_folder, lpm_correct=False, lpm_2d=False, n_components=2, smoothing=False, max_iter=100):
    print("max_iter=", max_iter)

    if lpm_correct:
        from CorrectedData import  CorrectedXray,  CorrectedUv
        ru = CorrectedUv(in_folder)
        rx = CorrectedXray(in_folder)
    else:
        from RawData import RawXray, RawUv
        ru = RawUv(in_folder)
        rx = RawXray(in_folder)

    i = ru.get_row_index(280)
    y1 = ru.data[i,:]
    if lpm_2d:
        from LPM import get_corrected
        y1 = get_corrected(y1)

    i = rx.get_row_index(0.02)
    y2 = rx.data[i,:]
    if lpm_2d:
        from LPM import get_corrected
        y2 = get_corrected(y2)

    if smoothing:
        from molass_legacy.KekLib.SciPyCookbook import smooth
        y1 = smooth(y1)
        y2 = smooth(y2)

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
