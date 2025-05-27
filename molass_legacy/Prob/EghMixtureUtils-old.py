# coding: utf-8
"""
    EghMixtureUtils.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.mixture import GaussianMixture
from lmfit import minimize, Parameters
import molass_legacy.KekLib.DebugPlot as dplt
from .ProbDataUtils import generate_samle_datasets, plot_hist_data
from .EghMixture import egh_pdf

def get_gmm_(size, data, k):
    gmm = GaussianMixture(k, max_iter=1000, covariance_type='spherical')
    gmm.fit(X=np.expand_dims(data,1))
    return gmm

def get_gmm__(size, data, k):
    from Gmm.SingleGmm import SingleGmm
    gmm = SingleGmm(k, max_iter=1000)
    gmm.fit(X=np.expand_dims(data,1))
    for k_ in range(k):
        print('gamma[0:5,%d]=' % k_, gmm.gamma[0:5,k_])
    return gmm

def get_gmm___(size, data, k):
    from Gmm.ZhiyaZuoGmm import GmmAdaptor
    gmm = GmmAdaptor(k, max_iter=20)
    gmm.fit(X=np.expand_dims(data,1))
    return gmm

def get_gmm____(size, data, k):
    from Gmm.MatsukenGmm import GMM
    gmm = GMM(k, max_iter=20)
    gmm.fit(X=np.expand_dims(data,1))
    return gmm

def get_gmm(size, data, k, max_iter=100, anim_data=False):
    from .EghMixture import EghMixture
    gmm = EghMixture(k, max_iter=max_iter, anim_data=anim_data)
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
        m = gmm.tR[k]
        s = gmm.sigma[k]
        t = gmm.tau[k]
        print([k], w, m, s, t)
        gy = egh_pdf(x, m, s, t, w)
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
    ret_lines = []
    for k, gy in enumerate(cgys):
        line, = ax.plot(x, gy, ':', label='component-%d' % k)
        ret_lines.append(line)
    return ret_lines

def plot_gmm_elution(y1, y2, data1, data2, k, max_iter=100, plot_class=plt, anim_data=False):
    from DataUtils import get_in_folder

    gmm1 = get_gmm(len(y1), data1, k, max_iter=max_iter, anim_data=anim_data)
    gmm2 = get_gmm(len(y2), data2, k, max_iter=max_iter, anim_data=anim_data)

    fig, axes = plot_class.subplots(nrows=1, ncols=2, figsize=(14,7))
    ax1, ax2 = axes

    fig.suptitle("Decomposition of %s using EGH Mixture Model" % get_in_folder(), fontsize=20)
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

    return gmm1, gmm2

def get_anim_components(x, y, gmm, n):
    gy_list = []
    ty = np.zeros(len(x))
    for k in range(len(gmm.pi)):
        w = gmm.pi_array[n,k]
        m = gmm.tR_array[n,k]
        s = gmm.sigma_array[n,k]
        t = gmm.tau_array[n,k]
        # print([n,k], w, m, s, t)
        gy = egh_pdf(x, m, s, t, w)
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

def animate_gmm_elution(y1, y2, gmm1, gmm2, plot_class=plt):
    from DataUtils import get_in_folder

    # fig, axes = plot_class.subplots(nrows=1, ncols=2, figsize=(14,7))     # does not work for animation
    # ax1, ax2 = axes

    fig = plot_class.figure(figsize=(14,7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    fig.suptitle("Decomposition of %s using EGH Mixture Model Animation" % get_in_folder(), fontsize=20)
    ax1.set_title("UV Elution Decomposition", fontsize=16)
    ax2.set_title("X-ray Elution Decomposition", fontsize=16)
    ax1.plot(y1, label='input data')
    ax2.plot(y2, label='input data')

    x1 = np.arange(len(y1))
    cgys1, gy1 = get_anim_components(x1, y1, gmm1, 0)
    line1, = ax1.plot(gy1, label='eghmm-fit')
    lines1 = plot_components(ax1, x1, cgys1)

    x2 = np.arange(len(y2))
    cgys2, gy2 = get_anim_components(x2, y2, gmm2, 0)
    line2, = ax2.plot(gy2, label='eghmm-fit')
    lines2 = plot_components(ax2, x2, cgys2)

    def add_number_text(ax):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        tx = (xmin + xmax)/2
        ty = (ymin + ymax)/2
        text = ax.text(tx, ty, "_", ha='center', va='center', alpha=0.05, fontsize=240)
        return text

    text1 = add_number_text(ax1)
    text2 = add_number_text(ax2)

    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    anim_artists = [line1] + lines1 + [line2] + lines2 + [text1, text2]

    def animate(i):
        # print('animate', i)

        cgys1, gy1 = get_anim_components(x1, y1, gmm1, i+1)
        line1.set_data(x1, gy1)
        for line, gy in zip(lines1, cgys1):
            line.set_data(x1, gy)

        cgys2, gy2 = get_anim_components(x2, y2, gmm2, i+1)
        line2.set_data(x2, gy2)
        for line, gy in zip(lines2, cgys2):
            line.set_data(x2, gy)

        text_num = str(i+1)
        text1.set_text(text_num)
        text2.set_text(text_num)

        dplt.update()   # need a chance to get events

        return anim_artists

    def reset():
        print('reset')
        return animate(-1)

    num_frames = gmm1.max_iter
    print('num_frames=', num_frames)
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, blit=True, init_func=reset, interval=100)

    return anim

def spike_demo(in_folder, **kwargs):
    n_components = kwargs.pop('n_components', 2)
    max_iter = kwargs.pop('max_iter', 100)
    anim_data = kwargs.pop('anim_data', False)

    print("max_iter=", max_iter)

    y1, y2, data1, data2 = generate_samle_datasets(in_folder, **kwargs)

    dplt.push()
    plot_hist_data(y1, y2, data1, data2, plot_class=dplt)
    dplt.show()
    dplt.pop()

    dplt.push()
    gmm1, gmm2 = plot_gmm_elution(y1, y2, data1, data2, n_components, max_iter=max_iter, plot_class=dplt, anim_data=anim_data)
    dplt.show()
    dplt.pop()

    if anim_data:
        dplt.push()
        anim = animate_gmm_elution(y1, y2, gmm1, gmm2, plot_class=dplt)
        print('anim=', anim)
        dplt.show()
        dplt.pop()
