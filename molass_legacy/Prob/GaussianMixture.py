# coding: utf-8
"""
    GaussianMixture.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np

# gaussian pdf function
SQRT_2PI = np.sqrt(2*np.pi)
def gaussian_pdf(x, m, s, scale=1):
    return scale * 1/(s*SQRT_2PI) * np.exp(-0.5*((x-m)/s)**2)

# mixture curve data
PARAMS = [(100, 30, 1), (200, 20, 0.5)]

def gm_curve(x_length = 300, params=PARAMS, ret_detail=False):
    x = np.arange(x_length)
    gy = [gaussian_pdf(x,*p) for p in params]
    y = np.sum(gy, axis=0)
    if ret_detail:
        return x, y, gy
    else:
        return x, y

def hist_to_source(x, y, num_presision = 100):
    # conversion from curve (histogram) to source data
    data = []
    max_y = np.max(y)
    for x_, y_ in zip(x, y):
        data += [x_]*int(y_/max_y*num_presision)

    sy = np.array(data)
    return sy

def get_sorted_params(gmm):
    # sort results in the order of the means
    params = zip(gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_)
    sorted_params = sorted([ (m, v, w) for m, v, w in params], key=lambda p:p[0])
    return sorted_params

def get_curves(gmm, x):
    sorted_params = get_sorted_params(gmm)

    # plot the results
    gy_list = []
    ty = np.zeros(len(x))
    for m, v, w in sorted_params:
        gy = gaussian_pdf(x, m, np.sqrt(v), w)
        gy_list.append(gy)
        ty += gy

    return ty, gy_list

def debug_plot_gmm(x, y, sy, gmm, gmm2=None, num_presision = 100):
    import molass_legacy.KekLib.DebugPlot as plt
    max_y = np.max(y)

    if gmm2 is None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
        ax1, ax3 = axes
        ax2 = None
        pmm_plot_params = [(ax3, gmm)]
    else:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,5))
        ax1, ax2, ax3 = axes
        pmm_plot_params = [(ax2, gmm2), (ax3, gmm)]

    ax1.set_title("Source Data Generation")
    if ax2 is not None:
        ax3.set_title("Gaussian Mixture Model Fitting (intermediate)")
    ax3.set_title("Gaussian Mixture Model Fitting")

    ax1t = ax1.twinx()
    ax1.plot(y, linewidth=3, label='given curve')
    ax1t.hist(sy, bins=len(x), color='orange', label='histogram of generated data')

    fig_scale = 1.05
    ax1.set_ylim(0, max_y*fig_scale)
    ax1t.set_ylim(0, num_presision*fig_scale)

    ax1.legend(bbox_to_anchor=(1, 0.98), loc='upper right')
    ax1t.legend(bbox_to_anchor=(1, 0.92), loc='upper right')

    for ax, gmm_ in pmm_plot_params:
        ty, gy_list = get_curves(gmm_, x)
        ax.plot(y, linewidth=3, label='given curve')
        axt = ax.twinx()
        axt.plot(ty, ':', color='red', linewidth=3, label='GMM fitted curve')
        for k, gy in enumerate(gy_list):
            axt.plot(gy, ':', label='%d-th component' % k)

        ax.legend(bbox_to_anchor=(1, 0.98), loc='upper right')
        axt.legend(bbox_to_anchor=(1, 0.92), loc='upper right')

    fig.tight_layout()
    plt.show()
