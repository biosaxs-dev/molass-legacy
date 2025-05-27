# coding: utf-8
"""
    Rgg.GmmSpike.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import time
import asyncio
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.stats import multivariate_normal
from sklearn import mixture
import MatplotlibUtils          # required for annotate3D
from molass_legacy.Peaks.ElutionModels import egh
from Prob.GaussianMixture import gaussian_pdf
import molass_legacy.KekLib.DebugPlot as plt
from RgProcess.RgCurve import make_availability_slices
from .SpikeData import generate_demo_data, FULL_PARAMS, normal_pdf
from .RggUtils import convert_to_probabilitic_data, plot_histogram_2d

def data_demo():
    x, y, rg, num_components = generate_demo_data()
    max_y = np.max(y)
    slices, states = make_availability_slices(y, max_y=max_y, min_ratio=0.03)

    plt.push()
    fig = plt.figure(figsize=(21,6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    axt = ax1.twinx()
    axt.grid(False)

    fig.suptitle("Conversion to 2D Probabilistic Data compared to the True Distribution", fontsize=20)

    ax1.set_title("Observed Elution Curve and Rg", fontsize=16)
    ax2.set_title("Apparent Histogram of 2D Data Distribution", fontsize=16)
    ax3.set_title("True Histogram of 2D Data Distribution", fontsize=16)

    for ax in [ax2, ax3]:
        ax.set_xlabel("Eno")
        ax.set_ylabel("Rg")
        ax.set_zlabel("Counts")
        ax.set_ylim(10, 50)

    axt.set_ylim(10, 50)

    for slice_, state in zip(slices, states):
        x_ = x[slice_]
        y_ = y[slice_]
        rg_ = rg[slice_]
        linestyle = '-' if state else ':'
        ax1.plot(x_, y_, linestyle, color='C0')
        axt.plot(x_, rg_, linestyle, color='C1')
        if state:
            plot_histogram_2d(ax2, x_, y_, rg_, max_y)

    peaktop = (200, 35, 100)
    for ax in [ax2, ax3]:
        ax.annotate3D('(%.3g, %.3g, %.3g)' % peaktop, peaktop,
          xytext=(30,-30),
          textcoords='offset points',
          bbox=dict(boxstyle="round", fc="lightyellow"),
          arrowprops = dict(arrowstyle="-|>", ec='black', fc='white', lw=1))

    for rg, params in FULL_PARAMS:
        h, mu, sigma, tau = params
        y_ = h * normal_pdf(x, mu, sigma)
        rg_ = np.ones(len(x))*rg
        plot_histogram_2d(ax3, x, y_, rg_, max_y)

    ax2.set_xlim(ax3.get_xlim())

    fig.tight_layout()
    fig.subplots_adjust(top=0.88, left=0.04, wspace=0.12, right=0.96)
    plt.show()
    plt.pop()

USE_VARIATIONAL = True
USE_SCIKIT_LEARN = True

def spike_demo():
    x, y, rg, num_components = generate_demo_data()
    spike_demo_impl([(x, y, rg, None)], num_components=num_components)

def spike_demo_impl(data_list, num_components=None, num_precision=100):

    num_datasets = len(data_list)

    plt.push()
    fig = plt.figure(figsize=(21,7))
    gs = GridSpec(num_datasets, 3)

    for k, (x, y, rg, X) in enumerate(data_list):
        ax1 = fig.add_subplot(gs[k,0])
        ax2 = fig.add_subplot(gs[k,1], projection='3d')
        ax3 = fig.add_subplot(gs[k,2])

        axt = ax1.twinx()
        axt.grid(False)

        # ax.plot(x, y)
        # axt.plot(x, rg, color='C1')
        max_y = np.max(y)
        X_list = []
        slices, states = make_availability_slices(y, max_y=max_y, min_ratio=0.03)
        for slice_, state in zip(slices, states):
            x_ = x[slice_]
            y_ = y[slice_]
            rg_ = rg[slice_]
            if state:
                X_list.append(convert_to_probabilitic_data(x_, y_, rg_, max_y=max_y, num_precision=num_precision))
            linestyle = '-' if state else ':'
            ax1.plot(x_, y_, linestyle, color='C0')
            axt.plot(x_, rg_, linestyle, color='C1')
            if state:
                if True:
                    plot_histogram_2d(ax2, x_, y_, rg_, max_y, num_precision=num_precision)
                else:
                    b_ = np.zeros(len(x_))
                    ax2.bar3d(x_, rg_, b_, 0.5, 0.05, y_/max_y*100, shade=True, edgecolor='green')
                ax2.set_ylim(10, 60)

        ymin, ymax = axt.get_ylim()
        axt.set_ylim(0, ymax*1.2)

        X = np.concatenate(X_list)
        # covariance_type='spherical'

        max_iter = 100
        covariance_type='full'
        # means_init = [(200, 35), (250, 30), (400, 23)]

        if USE_VARIATIONAL:
            if USE_SCIKIT_LEARN:
                gmm = mixture.BayesianGaussianMixture(n_components=num_components, covariance_type=covariance_type, max_iter=max_iter)
                gmm.fit(X)
            else:
                assert False
        else:
            gmm = mixture.GaussianMixture(n_components=num_components, covariance_type=covariance_type, max_iter=max_iter)
            gmm.fit(X)
        print(dir(gmm))
        print("weights_=", gmm.weights_)
        print("means_=", gmm.means_)
        print("covariances_=", gmm.covariances_)
 
        cy_list = []
        ty = np.zeros(len(x))
        for w, m, cv in zip(gmm.weights_, gmm.means_, gmm.covariances_):
            var = cv if covariance_type == 'spherical' else cv[0,0]
            cy = w * gaussian_pdf(x, m[0], np.sqrt(var))
            cy_list.append(cy)
            ty += cy

        scale = max_y/np.max(ty)

        ax3.plot(x, y)
        for cy in cy_list:
            ax3.plot(x, scale*cy, ':')
        ax3.plot(x, scale*ty, ':', color='red')

    fig.tight_layout()
    cont = plt.show()
    plt.pop()
    return gmm

def spike_demo_real(in_folder, num_components=3, correction=True, use_peaks=False):
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from RgProcess.RgCurve import RgCurve

    sp = StandardProcedure()
    sp.load(in_folder, debug=False)
    if correction:
        sd = sp.get_corrected_sd()
    else:
        sd = sp.get_sd()
    D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
    rgc = RgCurve(qv, xr_curve, D, E)
    rgc.proof_plot()
    X = rgc.get_probabilistic_data()
    spike_demo_impl([[xr_curve.x, xr_curve.y, None, X]], num_components=3, use_peaks=use_peaks)

async def get_content(n):
    await asyncio.sleep(3)
    return n + 1

async def f(n):
    content = await get_content(n)
    return content

def coroutine_spike():
    loop = asyncio.get_event_loop()
    v = loop.run_until_complete(f(1))
    print(v)
