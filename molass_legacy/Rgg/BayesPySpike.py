# coding: utf-8
"""
    Rgg.BayesPySpike.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import time
import asyncio
import numpy as np
from matplotlib.gridspec import GridSpec
import molass_legacy.KekLib.DebugPlot as plt
from bayespy.nodes import Dirichlet, Categorical
from bayespy.nodes import Gaussian, Wishart
from bayespy.nodes import Mixture
from bayespy.inference import VB
import bayespy.plot as bpplt
from RgProcess.RgCurve import make_availability_slices
from .SpikeData import generate_demo_data
from .RggUtils import convert_to_probabilitic_data, plot_histogram_2d

BAYESPY_PLOT = False

def spike_demo():
    x, y, rg, num_components = generate_demo_data()
    spike_demo_impl([(x, y, rg, None)], num_components=num_components)

def spike_demo_impl(data_list, num_components=None):

    num_datasets = len(data_list)

    if BAYESPY_PLOT:
        pass
    else:
        plt.push()
        fig = plt.figure(figsize=(21,7))
        gs = GridSpec(num_datasets, 3)

    for k, (x, y, rg, X) in enumerate(data_list):

        if BAYESPY_PLOT:
            pass
        else:
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
                X_list.append(convert_to_probabilitic_data(x_, y_, rg_, max_y=max_y))

            if BAYESPY_PLOT:
                pass
            else:
                linestyle = '-' if state else ':'
                ax1.plot(x_, y_, linestyle, color='C0')
                axt.plot(x_, rg_, linestyle, color='C1')
                if state:
                    if True:
                        plot_histogram_2d(ax2, x_, y_, rg_, max_y)
                    else:
                        b_ = np.zeros(len(x_))
                        ax2.bar3d(x_, rg_, b_, 0.5, 0.05, y_/max_y*100, shade=True, edgecolor='green')
                    ax2.set_ylim(10, 60)

                ymin, ymax = axt.get_ylim()
                axt.set_ylim(0, ymax*1.2)

        X = np.concatenate(X_list)
        print("X.shape=", X.shape)

        N, D = X.shape
        K = 3
        alpha = Dirichlet(1e-5*np.ones(K),
                          name='alpha')
        Z = Categorical(alpha,
                        plates=(N,),
                        name='z')

        mu = Gaussian(np.zeros(D), 1e-5*np.identity(D),
                      plates=(K,),
                      name='mu')
        Lambda = Wishart(D, 1e-5*np.identity(D),
                         plates=(K,),
                         name='Lambda')

        Y = Mixture(Z, Gaussian, mu, Lambda,
                    name='Y')
        Z.initialize_from_random()
        Q = VB(Y, mu, Lambda, Z, alpha)
        Y.observe(X)
        Q.update(repeat=1000)
        # print("dir(Y)", dir(Y))
        get_component_params(Y)

        if BAYESPY_PLOT:
            bpplt.gaussian_mixture_2d(Y, alpha=alpha, scale=2)
            bpplt.pyplot.show()

    if BAYESPY_PLOT:
        pass
    else:
        fig.tight_layout()
        cont = plt.show()
        plt.pop()

def get_component_params(X):
    """
    from
    bayespy.plot.py - gaussian_mixture_2d
    """
    import scipy
    from bayespy.inference.vmp.nodes.gaussian import (GaussianMoments,
                                                      GaussianWishartMoments)

    mu_Lambda = X._ensure_moments(X.parents[1], GaussianWishartMoments)

    (mu, _, Lambda, _) = mu_Lambda.get_moments()
    mu = np.linalg.solve(Lambda, mu)

    if len(mu_Lambda.plates) != 1:
        raise NotImplementedError("Not yet implemented for more plates")

    K = mu_Lambda.plates[0]

    width = np.zeros(K)
    height = np.zeros(K)
    angle = np.zeros(K)

    for k in range(K):
        m = mu[k]
        L = Lambda[k]
        (u, W) = scipy.linalg.eigh(L)
        u[0] = np.sqrt(1/u[0])
        u[1] = np.sqrt(1/u[1])
        width[k] = 2*u[0]
        height[k] = 2*u[1]
        angle[k] = np.arctan(W[0,1] / W[0,0])

    angle = 180 * angle / np.pi
    mode_height = 1 / (width * height)
    print("mu=", mu)
    print("width=", width)
    print("height=", height)
    print("angle=", angle)

    params = sorted([(m, w, h) for m, w, a in zip(mu[:,0], width, height)], key=lambda x: x[0])
    print("params=", params)

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
