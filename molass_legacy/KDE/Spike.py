# coding: utf-8
"""
    Pomegranate.Spike.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import time
import asyncio
import numpy as np
# from pomegranate import *
from sklearn.neighbors import KernelDensity
from molass_legacy.KekLib.SciPyCookbook import smooth
from molass_legacy.Peaks.ElutionModels import egh
import molass_legacy.KekLib.DebugPlot as plt
from Pomegranate.EghMixtureModel import EghMixtureModel
from Prob.GaussianMixture import hist_to_source

def normal_pdf(x, mu, sigma):
    return np.exp(-1/2*((x - mu)/sigma)**2)/(sigma*np.sqrt(np.pi))

def distributions_demo():
    x = np.arange(500)

    plt.push()
    fig, ax = plt.subplots()

    d = EghDistribution(200, 30, 5)
    ax.plot(x, d.probability(x))

    fig.tight_layout()
    plt.show()
    plt.pop()

def spike_demo():
    x = np.arange(500)
    y = np.zeros(len(x))
    for params in [
            # [0.6, 200, 40, 10],
            [0.6, 200, 30, 5],
            [1.0, 300, 30, 10],
            ]:
        y += egh(x, *params)

    spike_demo_impl([(x, y)], num_components=2)

def spike_demo_impl(xy_list, num_components):
    for x, y in xy_list:
        sy = smooth(y)
        X = hist_to_source(x, sy)[:,np.newaxis] 
        kde = KernelDensity(kernel='gaussian', bandwidth=20).fit(X)
        # print(dir(kde))
        log_dens = kde.score_samples(x[:,np.newaxis])
        fy = np.exp(log_dens)
        scale = np.max(y)/np.max(fy)

        plt.push()
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(x, fy*scale)
        plt.show()
        plt.pop()

def spike_demo_real(in_folder, num_components=3, correction=False, use_peaks=False):
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    sp = StandardProcedure()
    sp.load(in_folder, debug=False)
    if correction:
        sd = sp.get_corrected_sd()
    else:
        sd = sp.get_sd()
    uv_curve = sd.get_uv_curve()
    xr_curve = sd.get_xray_curve()
    xy_list = [(curve.x, curve.y) for curve in [uv_curve, xr_curve]]
    spike_demo_impl(xy_list, num_components=3)

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
