# coding: utf-8
"""
    Pomegranate.Spike.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import time
import asyncio
import numpy as np
from pomegranate import *
from molass_legacy.Peaks.ElutionModels import egh
import molass_legacy.KekLib.DebugPlot as plt
from .EghMixtureModel import EghMixtureModel

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

def spike_demo_impl(xy_list, num_components=None, use_peaks=False):
    if use_peaks:
        from molass_legacy.Peaks.RobustPeaks import RobustPeaks

    models = []
    for x, y in xy_list:
        model = EghMixtureModel(x, y, num_components)
        models.append(model)

    if use_peaks:
        peaks_list = []
        for x, y in xy_list:
            rp = RobustPeaks(x, y)
            peaks_list.append(rp)

    for n in range(100):
        t0 = time()
        results = []
        for k, model in enumerate(models):
            x, y = xy_list[k]
            if n == 0:
                peaks = peaks_list[k] if use_peaks else None
                res = model.fit(peaks=peaks)
            else:
                res = model.fit()
            cy_list, ty = model.get_components(res, x, y)
            results.append((cy_list, ty))

        print("It took %.3g seconds." % (time()-t0))

        plt.push()
        fig, axes = plt.subplots(ncols=2, figsize=(14,7))

        for k, (x, y) in enumerate(xy_list):
            ax = axes[k]
            cy_list, ty = results[k]
            ax.plot(x, y)
            for cy in cy_list:
                ax.plot(x, cy, ':')
            ax.plot(x, ty, ':', color='red')

        fig.tight_layout()
        cont = plt.show()
        plt.pop()

        if not cont:
            break

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
    spike_demo_impl(xy_list, num_components=4, use_peaks=use_peaks)

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
