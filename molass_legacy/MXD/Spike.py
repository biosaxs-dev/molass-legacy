# coding: utf-8
"""
    MXD.Spike.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import time
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from molass_legacy.Peaks.ElutionModels import egh
from .FastDecomposer import FastDecomposer
import molass_legacy.KekLib.DebugPlot as plt

def spike_demo():
    x = np.arange(500)
    y = np.zeros(len(x))
    for params in [
            # [0.6, 200, 40, 10],
            [0.6, 200, 30, 5],
            [1.0, 300, 30, 10],
            ]:
        y += egh(x, *params)

    spike_demo_impl(x, y, num_compoments=2)

def spike_demo_impl(x, y, num_compoments=None):
    from Prob.GaussianMixture import hist_to_source
    from Prob.EghMixture import EghMixture
    from Prob.EghMixtureUtils import get_components

    t0 = time()
    sy = hist_to_source(x, y)
    mm = EghMixture(num_compoments)
    mm.fit(np.expand_dims(sy,1))
    print("it took %.3g seconds." % (time() - t0))
    cy_list, ty = get_components(x, y, mm)

    try:
        t0 = time()
        fd = FastDecomposer(num_compoments)
        fd.fit(x, y)
        print("it took %.3g seconds." % (time() - t0))
        cy_list2, ty2 = fd.get_components()
    except:
        print("FastDecomposer failed")
        mm = EghMixture(num_compoments)
        mm.fit(np.expand_dims(sy,1))
        cy_list2, ty2 = get_components(x, y, mm)

    plt.push()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,7))
    ax1.plot(x, y)
    ax1.plot(x, ty, ':')
    for cy in cy_list:
        ax1.plot(x, cy, ':')

    ax2.plot(x, y)
    ax2.plot(x, ty2, ':')
    for cy in cy_list2:
        ax2.plot(x, cy, ':')

    fig.tight_layout()
    plt.show()
    plt.pop()

def spike_demo_real(in_folder):
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    sp = StandardProcedure()
    sp.load(in_folder, debug=False)
    sd = sp.get_sd()
    xr_curve = sd.get_xray_curve()
    spike_demo_impl(xr_curve.x, xr_curve.y, num_compoments=5)
