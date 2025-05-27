# coding: utf-8
"""
    DMM.Spike.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import time
import numpy as np
from molass_legacy.Peaks.ElutionModels import egh
import molass_legacy.KekLib.DebugPlot as plt
from .dmm import DMM
from Prob.GaussianMixture import hist_to_source

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
    k = 2

    dmm = DMM(k, sigma=None)
    sy = hist_to_source(x, y)
    print(dmm.estimate(sy))

    plt.push()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,7))
    ax1.plot(x, y)

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
