# coding: utf-8
"""
    MXD.Bayesian.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import time
import numpy as np
from bayes_opt import BayesianOptimization
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
            # [0.6, 200, 30, 5],
            [1.0, 300, 30, 10],
            ]:
        y += egh(x, *params)

    spike_demo_impl(x, y, num_compoments=2)



def spike_demo_impl(x, y, num_compoments=None):

    def target(h1, m1, s1, t1):
        ty = egh(x, h1, m1, s1, t1)
        r2 = np.sum((ty - y)**2)
        return -r2

    pbounds = {
        'h1': (0, 2),
        'm1': (0, 400),
        's1': (0, 50),
        't1': (0, 20),
        }

    optimizer = BayesianOptimization(
        f=target,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=5,
    )

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
    spike_demo_impl(xr_curve.x, xr_curve.y, num_compoments=3)
