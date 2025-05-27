# coding: utf-8
"""
    EmgEstimation.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
# import matplotlib.pyplot as plt
import molass_legacy.KekLib.DebugPlot as plt

def spike():
    N = 1000000

    mu = 5
    sigma = 1.5
    tau = 2

    X = np.random.normal(mu, sigma, N)
    Y = np.random.exponential(tau, N)
    Z = X + Y

    mean = np.mean(Z)
    print('mean=', mean)

    m2 = np.sum((Z - mean)**2)/N
    print('m2=', m2)

    m3 = np.sum((Z - mean)**3)/N
    print('m3=', m3)
    print('tau=', np.power(m3/2, 1/3))
    print('sigma=', np.sqrt(m2 - tau**2))
