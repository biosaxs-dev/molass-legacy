# coding: utf-8
"""
    SecTools.CorMap.SchillingDist.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np

def schilling_prob(a, b, n, p):
    r = -np.log(n*(1-p))/np.log(p)
    cdf = lambda x: np.exp(- p**x )
    return cdf(b + 1 - r) - cdf(a - r)

def schilling_pdf(n, p, x):
    y = np.zeros(len(x))
    dx = (x[1] - x[0])/2        # assuming that x is regularly divided
    for k, x_ in enumerate(x):
        y[k] = schilling_prob(x_-dx, x_+dx, n, p)
        px = x_
    return y
