# coding: utf-8
"""
    SecTools.CorMap.SchillingDist.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np

# Notes on this approximation (Schilling 1990, eq. 7); documentation only, no behavior change.
#   - It models P(longest run <= x) as exp(-n*(1-p)*p**(x+1)).
#   - r = -log(n*(1-p))/log(p) is the TYPICAL length (the peak of the density), not the mean.
#     The mean longest run is about r + gamma/log(1/p) - 1/2 (eq. 5), which is larger,
#     especially when p is close to 1.
#   - Accuracy: max CDF error is about 0.01 for p = 0.5, but about 0.09 for p = 0.95.
#     The only caller (PairwiseMap.draw_pvalue_in_schilling) uses p = 0.5, where it is adequate.
#     For other p, compute the exact distribution (a run-length-state recursion), see
#     molass-researcher/experiments/05_schilling_distribution/05a_distribution_choice.ipynb.

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
