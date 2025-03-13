"""
    SecTheory.CumsumInverse.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize

PENALTY_SCALE = 1e6

def cumsum_inverse(spline, y_values, xmin, xmax):

    def func(x):
        if len(x) > 1:
            order_penalty = min(0, np.min(x[1:] - x[:-1]))**2
        else:
            order_penalty = 0
        range_penalty = min(0, x[0] - xmin)**2 + min(0, xmax - x[-1])**2
        return np.sum((spline(x) - y_values)**2) + PENALTY_SCALE*(order_penalty + range_penalty)

    n = len(y_values)
    # x_guess = np.linspace(xmin, xmax, n + 2)[1:-1]
    x_guess = xmin*(1 - y_values) + xmax*y_values
    sol = minimize(func, x_guess)
    return sol.x
