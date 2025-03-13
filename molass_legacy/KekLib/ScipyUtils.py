# coding: utf-8
"""
    ScipyUtils.py

    Copyright (c) 2018-2021, Masatsuyo Takahashi, KEK-PF
"""

import numpy as np
from scipy.special import erfcx

"""
    inverse of erfcx (scaled complementary error function)
    https://jp.mathworks.com/matlabcentral/answers/302915-inverse-of-erfcx-scaled-complementary-error-function#
"""
def erfcxinv_impl(y):
    # erfcx inverse, for y no larger than 1.
    # for y <= 1, use the large x approximation for a starting value.
    k = y <= 1
    x = np.zeros(len(y));
    x[k] = 1/(y[k]*np.sqrt(np.pi))
    # for y > 1, use exp(x^2) as a very rough approximation
    # to erfcx
    _k = y > 1
    x[_k] = -np.sqrt(np.log(y[_k]));
    for n in range(7):
        x = x - (erfcx(x) - y)/(2*x*erfcx(x) - 2/np.sqrt(np.pi))
    return x

def erfcxinv(y):
    if np.isscalar(y):
        return erfcxinv_impl(np.array([y]))[0]
    else:
        return erfcxinv_impl(y)

def get_spline(x, y, num_knots=8):
    from scipy.interpolate import LSQUnivariateSpline
    knots = np.linspace(x[0], x[-1], num_knots + 2)
    return  LSQUnivariateSpline(x, y, knots[1:-1], ext=3)
