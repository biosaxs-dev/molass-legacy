# coding: utf-8
"""

    NumInverse.py

    Copyright (c) 2020, Masatsuyo Takahashi, KEK-PF

"""
import numpy as np
from scipy.optimize import minimize

def inverse_at(func, y, x_init):
    def obj_func(p):
        return (y - func(p[0]))**2

    result = minimize(obj_func, np.array([x_init]))
    return result.x
