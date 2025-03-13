"""
    Proportional.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from molass_legacy.KekLib.SciPyCookbook import smooth
from molass_legacy.Models.ElutionCurveModels import egha, emga

VARY_SMALL_VALUE = 1e-6

def get_curve_list(x, y, func, params):

    cy_list = []
    for p in params:
        cy = func(x, *p)
        cy_list.append(cy)

    ty = np.sum(cy_list, axis=0)
    ty[ty < VARY_SMALL_VALUE] = VARY_SMALL_VALUE

    ret_list = []
    for cy in cy_list:
        ret_list.append(y*cy/ty)

    return ret_list

def get_proportional_curves(x, y, egha_params, emga_params):
    egha_list = get_curve_list(x, y, egha, egha_params)
    emga_list = get_curve_list(x, y, emga, emga_params)
    return egha_list, emga_list
