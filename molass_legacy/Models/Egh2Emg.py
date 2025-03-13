"""
    Egh2Emg.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.Models.ElutionCurveModels import emga

def to_emga_params(params):
    ret_params = params.copy()
    tR = params[1]
    mu = tR - params[3]
    ret_params[1] = mu
    x = np.array([tR])
    y = emga(x, *ret_params)        # note that emga returns a vector
    ret_params[0] *= params[0]/y[0]
    return ret_params
