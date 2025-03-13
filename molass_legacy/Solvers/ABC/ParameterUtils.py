"""
    Solvers.ABC.ParameterUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from pyabc import Parameter

PARAM_KEY_FORMAT = "p%02d"

def parameter_to_vector(parameter):
    return np.array([parameter[PARAM_KEY_FORMAT % k] for k in range(len(parameter))])

def vector_to_parameter(values):
    pdict = {}
    for i, v in enumerate(values):
        pdict[PARAM_KEY_FORMAT % i] = v
    return Parameter(pdict)