"""
    Optimizer.FvSynthesisOptimizer.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from .FvSynthesizer import synthesize
from .FvScoreConverter import convert_score

# (XR_2D_fitting, XR_LRF_residual, UV_2D_fitting, UV_LRF_residual, Guinier_deviation, Kratky_smoothness, SEC_conformance), sv0, sv1,
VALUE_TABLE = [
        # [(-1.45392, -1.19308,  -1.35939,  -0.076241,  -1.1585, -1.03118,  -0.6), 70.3, 90.],        # from 20230303/HasA
        # [(-1.22972, -0.852194, -0.562419, -0.895275,  -1.1217, -0.992915, -0.6), 71.5, 50.],        # from 20230303/HasA
        [(-0.908609, -1.10815, -0.999691, -0.0794044, -1.12317, -1.07021, -0.6), 71.1, 90],         # from 20230303/HasA
        [(-0.894142, -0.996312, -0.37274, -0.78015,   -1.11888, -1.04737,  -0.6), 71.4, 50.],       # from 20230303/HasA
    ]

def objective_func(w):
    w1 = 1 - np.sum(w)
    w_ = np.concatenate([w, [w1]])
    svdev = 0
    for v, sv0, sv1 in VALUE_TABLE:
        fv = synthesize(v, positive_elevate=3, value_weights=w_)
        sv = convert_score(fv)
        svdev += (sv - sv1)**2 + (sv - sv0)**2
    return np.log10(svdev) + max(-1.5, np.log10(np.std(w_)))
    # return np.log10(svdev)

def compute_optimized_weights():
    from molass_legacy._MOLASS.SerialSettings import get_setting

    NUM_MAJOR_SCORES = get_setting("NUM_MAJOR_SCORES")
    init_weights = np.ones(NUM_MAJOR_SCORES)/NUM_MAJOR_SCORES

    bounds = [(0, 1)] * (NUM_MAJOR_SCORES - 1)
    ret = minimize(objective_func, init_weights[0:-1], bounds=bounds)

    weights = np.concatenate([ret.x, [1 - np.sum(ret.x)]])
    # [0.13661366 0.13661366 0.15548285 0.13661366 0.13661366 0.13661366 0.16144883]

    # print(ret.fun)
    show_weights_performance(weights)

def show_weights_performance(weights):
    print(weights)
    for k, (v, sv0, sv1) in enumerate(VALUE_TABLE):
        fv = synthesize(v, positive_elevate=3)
        sv = float("%.3g" % convert_score(fv))
        print([k], sv0, sv1, sv)
