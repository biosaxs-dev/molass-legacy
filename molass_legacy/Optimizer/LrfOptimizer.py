"""
    Optimizer.LrfOptimizer.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize

def optimize_decomposition_for_lrf(optimizer, params):

    # indeces = optimizer.get_lrfopt_indeces()
    indeces = np.array([], dtype=int)

    temp_params = params.copy()

    def non_negative_lrf_objective(p):
        temp_params[indeces] = p
        fv = optimizer.objective_func(temp_params)