"""
    IFT/IftOptimizeImpl.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.Models.ModelUtils import compute_cy_list

def ift_optimize(M, E, qv, x, y, model, params_array, debug=True):
    cy_list = compute_cy_list(model, x, params_array)
    C = np.array(cy_list)
    # P = MC⁺
    
    Cinv = np.linalg.pinv(C)
    P = M @ Cinv
    